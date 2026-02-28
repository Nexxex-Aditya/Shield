"""
Shield GPT — Training Script

Trains the ShieldGPT model with all optimizations for 4GB VRAM.

Features:
  - FP16 mixed precision
  - Gradient checkpointing (saves ~60% activation memory)
  - Gradient accumulation (effective batch size = 32)
  - 8-bit Adam optimizer (75% less optimizer memory)
  - Auto-checkpoint every N steps
  - Graceful Ctrl+C: saves checkpoint before exit
  - Resume from any checkpoint

Usage:
  python train.py                    # Start training from scratch
  python train.py --resume           # Resume from latest checkpoint
  python train.py --resume step_5000 # Resume from specific checkpoint
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    model_config, train_config, CHECKPOINTS_DIR, LOGS_DIR,
    PROCESSED_DIR, print_config,
)
from model.architecture import ShieldGPT, build_model


# ---------------------------------------------------------------------------
# Dataset: memory-mapped binary token files
# ---------------------------------------------------------------------------
class TokenDataset:
    """Memory-mapped dataset for efficient training."""
    
    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        print(f"  Dataset: {self.n_tokens:,} tokens from {data_path}")
    
    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of (input, target) pairs."""
        ix = torch.randint(self.n_tokens - self.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Learning Rate Schedule: cosine with warmup
# ---------------------------------------------------------------------------
def get_lr(step: int, config=None) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if config is None:
        config = train_config
    
    # Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    
    # Cosine decay
    if step >= config.max_steps:
        return config.min_lr
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scaler, step, loss, path):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "loss": loss,
        "model_config": {
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "block_size": model_config.block_size,
            "vocab_size": model_config.vocab_size,
            "dropout": model_config.dropout,
            "bias": model_config.bias,
            "rope": model_config.rope,
        },
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved: {path} (step {step}, loss {loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float('inf'))
    print(f"  📂 Checkpoint loaded: {path} (step {step}, loss {loss:.4f})")
    return step, loss


def find_latest_checkpoint() -> str:
    """Find the most recent checkpoint."""
    checkpoints = list(CHECKPOINTS_DIR.glob("step_*.pt"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
    return str(checkpoints[-1])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, resume_from=None):
        self.device = train_config.device
        self._interrupted = False
        
        # Setup signal handler for graceful Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Build model
        print("\n🏗️  Building model...")
        self.model = build_model(model_config).to(self.device)
        
        # Setup optimizer
        print("🔧 Setting up optimizer...")
        self._setup_optimizer()
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.device == "cuda" else None
        self.autocast_ctx = lambda: autocast(dtype=train_config.torch_dtype)
        
        # Load data
        print("📂 Loading datasets...")
        self.train_data = TokenDataset(train_config.train_file, model_config.block_size)
        self.val_data = TokenDataset(train_config.val_file, model_config.block_size)
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint
        if resume_from:
            if resume_from == "latest":
                ckpt_path = find_latest_checkpoint()
            else:
                ckpt_path = str(CHECKPOINTS_DIR / f"{resume_from}.pt")
            
            if ckpt_path and os.path.exists(ckpt_path):
                self.step, _ = load_checkpoint(
                    ckpt_path, self.model, self.optimizer, self.scaler
                )
            else:
                print(f"  ⚠️  Checkpoint not found: {ckpt_path}, starting from scratch")
        
        # Setup logging
        self.log_file = LOGS_DIR / "training_log.csv"
        if not self.log_file.exists() or self.step == 0:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "val_loss", "lr", "time_ms", "tokens_per_sec"])
    
    def _setup_optimizer(self):
        """Setup optimizer with optional 8-bit mode."""
        # Separate weight decay and non-weight-decay params
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'weight' in name and 'ln' not in name and 'norm' not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        if train_config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.Adam8bit(
                    optim_groups,
                    lr=train_config.learning_rate,
                    betas=(train_config.beta1, train_config.beta2),
                )
                print("  ✅ Using 8-bit Adam optimizer (75% memory savings)")
            except ImportError:
                print("  ⚠️  bitsandbytes not installed, falling back to regular Adam")
                self.optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=train_config.learning_rate,
                    betas=(train_config.beta1, train_config.beta2),
                )
        else:
            self.optimizer = torch.optim.AdamW(
                optim_groups,
                lr=train_config.learning_rate,
                betas=(train_config.beta1, train_config.beta2),
            )
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self._interrupted:
            print("\n\n⚡ Force quit!")
            sys.exit(1)
        
        self._interrupted = True
        print("\n\n⏸️  Interrupt received! Saving checkpoint before exit...")
    
    @torch.no_grad()
    def evaluate(self, n_batches=20):
        """Run validation."""
        self.model.eval()
        losses = []
        for _ in range(n_batches):
            x, y = self.val_data.get_batch(train_config.micro_batch_size, self.device)
            with self.autocast_ctx():
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)
    
    def train(self):
        """Main training loop."""
        self.model.train()
        cfg = train_config
        
        print(f"\n{'=' * 60}")
        print("🚀 TRAINING STARTED")
        print(f"   Starting from step: {self.step}")
        print(f"   Target steps: {cfg.max_steps}")
        print(f"   Effective batch size: {cfg.effective_batch_size}")
        print(f"   Checkpoint every: {cfg.checkpoint_every} steps")
        print(f"   Press Ctrl+C to pause (checkpoint will be saved)")
        print(f"{'=' * 60}\n")
        
        # Training loop
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        t_start = time.time()
        
        while self.step < cfg.max_steps:
            # Check for interrupt
            if self._interrupted:
                break
            
            # Gradient accumulation loop
            for micro_step in range(cfg.gradient_accumulation_steps):
                x, y = self.train_data.get_batch(cfg.micro_batch_size, self.device)
                
                with self.autocast_ctx():
                    _, loss = self.model(x, y)
                    loss = loss / cfg.gradient_accumulation_steps
                
                accumulated_loss += loss.item()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            
            # Update learning rate
            lr = get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            self.step += 1
            
            # Logging
            if self.step % cfg.log_every == 0:
                t_now = time.time()
                dt = t_now - t_start
                tokens_per_sec = (cfg.log_every * cfg.effective_batch_size * model_config.block_size) / dt
                
                print(f"  step {self.step:>6d} | loss {accumulated_loss:.4f} | "
                      f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                      f"{dt*1000:.0f}ms")
                
                accumulated_loss = 0.0
                t_start = t_now
            
            # Evaluation
            if self.step % cfg.eval_every == 0:
                val_loss = self.evaluate()
                print(f"  📊 Validation loss: {val_loss:.4f} "
                      f"{'(new best!)' if val_loss < self.best_val_loss else ''}")
                
                # Log to CSV
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.step, accumulated_loss, val_loss, lr, "", ""])
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        self.model, self.optimizer, self.scaler,
                        self.step, val_loss,
                        str(CHECKPOINTS_DIR / "best.pt")
                    )
            
            # Periodic checkpoint
            if self.step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scaler,
                    self.step, accumulated_loss,
                    str(CHECKPOINTS_DIR / f"step_{self.step}.pt")
                )
        
        # Final checkpoint (or interrupt checkpoint)
        save_checkpoint(
            self.model, self.optimizer, self.scaler,
            self.step, accumulated_loss,
            str(CHECKPOINTS_DIR / f"step_{self.step}.pt")
        )
        
        if self._interrupted:
            print(f"\n⏸️  Training PAUSED at step {self.step}")
            print(f"   Resume with: python train.py --resume")
        else:
            print(f"\n🎉 Training COMPLETE at step {self.step}!")
        
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Checkpoints saved in: {CHECKPOINTS_DIR}")


# ---------------------------------------------------------------------------
# VRAM check
# ---------------------------------------------------------------------------
def check_vram():
    """Quick VRAM availability check."""
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected! Training will be extremely slow.")
        return False
    
    gpu = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"🖥️  GPU: {gpu}")
    print(f"   VRAM: {total_vram:.1f} GB")
    
    if total_vram < 3.5:
        print("⚠️  Less than 3.5 GB VRAM detected. Training may fail.")
        print("   Consider reducing model size in config.py")
    
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Shield GPT Training")
    parser.add_argument("--resume", nargs='?', const="latest", default=None,
                       help="Resume from checkpoint (default: latest)")
    parser.add_argument("--check", action="store_true",
                       help="Check configuration without training")
    args = parser.parse_args()
    
    print_config()
    
    if args.check:
        check_vram()
        print("\n✅ Configuration check complete. Ready to train!")
        return
    
    # Check data exists
    if not os.path.exists(train_config.train_file):
        print(f"\n❌ Training data not found: {train_config.train_file}")
        print("   Run these steps first:")
        print("   1. python scripts/generate_data.py --show-prompts")
        print("   2. Collect data from ChatGPT")
        print("   3. python scripts/generate_data.py --parse all")
        print("   4. python scripts/prepare_data.py")
        return
    
    check_vram()
    
    # Create trainer and start
    trainer = Trainer(resume_from=args.resume)
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n💥 OUT OF MEMORY!")
            print(f"   Your GPU ran out of VRAM. Try these fixes in config.py:")
            print(f"   1. Reduce block_size (currently {model_config.block_size})")
            print(f"   2. Reduce n_layer (currently {model_config.n_layer})")
            print(f"   3. Reduce n_embd (currently {model_config.n_embd})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise


if __name__ == "__main__":
    main()
