"""
Shield GPT — Central Configuration
All model, training, and path settings in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROMPTS_DIR = DATA_DIR / "prompts"
CHECKPOINTS_DIR = ROOT_DIR / "model" / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"

# Auto-create directories
for d in [RAW_DIR, PROCESSED_DIR, PROMPTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Special Tokens
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = {
    "task": "<|task|>",
    "input": "<|input|>",
    "output": "<|output|>",
    "end": "<|end|>",
    "pad": "<|pad|>",
}

# Task names used in training format
TASK_NAMES = [
    "injection_detect",
    "policy_classify",
    "nl_to_policy",
    "goal_decompose",
]


# ---------------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """GPT-2 style model — ~110M parameters, optimized for 4GB VRAM."""
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 512       # Context length (tokens)
    vocab_size: int = 50304     # GPT-2 vocab (50257) padded to nearest 64
    dropout: float = 0.1
    bias: bool = False          # No bias = fewer params + cleaner
    use_gradient_checkpointing: bool = True
    rope: bool = True           # Rotary Position Embeddings


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training hyperparameters tuned for GTX 1650 (4GB VRAM)."""

    # Batch sizes
    micro_batch_size: int = 1           # What fits in VRAM
    gradient_accumulation_steps: int = 32  # Effective batch = 32
    
    # Learning rate schedule
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 50000              # Total training steps
    
    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    use_8bit_optimizer: bool = True     # 8-bit Adam via bitsandbytes
    
    # Precision
    dtype: str = "float16"              # FP16 for GTX 1650 (no BF16 support)
    
    # Checkpointing
    checkpoint_every: int = 500         # Save checkpoint every N steps
    eval_every: int = 250               # Run validation every N steps
    log_every: int = 10                 # Log loss every N steps
    
    # Data
    train_file: str = str(PROCESSED_DIR / "train.bin")
    val_file: str = str(PROCESSED_DIR / "val.bin")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def torch_dtype(self):
        return torch.float16 if self.dtype == "float16" else torch.bfloat16


# ---------------------------------------------------------------------------
# Data Generation Config
# ---------------------------------------------------------------------------
@dataclass
class DataGenConfig:
    """Settings for data generated via ChatGPT web interface."""
    
    # Target examples per task
    target_examples: dict = field(default_factory=lambda: {
        "injection_detect": 5000,
        "policy_classify": 3000,
        "nl_to_policy": 3000,
        "goal_decompose": 2000,
    })
    
    # Validation split ratio
    val_ratio: float = 0.05
    
    # Quality filters
    min_input_length: int = 5           # Minimum input chars
    max_input_length: int = 2000        # Maximum input chars
    min_output_length: int = 1          # Minimum output chars
    max_output_length: int = 2000       # Maximum output chars


# ---------------------------------------------------------------------------
# Convenience: default instances
# ---------------------------------------------------------------------------
model_config = ModelConfig()
train_config = TrainConfig()
data_config = DataGenConfig()


def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("SHIELD GPT — Configuration")
    print("=" * 60)
    
    print(f"\n📐 Model:")
    print(f"   Layers: {model_config.n_layer}")
    print(f"   Heads: {model_config.n_head}")
    print(f"   Embedding dim: {model_config.n_embd}")
    print(f"   Context length: {model_config.block_size}")
    print(f"   Vocab size: {model_config.vocab_size}")
    est_params = (
        model_config.vocab_size * model_config.n_embd +  # token embeddings
        model_config.n_layer * (
            4 * model_config.n_embd * model_config.n_embd +  # attn QKV + proj
            2 * model_config.n_embd * (4 * model_config.n_embd)  # MLP
        ) +
        model_config.vocab_size * model_config.n_embd  # lm_head (tied or not)
    )
    print(f"   Est. parameters: ~{est_params / 1e6:.0f}M")
    print(f"   Est. VRAM (FP16): ~{est_params * 2 / 1e9:.2f} GB (weights only)")
    
    print(f"\n🏋️ Training:")
    print(f"   Micro batch: {train_config.micro_batch_size}")
    print(f"   Gradient accum: {train_config.gradient_accumulation_steps}")
    print(f"   Effective batch: {train_config.effective_batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Max steps: {train_config.max_steps}")
    print(f"   8-bit optimizer: {train_config.use_8bit_optimizer}")
    print(f"   Gradient checkpointing: {model_config.use_gradient_checkpointing}")
    print(f"   Precision: {train_config.dtype}")
    
    print(f"\n💾 Paths:")
    print(f"   Data: {DATA_DIR}")
    print(f"   Checkpoints: {CHECKPOINTS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    
    device = train_config.device
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n🖥️ GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        print(f"\n⚠️ No GPU detected — training will be VERY slow on CPU")
    
    print("=" * 60)


if __name__ == "__main__":
    print_config()
