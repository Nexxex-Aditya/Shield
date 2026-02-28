"""
Shield GPT — Model Architecture

GPT-2 style transformer decoder optimized for 4GB VRAM:
- RMSNorm (faster than LayerNorm)
- Rotary Position Embeddings (RoPE)
- Built-in gradient checkpointing
- FP16-safe design
- ~110M parameters with default config

Based on nanoGPT with key optimizations for memory efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm — faster than LayerNorm, no mean computation
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the rotation frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to queries/keys."""
    # x shape: (batch, n_head, seq_len, head_dim)
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        
        # QKV projection in one matrix for efficiency
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.use_rope = config.rope
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, C = x.size()
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope and rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Combine heads
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU variant — used by LLaMA/modern models)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * 4 * 2 / 3)  # SwiGLU reduces hidden
        hidden_dim = ((hidden_dim + 63) // 64) * 64   # Round to nearest 64
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU activation
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.ff = FeedForward(config)
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Shield GPT Model
# ---------------------------------------------------------------------------
class ShieldGPT(nn.Module):
    """
    The core language model for Shield.
    
    GPT-2 architecture with modern optimizations:
    - RMSNorm, RoPE, SwiGLU FFN
    - Built-in gradient checkpointing for memory savings
    - Weight tying (embedding + lm_head share weights)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (no positional embeddings — using RoPE instead)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final norm + language model head
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying — saves memory
        self.tok_emb.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        if config.rope:
            cos, sin = precompute_rope_freqs(
                config.n_embd // config.n_head,
                config.block_size
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        
        # Gradient checkpointing flag
        self._use_gradient_checkpointing = config.use_gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = n_params - self.tok_emb.weight.numel()
        print(f"ShieldGPT initialized: {n_params/1e6:.1f}M params "
              f"({n_params_no_emb/1e6:.1f}M non-embedding)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        )
        
        # Token embeddings
        x = self.tok_emb(idx)
        x = self.dropout(x)
        
        # RoPE buffers
        rope_cos = self.rope_cos if self.config.rope else None
        rope_sin = self.rope_sin if self.config.rope else None
        
        # Transformer blocks (with optional gradient checkpointing)
        for block in self.blocks:
            if self._use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(
                    block, x, rope_cos, rope_sin,
                    use_reentrant=False
                )
            else:
                x = block(x, rope_cos, rope_sin)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding in loss
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """
        Autoregressive generation.
        
        Args:
            idx: (batch, seq_len) conditioning tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else \
                       idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop at <|end|> token
            if idx_next.item() == 50260:  # <|end|> token ID
                break
        
        return idx
    
    def get_num_params(self):
        """Return total and non-embedding parameter count."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        return total, total - emb


def build_model(config: ModelConfig = None) -> ShieldGPT:
    """Build a ShieldGPT model from config."""
    if config is None:
        config = ModelConfig()
    return ShieldGPT(config)


if __name__ == "__main__":
    # Quick test
    config = ModelConfig()
    model = build_model(config)
    
    total, non_emb = model.get_num_params()
    print(f"\nParameter count: {total:,} total, {non_emb:,} non-embedding")
    print(f"Memory (FP16): {total * 2 / 1e9:.3f} GB")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (1, 64))
    logits, loss = model(x, targets=x)
    print(f"\nForward pass: logits shape = {logits.shape}, loss = {loss.item():.4f}")
    print("✅ Model architecture OK!")
