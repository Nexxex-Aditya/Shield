"""
Shield GPT — Tokenizer Configuration

Uses tiktoken (GPT-2 BPE tokenizer) + custom special tokens for multi-task format.
"""

import tiktoken
import numpy as np

# Shield's special tokens (added on top of GPT-2 vocabulary)
SPECIAL_TOKENS = {
    "<|task|>": 50257,
    "<|input|>": 50258,
    "<|output|>": 50259,
    "<|end|>": 50260,
    "<|pad|>": 50261,
}

# Reverse lookup
SPECIAL_TOKEN_IDS = {v: k for k, v in SPECIAL_TOKENS.items()}


class ShieldTokenizer:
    """
    Tokenizer for Shield GPT training.
    
    Wraps tiktoken's GPT-2 tokenizer and adds special task tokens.
    """
    
    def __init__(self):
        self._base = tiktoken.get_encoding("gpt2")
        self.special_tokens = SPECIAL_TOKENS
        self.pad_token_id = SPECIAL_TOKENS["<|pad|>"]
        self.end_token_id = SPECIAL_TOKENS["<|end|>"]
        self.vocab_size = 50304  # Padded to nearest 64 for GPU efficiency
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, handling special tokens."""
        tokens = []
        remaining = text
        
        while remaining:
            # Check if remaining text starts with a special token
            found_special = False
            for token_str, token_id in self.special_tokens.items():
                if remaining.startswith(token_str):
                    tokens.append(token_id)
                    remaining = remaining[len(token_str):]
                    found_special = True
                    break
            
            if not found_special:
                # Find the next special token position
                next_special_pos = len(remaining)
                for token_str in self.special_tokens:
                    pos = remaining.find(token_str)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
                
                # Encode the text chunk before the next special token
                chunk = remaining[:next_special_pos]
                if chunk:
                    tokens.extend(self._base.encode(chunk))
                remaining = remaining[next_special_pos:]
        
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        result = []
        regular_ids = []
        
        for tid in token_ids:
            if tid in SPECIAL_TOKEN_IDS:
                # Flush any accumulated regular tokens
                if regular_ids:
                    result.append(self._base.decode(regular_ids))
                    regular_ids = []
                result.append(SPECIAL_TOKEN_IDS[tid])
            elif tid < 50257:  # Regular GPT-2 token
                regular_ids.append(tid)
            # Skip padding and unknown tokens
        
        if regular_ids:
            result.append(self._base.decode(regular_ids))
        
        return "".join(result)
    
    def format_example(self, task: str, input_text: str, output_text: str) -> str:
        """Format a training example in the multi-task format."""
        return (
            f"<|task|>{task}"
            f"<|input|>{input_text}"
            f"<|output|>{output_text}"
            f"<|end|>"
        )
    
    def encode_example(self, task: str, input_text: str, output_text: str) -> list[int]:
        """Encode a complete training example."""
        formatted = self.format_example(task, input_text, output_text)
        return self.encode(formatted)


# Singleton
_tokenizer = None

def get_tokenizer() -> ShieldTokenizer:
    """Get the global tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ShieldTokenizer()
    return _tokenizer


if __name__ == "__main__":
    tok = get_tokenizer()
    
    # Test encoding/decoding
    example = tok.format_example(
        "injection_detect",
        "Please search for quarterly reports",
        "safe"
    )
    print(f"Formatted: {example}")
    
    ids = tok.encode(example)
    print(f"Token IDs ({len(ids)} tokens): {ids[:20]}...")
    
    decoded = tok.decode(ids)
    print(f"Decoded: {decoded}")
    
    assert decoded == example, "Round-trip failed!"
    print("\n✅ Tokenizer working correctly!")
    print(f"   Vocab size: {tok.vocab_size}")
    print(f"   Special tokens: {len(tok.special_tokens)}")
