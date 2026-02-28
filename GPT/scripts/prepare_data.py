"""
Shield GPT — Data Engineering Pipeline

Transforms raw JSONL data into tokenized, shuffled, binary training files.

Pipeline:
  1. Load raw JSONL files (one per task)
  2. Clean + validate
  3. Deduplicate (exact match)
  4. Format into multi-task training format
  5. Tokenize with tiktoken
  6. Shuffle
  7. Split into train/val
  8. Save as memory-mapped binary files

Usage:
  python scripts/prepare_data.py
"""

import json
import os
import sys
import hashlib
import random
import struct
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, data_config, model_config, TASK_NAMES
from model.tokenizer_config import get_tokenizer


def load_raw_data() -> dict[str, list[dict]]:
    """Load all raw JSONL files."""
    data = {}
    for task in TASK_NAMES:
        jsonl_file = RAW_DIR / f"{task}.jsonl"
        if not jsonl_file.exists():
            print(f"  ⚠️  {task}: no data file found ({jsonl_file})")
            data[task] = []
            continue
        
        examples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        print(f"  📂 {task}: {len(examples)} raw examples loaded")
        data[task] = examples
    
    return data


def clean_examples(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Clean and validate examples."""
    cleaned = {}
    cfg = data_config
    
    for task, examples in data.items():
        valid = []
        for ex in examples:
            # Check required fields
            if task in ("injection_detect", "policy_classify"):
                if "input" not in ex or "label" not in ex:
                    continue
                inp = str(ex["input"]).strip()
                out = str(ex["label"]).strip().lower()
                
                # Validate labels
                if task == "injection_detect" and out not in ("safe", "injection"):
                    continue
                if task == "policy_classify" and out not in ("allow", "deny", "escalate", "require_approval"):
                    continue
                    
            elif task in ("nl_to_policy", "goal_decompose"):
                if "input" not in ex or "output" not in ex:
                    continue
                inp = str(ex["input"]).strip()
                out = str(ex["output"]).strip()
                
                # Validate JSON output
                try:
                    json.loads(out) if isinstance(out, str) else out
                except (json.JSONDecodeError, TypeError):
                    # Try to fix common issues
                    if isinstance(out, (dict, list)):
                        out = json.dumps(out, ensure_ascii=False)
                    else:
                        continue
            else:
                continue
            
            # Length checks
            if len(inp) < cfg.min_input_length or len(inp) > cfg.max_input_length:
                continue
            if len(out) < cfg.min_output_length or len(out) > cfg.max_output_length:
                continue
            
            valid.append({"input": inp, "output": out})
        
        print(f"  🧹 {task}: {len(valid)} valid after cleaning (dropped {len(examples) - len(valid)})")
        cleaned[task] = valid
    
    return cleaned


def deduplicate(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Remove exact duplicates within each task."""
    deduped = {}
    
    for task, examples in data.items():
        seen = set()
        unique = []
        for ex in examples:
            key = hashlib.md5((ex["input"] + ex["output"]).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        
        removed = len(examples) - len(unique)
        if removed > 0:
            print(f"  🔄 {task}: removed {removed} duplicates, {len(unique)} remaining")
        else:
            print(f"  🔄 {task}: no duplicates found, {len(unique)} examples")
        deduped[task] = unique
    
    return deduped


def format_and_tokenize(data: dict[str, list[dict]]) -> list[list[int]]:
    """Format into multi-task training format and tokenize."""
    tokenizer = get_tokenizer()
    all_sequences = []
    task_counts = Counter()
    
    for task, examples in data.items():
        for ex in examples:
            tokens = tokenizer.encode_example(task, ex["input"], ex["output"])
            
            # Skip sequences that are too long
            if len(tokens) > model_config.block_size:
                continue
            
            all_sequences.append(tokens)
            task_counts[task] += 1
    
    print(f"\n  📊 Tokenized sequences per task:")
    for task, count in task_counts.items():
        print(f"     {task}: {count}")
    print(f"     Total: {len(all_sequences)}")
    
    # Print token length stats
    lengths = [len(s) for s in all_sequences]
    print(f"\n  📏 Sequence lengths:")
    print(f"     Min: {min(lengths)}, Max: {max(lengths)}, "
          f"Mean: {sum(lengths)/len(lengths):.0f}, Median: {sorted(lengths)[len(lengths)//2]}")
    
    return all_sequences


def create_binary_files(sequences: list[list[int]], val_ratio: float = 0.05):
    """
    Create memory-mapped binary files for training.
    
    Format: sequences are concatenated end-to-end, separated by <|end|> tokens.
    Stored as uint16 numpy arrays (token IDs fit in 16 bits).
    """
    # Shuffle
    random.shuffle(sequences)
    
    # Split train/val
    n_val = max(1, int(len(sequences) * val_ratio))
    val_sequences = sequences[:n_val]
    train_sequences = sequences[n_val:]
    
    print(f"\n  📦 Split: {len(train_sequences)} train, {len(val_sequences)} val")
    
    # Concatenate into flat arrays
    def concat_sequences(seqs):
        all_tokens = []
        for s in seqs:
            all_tokens.extend(s)
        return np.array(all_tokens, dtype=np.uint16)
    
    train_tokens = concat_sequences(train_sequences)
    val_tokens = concat_sequences(val_sequences)
    
    # Save
    train_file = PROCESSED_DIR / "train.bin"
    val_file = PROCESSED_DIR / "val.bin"
    
    train_tokens.tofile(str(train_file))
    val_tokens.tofile(str(val_file))
    
    print(f"  💾 Saved: {train_file} ({len(train_tokens):,} tokens, "
          f"{train_tokens.nbytes / 1e6:.1f} MB)")
    print(f"  💾 Saved: {val_file} ({len(val_tokens):,} tokens, "
          f"{val_tokens.nbytes / 1e6:.1f} MB)")
    
    # Save metadata
    meta = {
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "block_size": model_config.block_size,
        "vocab_size": model_config.vocab_size,
    }
    with open(PROCESSED_DIR / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    return meta


def main():
    print("=" * 60)
    print("SHIELD GPT — DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    print("\n1️⃣  Loading raw data...")
    data = load_raw_data()
    
    total = sum(len(v) for v in data.values())
    if total == 0:
        print("\n❌ No data found! Generate data first:")
        print("   python scripts/generate_data.py --show-prompts")
        return
    
    print(f"\n2️⃣  Cleaning & validating...")
    data = clean_examples(data)
    
    print(f"\n3️⃣  Deduplicating...")
    data = deduplicate(data)
    
    print(f"\n4️⃣  Formatting & tokenizing...")
    sequences = format_and_tokenize(data)
    
    if not sequences:
        print("\n❌ No valid sequences after processing!")
        return
    
    print(f"\n5️⃣  Creating binary training files...")
    meta = create_binary_files(sequences, val_ratio=data_config.val_ratio)
    
    print(f"\n{'=' * 60}")
    print("✅ DATA PREPARATION COMPLETE!")
    print(f"   Train: {meta['train_tokens']:,} tokens ({meta['train_sequences']} sequences)")
    print(f"   Val:   {meta['val_tokens']:,} tokens ({meta['val_sequences']} sequences)")
    print(f"\n   Next step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
