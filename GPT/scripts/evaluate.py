"""
Shield GPT — Evaluation Script

Evaluates the trained model on held-out validation data and per-task metrics.

Usage:
  python scripts/evaluate.py                  # Evaluate best checkpoint
  python scripts/evaluate.py --checkpoint step_5000  # Evaluate specific checkpoint
  python scripts/evaluate.py --interactive    # Interactive test mode
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import model_config, CHECKPOINTS_DIR, PROCESSED_DIR
from model.architecture import build_model
from model.tokenizer_config import get_tokenizer, SPECIAL_TOKENS


def load_model(checkpoint_name="best"):
    """Load a trained model from checkpoint."""
    ckpt_path = CHECKPOINTS_DIR / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    # Build model with saved config
    cfg = model_config
    if "model_config" in ckpt:
        from config import ModelConfig
        saved = ckpt["model_config"]
        cfg = ModelConfig(**saved)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    print(f"✅ Model loaded from step {step} (loss: {loss})")
    
    return model, device


def run_inference(model, device, task: str, input_text: str, 
                  max_tokens=200, temperature=0.3) -> str:
    """Run inference on a single input."""
    tokenizer = get_tokenizer()
    
    # Format prompt (without output — model generates it)
    prompt = f"<|task|>{task}<|input|>{input_text}<|output|>"
    input_ids = tokenizer.encode(prompt)
    
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output_ids = model.generate(idx, max_new_tokens=max_tokens, 
                                     temperature=temperature)
    
    # Decode and extract output
    full_text = tokenizer.decode(output_ids[0].tolist())
    
    # Extract text between <|output|> and <|end|>
    if "<|output|>" in full_text:
        output = full_text.split("<|output|>")[-1]
        if "<|end|>" in output:
            output = output.split("<|end|>")[0]
        return output.strip()
    
    return full_text.strip()


def evaluate_task(model, device, task: str, examples: list[dict]) -> dict:
    """Evaluate model on a set of examples for one task."""
    correct = 0
    total = len(examples)
    results = []
    
    for ex in examples:
        prediction = run_inference(model, device, task, ex["input"])
        expected = ex["output"]
        
        # For classification tasks, exact match
        is_correct = prediction.strip().lower() == expected.strip().lower()
        if is_correct:
            correct += 1
        
        results.append({
            "input": ex["input"][:80],
            "expected": expected,
            "predicted": prediction,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    return {
        "task": task,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "examples": results[:10],  # Show first 10
    }


def interactive_mode(model, device):
    """Interactive testing mode."""
    tokenizer = get_tokenizer()
    
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Tasks: injection_detect, policy_classify, nl_to_policy, goal_decompose")
    print("Type 'quit' to exit\n")
    
    while True:
        task = input("Task> ").strip()
        if task == "quit":
            break
        if task not in ("injection_detect", "policy_classify", "nl_to_policy", "goal_decompose"):
            print(f"  Unknown task. Use: injection_detect, policy_classify, nl_to_policy, goal_decompose")
            continue
        
        inp = input("Input> ").strip()
        if not inp:
            continue
        
        output = run_inference(model, device, task, inp)
        print(f"Output> {output}\n")


def main():
    parser = argparse.ArgumentParser(description="Shield GPT Evaluation")
    parser.add_argument("--checkpoint", default="best",
                       help="Checkpoint name (default: best)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive test mode")
    parser.add_argument("--n-examples", type=int, default=50,
                       help="Number of validation examples to test per task")
    args = parser.parse_args()
    
    # Load model
    result = load_model(args.checkpoint)
    if result is None:
        return
    model, device = result
    
    if args.interactive:
        interactive_mode(model, device)
        return
    
    # Load validation examples from raw data
    from config import RAW_DIR, TASK_NAMES
    
    print("\n📊 EVALUATION RESULTS")
    print("=" * 60)
    
    for task in TASK_NAMES:
        jsonl_file = RAW_DIR / f"{task}.jsonl"
        if not jsonl_file.exists():
            print(f"\n⚠️  {task}: no validation data found")
            continue
        
        # Load examples
        examples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                # Map label → output for classification tasks
                if "label" in ex:
                    ex["output"] = ex["label"]
                examples.append(ex)
        
        # Take last N as validation (they weren't in training if we split correctly)
        val_examples = examples[-args.n_examples:]
        
        # Evaluate
        result = evaluate_task(model, device, task, val_examples)
        
        print(f"\n📋 {task}")
        print(f"   Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        
        # Show some examples
        print(f"   Sample predictions:")
        for ex in result["examples"][:5]:
            status = "✅" if ex["correct"] else "❌"
            print(f"     {status} Input: {ex['input'][:50]}...")
            print(f"        Expected: {ex['expected'][:50]}")
            print(f"        Got:      {ex['predicted'][:50]}")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
