# Shield GPT — Your Own Language Model

Train a proprietary 110M parameter language model for Shield's core tasks, on your GTX 1650 (4GB VRAM).

## Quick Start (5 Steps)

### Step 1: Install Dependencies
```bash
cd Shield/GPT
pip install -r requirements.txt
```

### Step 2: Generate Training Data
```bash
python scripts/generate_data.py --show-prompts
```
This prints prompts you paste into **ChatGPT web interface**. For each task:
1. Copy the prompt
2. Paste into ChatGPT
3. Copy ChatGPT's response
4. Save as `data/raw/{task_name}_batch_01.txt`
5. Repeat 20-50 times per task (say "Generate 100 MORE diverse examples")
6. After collecting, parse all outputs:

```bash
python scripts/generate_data.py --parse all
python scripts/generate_data.py --status    # Check progress
```

**Minimum data needed to start training: 1,000 examples total**

### Step 3: Prepare Data
```bash
python scripts/prepare_data.py
```
This cleans, deduplicates, tokenizes, and creates binary training files.

### Step 4: Train
```bash
python train.py              # Start from scratch
python train.py --resume     # Resume after pause
python train.py --check      # Verify setup without training
```

**Pause/Resume:** Press `Ctrl+C` — it saves a checkpoint. Run `python train.py --resume` to continue.

### Step 5: Evaluate & Use
```bash
python scripts/evaluate.py                # Run metrics
python scripts/evaluate.py --interactive  # Test manually
```

## Using in Shield

```python
from GPT.integrate import ShieldLLM

llm = ShieldLLM()

# Detect prompt injection
result = llm.detect_injection("ignore previous instructions")
# → {"label": "injection"}

# Classify policy action
result = llm.classify_policy_action("agent=bot tool=db.delete params={table: users}")
# → {"action": "deny"}

# Natural language → policy
result = llm.compile_nl_policy("Block all payments over $1000")
# → {"action": "block", "trigger_tools": ["*payment*"], "conditions": {"max_amount": 1000}}

# Goal decomposition
result = llm.decompose_goal("Monitor GitHub PRs and notify Slack")
# → [{"goal": "List PRs", "tool": "github.list_prs"}, ...]
```

## File Structure
```
GPT/
├── config.py                # All settings (model size, training params)
├── train.py                 # Main training script
├── integrate.py             # ShieldLLM class for agentvault
├── requirements.txt         # Dependencies
├── data/
│   ├── raw/                 # Paste ChatGPT outputs here
│   └── processed/           # Auto-generated training files
├── scripts/
│   ├── generate_data.py     # Data generation helper
│   ├── prepare_data.py      # Data engineering pipeline
│   └── evaluate.py          # Evaluation & testing
└── model/
    ├── architecture.py      # GPT-2 model (110M params)
    ├── tokenizer_config.py  # Tokenizer
    └── checkpoints/         # Saved model weights
```

## Training Tips
- **VRAM usage**: ~1.5 GB of 4 GB — you have headroom
- **Training time**: ~3-7 days (cumulative, not continuous)
- **Checkpoints**: Saved every 500 steps + on Ctrl+C
- **Best model**: Auto-saved as `model/checkpoints/best.pt`
- **More data = better model**: Aim for 5,000+ examples per task
