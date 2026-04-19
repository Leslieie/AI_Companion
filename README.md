# Interactive Character Companion Agent

A text-only AI companion for emotional support and companionship chat, built on a **stateful companion architecture**. Instead of treating each conversation turn independently, the system maintains relationship state (affection, trust, intimacy, mood), retrieves relevant memories from past interactions, and selects an interaction policy before generating each reply. The pipeline follows: perceive user emotion → update relationship state → retrieve memory → select response policy → generate reply, using Qwen2.5-1.5B-Instruct as the base model with optional LoRA/QLoRA fine-tuning.

## Team

- Lanshun Yuan
- Hengkai Zheng
- Dongdong Pan

## Course

95891 – Introduction to Artificial Intelligence (CMU)

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run baseline inference (downloads Qwen2.5-1.5B-Instruct on first run)
python -m src.inference.generate

# Run the interactive CLI companion
python -m src.app
```

**Windows with non-English locales:** If you hit a `UnicodeDecodeError` during training, set UTF-8 mode before running:

```powershell
# PowerShell
$env:PYTHONUTF8="1"

# Or set it permanently (cmd)
set PYTHONUTF8=1
```

## Model

- Primary: `Qwen/Qwen2.5-1.5B-Instruct`
- Alternate: `Qwen/Qwen2.5-3B-Instruct`

## Repo Structure

```
├── configs/            # YAML configuration files
│   ├── model.yaml          # Model name and generation parameters
│   ├── prompts.yaml        # Persona text and system prompt template
│   ├── state_rules.yaml    # State update thresholds
│   └── training_sft.yaml   # SFT training hyperparameters
├── data/               # Training and evaluation data
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and formatted data
│   ├── splits/             # Train/val/test splits
│   └── annotations/        # Human annotation files
├── docs/               # Design and specification documents
│   ├── persona.md          # Companion personality definition
│   ├── state_schema.md     # Relationship state variables
│   ├── memory_schema.md    # Memory system design
│   └── annotation_guidelines.md  # SFT data annotation rules
├── src/                # Source code
│   ├── app.py              # Interactive CLI companion
│   ├── inference/          # Model loading and generation
│   ├── modules/            # State, memory, policy, emotion modules
│   ├── training/           # SFT training and evaluation scripts
│   └── utils/              # Shared utilities
├── notebooks/          # Jupyter notebooks for exploration
└── outputs/            # Generated outputs
    ├── checkpoints/        # Model checkpoints
    ├── logs/               # Training logs
    └── eval/               # Evaluation results
```
