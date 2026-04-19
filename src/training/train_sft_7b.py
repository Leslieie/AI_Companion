"""SFT training script with LoRA for Qwen2.5-7B-Instruct.

Usage:
    python -m src.training.train_sft_7b

Uses the same training data as the 1.5B pipeline but targets the 7B model
with gradient checkpointing enabled to reduce VRAM usage.
"""

import os
os.environ["PYTHONUTF8"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/processed/train_sft_clean.jsonl"
VAL_PATH = "data/processed/val_sft_clean.jsonl"
OUTPUT_DIR = "outputs/checkpoints/sft_7b_run_01"


def main() -> None:
    """Run SFT training with LoRA on the 7B model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    val_dataset = load_dataset("json", data_files=VAL_PATH, split="train")

    def formatting_func(example: dict) -> str:
        """Format a training example into a single string for SFT."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        bf16=True,
        max_length=1024,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    if train_losses:
        print(f"Final train loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"Final eval loss:  {eval_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
