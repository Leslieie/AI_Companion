"""Starter SFT training script with LoRA.

Usage:
    python src/training/train_sft.py

Requires a prepared training file at data/processed/train_sft.jsonl.
Adjust hyperparameters in configs/training_sft.yaml or directly below.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/processed/train_sft.jsonl"
OUTPUT_DIR = "outputs/checkpoints/sft_run_01"


def main() -> None:
    """Run SFT training with LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

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
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        max_seq_length=1024,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
