import os

from src.utils.fine_tuning import TogetherFineTune, FineTuneConfig
from dotenv import load_dotenv

# Load the GROQ API KEY from a .env file
load_dotenv(".env")


# 3 million training tokens
# 2 million validation tokens

if __name__ == '__main__':
    key = os.environ["TOGETHER_API_KEY1"]

    config = FineTuneConfig(
        n_epochs=8,  # Increased epochs for convergence
        n_checkpoints=3,  # Retain frequent checkpoints
        batch_size=32,  # Keep larger batch size for stability
        learning_rate=3e-6,  # Further reduce learning rate for finer updates
        min_lr_ratio=0.1,  # Gradual reduction of learning rate remains
        warmup_ratio=0.3,  # Smoother warmup for early training stability
        max_grad_norm=0.3,  # Retain reduced gradient clipping
        weight_decay=0.05,  # Slightly increased regularization
        n_evals=3,  # Retain frequent evaluations
        lora=True,  # Continue using LoRA
        lora_r=16,  # Maintain expressive rank for LoRA
        lora_dropout=0.1,  # Retain dropout for regularization
        lora_alpha=16,  # Keep scaling factor unchanged
        lora_trainable_modules="all-linear",  # Focus on linear layers
        suffix="refined-finetune",  # Updated suffix
        verbose=True,  # Detailed logging
        train_on_inputs="auto",  # Retain auto-detection
        wandb_api_key=os.environ.get("WANDB_API_KEY", None)  # Monitoring
    )

    api = TogetherFineTune(
        key,
        config=config
    )

    job = api.create_finetune(
        "training_data.jsonl",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        "validation_data.jsonl",
        force=False,
    )


