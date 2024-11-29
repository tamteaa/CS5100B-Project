import os

from src.utils.fine_tuning import TogetherFineTune, FineTuneConfig
from dotenv import load_dotenv

# Load the GROQ API KEY from a .env file
load_dotenv(".env")


# Training tokens: 5,344,937
# Validation tokens: 1,625,921

if __name__ == '__main__':
    key = os.environ["TOGETHER_API_KEY1"]

    config = FineTuneConfig(
        n_epochs=6,  # Reduced from 8 since we see overfitting
        n_checkpoints=6,
        batch_size=32,  # Increased from 16 to reduce gradient variance
        learning_rate=5e-5,  # Slightly increased from 1e-4 for better convergence
        min_lr_ratio=0.05,  # Increased to prevent learning rate from going too low
        warmup_ratio=0.15,  # Reduced from 0.3 as current warmup seems too long
        max_grad_norm=0.3,  # Reduced from 0.4 to address gradient spikes
        weight_decay=0.03,  # Increased from 0.01 to combat overfitting
        n_evals=8,
        lora=True,
        lora_r=8,  # Increased from 4 to allow more expressivity
        lora_dropout=0.2,  # Reduced from 0.3 as we're adding other regularization
        lora_alpha=16,  # Reduced from 32 to prevent too aggressive updates
        lora_trainable_modules="all-linear",
        suffix="stabilized-finetune-v2",
        verbose=True,
        train_on_inputs="auto",
        wandb_api_key=os.environ.get("WANDB_API_KEY", None)
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


