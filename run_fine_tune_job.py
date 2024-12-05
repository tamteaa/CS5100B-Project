import os

from src.utils.fine_tuning import TogetherFineTune, FineTuneConfig
from dotenv import load_dotenv

# Load the GROQ API KEY from a .env file
load_dotenv(".env")

if __name__ == '__main__':
    key = os.environ["TOGETHER_API_KEY1"]

    config = FineTuneConfig(
        n_epochs=1,
        batch_size=8,
        n_evals=1,
        wandb_api_key=os.environ.get("WANDB_API_KEY", None)
    )

    api = TogetherFineTune(
        key,
        config=config
    )

    job = api.create_finetune(
        "training_data.jsonl",
        "mistralai/Mistral-7B-v0.1",
        "validation_data.jsonl",
        force=False,
    )
    result = api.monitor_job(job.id)


