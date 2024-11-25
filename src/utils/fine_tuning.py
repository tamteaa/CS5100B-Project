from importlib import import_module
from typing import Dict, Optional, List, Any, Tuple, Union, Literal
from datetime import datetime
import time
import json
from pathlib import Path
import sys
from dataclasses import dataclass
import together
from together.types import FileResponse, FileList, FinetuneResponse, FinetuneLRScheduler, TrainingType, FullTrainingType, LoRATrainingType
from together.types.finetune import FinetuneEvent, FinetuneJobStatus, StrictBool


@dataclass
class FineTuneConfig:
    n_epochs: int = 1
    n_checkpoints: int = 1
    batch_size: Union[int, Literal["max"]] = "max"
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    n_evals: int = 0
    lora: bool = False
    lora_r: Optional[int] = None
    lora_dropout: float = 0
    lora_alpha: Optional[float] = None
    lora_trainable_modules: str = "all-linear"
    suffix: Optional[str] = None
    wandb_api_key: Optional[str] = None
    verbose: bool = False
    train_on_inputs: Union[bool, Literal["auto"]] = "auto"


class TogetherFineTune:
    def __init__(self, api_key: str, log_dir: Optional[str] = None, config: Optional[FineTuneConfig] = None):
        self.tiktoken = self._import_tiktoken()
        self.client = together.Client(api_key=api_key)
        self.together = together.Together(api_key=api_key)
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "finetune_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.tokenizer = self.tiktoken.get_encoding("cl100k_base")
        self.config = config or FineTuneConfig()

    def _import_together(self):
        try:
            return import_module('together')
        except ImportError:
            raise ImportError("Please install the Together.ai SDK: pip install together")

    def _import_tiktoken(self):
        try:
            return import_module('tiktoken')
        except ImportError:
            raise ImportError("Please install tiktoken: pip install tiktoken")

    def check_file(self, file_path: str) -> Dict[str, Any]:
        print(f"\n=== Checking file: {file_path} ===")
        result = together.Files.check(file_path)
        print(f"✓ File check completed")
        return result

    def upload_file(self, file_path: str) -> FileResponse:
        print(f"\n=== Uploading file: {file_path} ===")
        result = self.client.files.upload(file_path)
        print(f"✓ File uploaded: {result.id}")
        return result

    def prepare_files(self, training_file: str, validation_file: Optional[str] = None) -> Dict[str, Any]:
        """Returns file details with IDs and metadata"""
        result = {
            "training": {
                "path": training_file,
                "check": self.check_file(training_file),
                "upload": self.upload_file(training_file)
            }
        }

        if validation_file:
            result["validation"] = {
                "path": validation_file,
                "check": self.check_file(validation_file),
                "upload": self.upload_file(validation_file)
            }

        return result

    def _calculate_tokens(self, file_path: str) -> Tuple[int, float]:
        with open(file_path, 'r') as f:
            text = f.read()
            total_tokens = len(self.tokenizer.encode(text))
        estimated_cost = (total_tokens / 1000) * 0.008
        return total_tokens, estimated_cost

    def create_finetune(
            self,
            training_file: str,
            model: str,
            validation_file: Optional[str] = None,
            force: bool = False,
            **kwargs
    ) -> FinetuneResponse:
        print("\n=== Stage 1: File Analysis ===")
        tokens, cost = self._calculate_tokens(training_file)
        val_tokens, val_cost = (0, 0) if not validation_file else self._calculate_tokens(validation_file)

        print(f"\nCurrent files in Together:")
        files = self.list_files()
        for file in files:
            print(f"ID: {file} ")

        print(f"\nTraining tokens: {tokens:,}")
        if validation_file:
            print(f"Validation tokens: {val_tokens:,}")
        print(f"Estimated cost: ${(cost + val_cost):.2f}")

        if files and not force:
            response = input("\nDelete all existing files? [y/N]: ")
            if response.lower() == 'y':
                for file in files:
                    print("delete file: ", file.id)
                    self.delete_file(file.id)
                print("All files deleted.")

        if not force:
            response = input("\nProceed with file upload? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(0)

        print("\n=== Stage 2: File Processing ===")
        file_details = self.prepare_files(training_file, validation_file)

        training_id = file_details["training"]["upload"].id
        validation_id = file_details.get("validation", {}).get("upload", {}).id

        print(f"\nTraining file ID: {training_id}")
        if validation_id:
            print(f"Validation file ID: {validation_id}")

        print("\n=== Stage 3: Job Creation ===")
        print(f"Model: {model}")
        print(f"Parameters: {self.config}")

        if not force:
            response = input("\nStart fine-tuning job? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(0)

        job = self.together.fine_tuning.create(
            training_file=training_id,
            model=model,
            validation_file=validation_id,
            **vars(self.config)
        )

        self._log_event(job.id, "job_created", {
            **job.model_dump(),
            "training_tokens": tokens,
            "validation_tokens": val_tokens,
            "estimated_cost": cost + val_cost
        })

        print(f"\n✓ Job created: {job.id}")
        return job

    # [Rest of the methods remain the same]
    def monitor_job(
            self,
            job_id: str,
            poll_interval: int = 60,
            callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        last_status = None
        metrics_history = []
        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)
            current_status = status.get("status")
            elapsed_time = time.time() - start_time

            if current_status != last_status:
                print(f"\nStatus changed to: {current_status}")
                print(f"Elapsed time: {elapsed_time:.1f}s")
                last_status = current_status

            if "metrics" in status:
                metrics_history.append(status["metrics"])
                self._log_metrics(job_id, status["metrics"])
                print(f"\rStep: {status['metrics'].get('step', 'N/A')} | "
                      f"Loss: {status['metrics'].get('loss', 'N/A'):.4f}", end="")

            if callback:
                callback(status)

            if current_status in ["succeeded", "failed", "cancelled"]:
                print(f"\nJob completed with status: {current_status}")
                print(f"Total time: {elapsed_time:.1f}s")
                return {
                    "final_status": status,
                    "metrics_history": metrics_history,
                    "total_time": elapsed_time
                }

            time.sleep(poll_interval)

    def _log_event(self, job_id: str, event_type: str, data: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data
        }

        log_file = self.log_dir / f"{job_id}_logs.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _log_metrics(self, job_id: str, metrics: Dict[str, Any]):
        metrics_file = self.log_dir / f"{job_id}_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }) + "\n")

    def get_job_status(self, job_id: str) -> FinetuneResponse:
        return self.client.fine_tuning.retrieve(job_id)

    def list_jobs(self) -> List[FinetuneResponse]:
        jobs = self.client.fine_tuning.list().data
        return [job for job in jobs]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self.client.fine_tuning.cancel(job_id)

    def list_files(self) -> List[FileResponse]:
        result = self.client.files.list()
        return result.data

    def get_file(self, file_id: str) -> FileResponse:
        return self.client.files.retrieve(file_id)

    def get_file_content(self, file_id: str) -> str:
        return self.client.files.retrieve_content(file_id)

    def delete_file(self, file_id: str) -> Dict[str, Any]:
        return self.client.files.delete(file_id)

