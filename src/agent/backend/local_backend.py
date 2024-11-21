from openai import OpenAI
from src.agent.backend.base_backend import Backend
from enum import Enum


class LocalModels(Enum):
    LLAMA_7B = "llama-2-7b-chat"
    MISTRAL_7B = "mistral-7b-instruct"
    NEURAL_7B = "neural-chat-7b"


class LocalBackend(Backend):
    def __init__(
            self,
            model_id: str = LocalModels.MISTRAL_7B.value,
            base_url: str = "http://localhost:1234/v1"
    ):
        super().__init__(
            name="local",
            api_key_prefix="LOCAL_API_KEY",  # Not really needed but kept for consistency
            rate_limit=1000,  # High limit since it's local
            min_delay=0.1
        )
        self.model = model_id
        self.base_url = base_url
        self.temperature = 0.7
        self.client = None

    def generate(self, messages):
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="lm-studio"
            )

            return self.client.chat.completions.create(
                messages=self._truncate_messages(messages),
                model=self.model,
                temperature=self.temperature
            ).choices[0].message.content
        except Exception as e:
            self.logger.error(f"Local inference error: {str(e)}")
            raise