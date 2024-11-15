from src.agent.backend.base_backend import Backend
from groq import Groq
import os


class GroqBackend(Backend):
    def __init__(
            self,
            model_id: str = "llama-3.1-8b-instant"
    ):
        super().__init__("groq")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_id
        self.temperature = 0.7

    def generate(self, messages):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature
        )
        return chat_completion.choices[0].message.content

