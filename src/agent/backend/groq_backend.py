from src.agent.backend.base_backend import Backend
from groq import Groq
import os


class GroqBackend(Backend):
    def __init__(self):
        super().__init__("groq")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"

    def generate(self, message):
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

