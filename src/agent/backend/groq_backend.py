from groq import Groq
from src.agent.backend.base_backend import Backend


class GroqBackend(Backend):
    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        super().__init__(
            name="groq",
            api_key_prefix="GROQ_API_KEY",
            rate_limit=15,
            min_delay=9,
            history_length=8
        )
        self.model = model_id
        self.temperature = 0.9
        self.client = None

    def generate(self, messages):
        while True:
            api_key = self._get_next_api_key()
            self._respect_rate_limit(api_key)
            self._update_api_call_stats(api_key)

            try:
                self.client = Groq(api_key=api_key)
                return self.client.chat.completions.create(
                    messages=self._truncate_messages(messages),
                    model=self.model,
                    temperature=self.temperature
                ).choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    self.logger.error(f"Rate limit (429) hit for key {api_key}")
                    self.handle_rate_limit_error(api_key, error_msg)
                    continue  # Try again with a different key
                if "400" in error_msg:
                    self.logger.error(f"(400) hit for key {api_key}")
                raise  # Re-raise non-rate-limit errors

