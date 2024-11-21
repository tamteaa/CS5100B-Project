from src.agent.backend.base_backend import Backend


class OpenAIBackend(Backend):
    def __init__(self, model_id: str = "gpt-3.5-turbo"):
        super().__init__(
            name="openai",
            api_key_prefix="OPENAI_API_KEY",
            rate_limit=3500,  # RPM for most models
            min_delay=0.05,   # 50ms between requests
            history_length=10
        )
        self.model = model_id
        self.temperature = 0.9
        self.client = None

    def generate(self, messages):
        from openai import OpenAI
        api_key = self._get_next_api_key()
        self._respect_rate_limit(api_key)
        self._update_api_call_stats(api_key)

        try:
            self.client = OpenAI(api_key=api_key)
            return self.client.chat.completions.create(
                messages=self._truncate_messages(messages),
                model=self.model,
                temperature=self.temperature
            ).choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                self.logger.error(f"Rate limit (429) hit for {self.api_key_prefix}")
            raise