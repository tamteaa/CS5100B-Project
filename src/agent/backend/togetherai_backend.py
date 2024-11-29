from src.agent.backend.base_backend import Backend


class TogetherBackend(Backend):
    def __init__(self, model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        super().__init__(
            name="together",
            api_key_prefix="TOGETHER_API_KEY",
            rate_limit=1000,
            min_delay=1,
            history_length=7
        )
        self.model = model_id
        self.temperature = 0.9
        self.client = None

    def generate(self, messages):
        from together import Together
        api_key = self._get_next_api_key()
        self._respect_rate_limit(api_key)
        self._update_api_call_stats(api_key)

        try:
            self.client = Together(api_key=api_key)
            return self.client.chat.completions.create(
                messages=self._truncate_messages(messages),
                model=self.model,
                temperature=self.temperature,
                max_tokens=1024
            ).choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                self.logger.error(f"Rate limit (429) hit for {self.api_key_prefix}")
            raise

