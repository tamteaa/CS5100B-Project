from src.agent.backend.base_backend import Backend


class CohereBackend(Backend):
    def __init__(self, model_id: str = "command"):
        super().__init__(
            name="cohere",
            api_key_prefix="COHERE_API_KEY",
            rate_limit=20,
            min_delay=3,
            history_length=10
        )
        self.model = model_id
        self.temperature = 0.9
        self.client = None

    def generate(self, messages):
        from cohere import Client
        api_key = self._get_next_api_key()
        self._respect_rate_limit(api_key)
        self._update_api_call_stats(api_key)

        try:
            self.client = Client(client_name="CLIENT", api_key=api_key)

            # Convert chat format to Cohere format
            chat_history = []
            for msg in self._truncate_messages(messages)[:-1]:  # Exclude the last message
                role = "CHATBOT" if msg["role"] == "assistant" else "USER"
                chat_history.append({"role": role, "message": msg["content"]})

            # Get the last message
            last_message = messages[-1]["content"]

            return self.client.chat(
                message=last_message,
                chat_history=chat_history,
                model=self.model,
                temperature=self.temperature
            ).text
        except Exception as e:
            if "429" in str(e):
                self.logger.error(f"Rate limit (429) hit for {self.api_key_prefix}")
            raise