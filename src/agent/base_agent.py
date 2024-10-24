from typing import List, Dict
from src.agent.backend.groq_backend import GroqBackend
from src.utils.output_parsing import extract_json_from_string


class Agent:
    def __init__(
            self,
            agent_id: int,
            name: str,
            action_space: List[Dict],
            system_prompt: str = "You are a helpful assistant",
            enforce_json_output: bool = False,
            backend: str = "groq",
    ):
        self.id = agent_id
        self.name = name
        self.action_space = action_space
        self.enforce_json_output = enforce_json_output

        self.messages = []

        self.backend_map = {"groq": GroqBackend}
        if backend != "groq":
            raise KeyError("Must use groq as backend")

        self.backend = self.backend_map[backend]()

        # setting system prompt here for now
        self.set_system_prompt(system_prompt)

    def add_user_message(self, message: str):
        """
        Adds a user message to the conversation history.
        """
        self.messages.append({"role": "user", "content": message})

    def add_agent_message(self, message: str):
        """
        Adds an agent message to the conversation history.
        """
        self.messages.append({"role": "assistant", "content": str(message)})

    def get_message_history(self) -> List[Dict]:
        """
        Returns the conversation history.
        """
        return self.messages

    def set_system_prompt(self, prompt: str):
        """
        Sets a system prompt at the beginning of the conversation.
        """
        self.messages.insert(0, {"role": "system", "content": prompt})

    def step(self, observation: str):
        """
        Takes an observation, generates a response using the backend,
        and adds the response to the conversation history.
        """
        # Add user observation to messages
        self.add_user_message(observation)

        # Generate response from backend
        response = self.backend.generate(self.messages)

        # Add agent's response to messages
        self.add_agent_message(response)

        return extract_json_from_string(response)




