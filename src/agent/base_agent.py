from typing import List, Dict, Any
from src.agent.backend.groq_backend import GroqBackend
import ast
import json
import re


def extract_json_from_string(s: str) -> Dict[str, Any]:
    """
    Extracts the first JSON-like object from the given string.

    Parameters:
        s (str): The string to be parsed.

    Returns:
        Dict[str, Any]: The parsed JSON object or an empty dictionary if not found.
    """
    start_idx = s.find('{')
    end_idx = s.rfind('}')

    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"String does not contain valid brackets: {s}")

    s = s[start_idx:end_idx + 1]
    s = s.strip()

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass

    # Improved regex pattern
    pattern = r"['\"]?(\w+)['\"]?\s*:\s*['\"]?([^,'\"{}]+)['\"]?"
    matches = re.findall(pattern, s)
    result = {match[0]: match[1] for match in matches}

    # If we couldn't extract any key-value pairs, raise ValueError
    if not result:
        raise ValueError(f"Unable to parse string as dictionary: {s}")

    return result


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




