from typing import List, Dict, Tuple
from src.agent.backend.groq_backend import GroqBackend
from src.utils.output_parsing import extract_json_from_string
from src.agent.actions import format_actions, Action
from src.agent.prompts import PromptLoader

output_instruction_text = """
Use JSON mode. You are required to respond in JSON format only.

Your response must include the following keys:
1. **action_name**: The name of the action you intend to perform.
2. **action_parameters**: Any specific parameters related to the action, such as step count or target position. If there are no parameters, use an empty dictionary.
3. **rationale**: A brief explanation of why this action was chosen, considering the current state and objectives.
4. **message**: An optional message to send to other agents. If there is no message to send, use an empty string.

Here is an example of the expected format:

{
  "action_name": "",
  "action_parameters": {},
  "rationale": "",
  "message": ""
}

Remember, you must always output a JSON response following this structure.
"""


class Agent:
    def __init__(
            self,
            agent_id: int,
            name: str,
            action_space: List[Action],
            variables: Dict,
            start_position: Tuple = None,
            color: Tuple = None,
            enforce_json_output: bool = False,
            backend: str = "groq",
            backend_model: str = "llama3-70b-8192",
            debug: bool = False,
    ):
        self.id = agent_id
        self.name = name
        self.action_space = action_space
        self.variables = variables
        self.enforce_json_output = enforce_json_output
        self.position = start_position
        self.color = color
        self.debug = debug

        self.messages = []

        self.inbox = []

        self.observation = ""
        self.backend_model = backend_model
        self.backend_map = {"groq": GroqBackend}
        if backend != "groq":
            raise KeyError("Must use groq as backend")

        self.backend = self.backend_map[backend](model_id=self.backend_model)
        self.user_prompt = None

    def create_system_prompt(self, prompt_name: str):
        prompts = PromptLoader()

        system_prompt = prompts.load_prompt(prompt_name)

        # Printing the formatted actions using the new function
        actions_description = format_actions(self.action_space)

        self.variables["actions"] = actions_description

        system_prompt.set_variables(self.variables)

        system_prompt_str = str(system_prompt)
        system_prompt_str += output_instruction_text

        self.set_system_prompt(system_prompt_str)

    def add_inbox_message(self, name_from: str, msg: str):
        message = f"From: {name_from}\nMessage: {msg}\n"

        self.inbox.append(message)

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

    def set_user_prompt(self, prompt):
        """
        Sets a system prompt at the beginning of the conversation.
        """
        self.user_prompt = prompt

    def set_action_space(self, action_space: [Action]):
        self.action_space = action_space
        actions_description = format_actions(self.action_space)
        self.variables["actions"] = actions_description


    def set_start_position(self, position: Tuple):
        self.position = position

    def set_agent_color(self, color: Tuple):
        self.color = color

    def step(self):
        """
        Takes an observation, generates a response using the backend,
        and adds the response to the conversation history.
        """
        self.variables["observation"] = self.observation
        self.variables["x_position"] = self.position[1]
        self.variables["y_position"] = self.position[0]

        if self.inbox:  # Only format if there are messages
            formatted_messages = "============\nInbox:\n\n"
            for i, message in enumerate(self.inbox, 1):
                formatted_messages += f"{i}. {message}\n\n"
            formatted_messages += "============"
            self.variables["inbox"] = formatted_messages
        else:
            self.variables["inbox"] = "============\nInbox: Empty\n============"
        self.inbox.clear()

        self.user_prompt.set_variables(self.variables)

        # Add user observation to messages
        self.add_user_message(str(self.user_prompt) + " This is the list of valid actions: " + self.variables["actions"] )

        # Generate response from backend
        response = self.backend.generate(self.messages)

        # Add agent's response to messages
        self.add_agent_message(response)

        action_dict = extract_json_from_string(response)

        # Extract the action name from the agent's response
        action_name = action_dict.get("action_name", "invalid")

        if self.debug:
            print(f"Agent {self.id} Action: {action_name}")
            print(f"Rationale: {action_dict.get('rationale', 'No rationale provided.')}\n")

        return action_dict





