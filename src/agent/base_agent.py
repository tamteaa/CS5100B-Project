from typing import List, Dict, Tuple
from src.agent.backend.groq_backend import GroqBackend
from src.utils.output_parsing import extract_json_from_string
from src.agent.actions import format_actions, Action
from src.agent.backend.cohere_backend import CohereBackend
from src.agent.backend.togetherai_backend import TogetherBackend
from src.agent.backend.openai_backend import OpenAIBackend
from src.agent.backend import Provider
from src.agent.backend.local_backend import LocalBackend
from src.agent.prompts import PromptTemplate

DEFAULT_SYSTEM_PROMPT = """[ Introduction ]
You are an intelligent Agent in a novel simulated gridworld environment. Your goal is to reach a score of 100 by the end of the simulation (you have a limited amount of episodes to complete the objective). 
This is a novel, unknown environment. You must also explore and reason about the environment around you with your team.

[ Communication Guidelines ]
- Communicate clearly and constructively with other agents and the system.
- Share relevant observations or intentions to foster understanding and collaboration.
- Use concise, respectful language when interacting or negotiating with other agents.
- Ask questions when clarification is needed to make informed decisions.

[Agent Information]
**Name**: <<name>>

**Goal** : <<goal>>

[ Environment Information ]
- **General** :
  - Gridworld size: <<grid_size>>
    - The (0, 0) position in the gridworld is at the south-west corner, while the north-east corner is at <<grid_size>>.
  - Total Agents (including you): <<n_agents>>
  - Names of agents in simulation: <<agent_names>>

***IMPORTANT*** : Positions in the gridworld are in the form (x, y). 

[ Action Space ]
- **Actions** :
<<actions>>"""


DEFAULT_SYSTEM_PROMPT_OUTPUT_INSTRUCTIONS = """
You are required to respond in JSON format only.

Your response must include the following keys:
1. **reflection**: Reflect on your current progress towards the goal. Check if the score reflects that the objective is met. If not, analyze why and plan your next steps accordingly.
2. **rationale**: A brief explanation of why this action was chosen, considering the current state and objectives.
3. **action_name**: The name of the action you intend to perform.
4. **action_parameters**: Any specific parameters related to the action, such as step count or target position. If there are no parameters, use an empty dictionary.
5. **message**: An optional but crucial message for communicating with other agents. Use this to propose corners, confirm assignments, or resolve conflicts. Every message will be shared with the entire group.
6. **add_memory**: (Optional) If you choose to use this, include any important information you want to remember for future turns. This memory will be appended to your existing memory and will be accessible in subsequent turns.

Here is an example of the expected format:

{
    "reflection": "",
    "rationale": "",
    "action_name": "",
    "action_parameters": {},
    "message": "",
    "add_memory": ""
}

Remember, you must always output a JSON response following this structure."""

DEFAULT_USER_PROMPT = """
[observation]

Episode <<current_episode>> of <<max_episodes>>

The score is <<score>> / 100

Memory: <<memory>>

Your current goal is: <<goal>>

The observation from your previous action is:
<<observation>>

Your current position:
  x-position: <<x_position>>
  y-position: <<y_position>>

Your current inbox reads:
<<inbox>>

**Note**: Before declaring that you have completed the objective, verify that the score is 100/100. If it's less, reflect on what might be missing and adjust your actions accordingly.
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
            backend_provider: Provider = Provider.GROQ,
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
        
        self.item = None

        valid_backends = [
            Provider.GROQ,
            Provider.TOGETHER,
            Provider.LOCAL,
        ]

        if backend_provider not in valid_backends:
            raise KeyError(f"Must use one of {valid_backends} as backend")
        try:
            self.backend_model = backend_model.value
        except:
            self.backend_model = backend_model

        self.backend_map = {
            "GROQ": GroqBackend,
            "cohere": CohereBackend,
            "TOGETHER": TogetherBackend,
            "openai": OpenAIBackend,
            "LOCAL": LocalBackend
        }

        self.backend = self.backend_map[backend_provider.name](model_id=self.backend_model)

        self.user_prompt = None
        self.output_instructions = None

        self.last_user_message = None
        self.last_assistant_message = None

    def create_system_prompt(self, prompt_name: str):
        raise ValueError("Use a Yaml, this method is outdated")

    def add_inbox_message(self, message: str):
        self.inbox.append(message)

    def add_user_message(self, message: str):
        """
        Adds a user message to the conversation history.
        """
        self.messages.append({"role": "user", "content": message})
        self.last_user_message = message

    def add_agent_message(self, message: str):
        """
        Adds an agent message to the conversation history.
        """
        self.messages.append({"role": "assistant", "content": str(message)})
        self.last_assistant_message = message

    def get_message_history(self) -> List[Dict]:
        """
        Returns the conversation history.
        """
        return self.messages

    def set_system_prompt(self, system_prompt: str):
        """
        Sets a system prompt at the beginning of the conversation.
        """
        if self.output_instructions == "NONE":
            raise ValueError("must set output instructions first, use agent.set_output_instructions(output_instructions: str) or agent.use_default_output_instructions()")

        system_prompt = PromptTemplate(initial_data=system_prompt)
        system_prompt.set_variables(self.variables)
        self.messages.insert(0, {"role": "system", "content": str(system_prompt) + "\n" + self.output_instructions})

    def use_default_system_prompt(self):
        self.set_system_prompt(DEFAULT_SYSTEM_PROMPT)

    def set_output_instructions_prompt(self, output_instructions: str):
        """
        Sets a system prompt at the beginning of the conversation.
        """
        self.output_instructions = output_instructions

    def use_output_instructions_prompt(self):
        self.output_instructions = DEFAULT_SYSTEM_PROMPT_OUTPUT_INSTRUCTIONS

    def set_user_prompt(self, user_prompt):
        """
        Sets the user prompt format.
        """
        self.user_prompt = PromptTemplate(initial_data=user_prompt)

    def use_default_user_prompt(self):
        self.user_prompt = PromptTemplate(initial_data=DEFAULT_USER_PROMPT)

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

        current_score = self.variables.get("score", "NONE")
        if current_score == "NONE":
            self.variables["score"] = 0

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
        self.add_user_message(str(self.user_prompt))

        # Generate response from backend
        response = self.backend.generate(self.messages)

        # Add agent's response to messages
        self.add_agent_message(response)

        action_dict = extract_json_from_string(response)

        # Extract the action name from the agent's response
        action_name = action_dict.get("action_name", "invalid")

        new_memory = action_dict.get("add_memory", None)
        if new_memory:
            self.variables["memory"] += f"\n {new_memory}\n"

        if self.debug:
            print(f"Agent {self.id} Action: {action_name}")
            print(f"Rationale: {action_dict.get('rationale', 'No rationale provided.')}\n")

        return action_dict





