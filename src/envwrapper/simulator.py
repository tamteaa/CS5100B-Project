import copy

import yaml
import re
import time
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld
from src.gui.gui import GUI
from src.agent.actions import Action, format_actions
from src.storage.database import DatabaseManager
from src.agent.base_agent import Agent
import random
from src.agent.prompts import PromptTemplate
from typing import List, Dict


# Define the simulation logic in a function
def run_simulation(env: ComplexGridworld):
    # Initial observation of the agent's position
    for agent_id, agent in env.agents.items():
        observation = env.get_agent_position(agent_id)
        agent.observation = f"Your current position is: {observation}"

    env.variables["group_messages"] = []
    env.score = 0

    for episode in range(env.max_episodes):
        print(episode)
        for agent_id, agent in env.agents.items():
            time.sleep(1.5)
            # Agent makes a decision based on the current observation
            action_dict = agent.step()

            # Check if there is a message to send and distribute it to other agents
            message = action_dict.get("message", "")
            if message:
                message = f"From: {agent.name}\nMessage: {message}\n"
                for other_agent_id, other_agent in env.agents.items():
                    if other_agent_id != agent_id:  # Only send to other agents
                        other_agent.add_inbox_message(message)
                env.variables["group_messages"].append(
                    {
                        "from": agent.name,
                        "message": message
                    }
                )

            if action_dict.get("action_name", None) == None:
                agent.observation = "your action was invalid"
            else:
                # Execute the action in the environment
                agent.observation = env.step(agent.id, action_dict["action_name"])

            if env.terminated:
                break

        if env.terminated:
            break

    # Final summary
    if env.terminated:
        print("Simulation Complete: The agent successfully reached the target position!")
    else:
        print("Simulation Complete: The agent did not reach the target position within the maximum number of episodes.")


class Simulator:
    """
    This is the main simulator class. It provides a common interface to manage and run environments.
    """

    def __init__(
            self,
            use_db: bool,
            use_gui: bool,
            configs: Dict[str, Dict]
    ):
        """
        Initializes an empty dictionary to keep track of each environment.

        :param use_db: Boolean to use database
        :param use_gui: Boolean to use GUI
        """
        self.use_db = use_db
        self.use_gui = use_gui
        self.configs = configs

        self.environments: dict[str, ComplexGridworld] = {}
        self.db_manager = DatabaseManager() if use_db else None

        self.name_bank = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
            "Grace", "Hank", "Ivy", "Jack", "Kathy", "Leo",
            "Mona", "Nina", "Oscar", "Paul", "Quincy", "Rachel",
            "Steve", "Tina", "Uma", "Vince", "Wendy", "Xander",
            "Yara", "Zane"
        ]

        self.action_bank = {
            "north": Action(name="north", description="Move one step upward on the grid."),
            "south": Action(name="south", description="Move one step downward on the grid."),
            "west": Action(name="west", description="Move one step to the left on the grid."),
            "east": Action(name="east", description="Move one step to the right on the grid."),
            "skip": Action(name="skip", description="Do nothing and skip this step."),
            "pick":  Action(name="pick", description="Pick up the item at current position"),
            "drop": Action(name="drop", description="Drop off the item at current position"),
        }

        self.random_variables = {}

    def run(self, config_key: str, num_simulations: int = 1):
        """
        Runs all environments in the simulator.

        :param num_simulations: Number of simulations for each environment.
        :param max_episodes: Maximum number of episodes for each environment.
        """
        scores = []  # List to store scores from each simulation

        for sim in range(num_simulations):
            print(f"Running simulation {sim + 1}/{num_simulations}...")

            # Load the environment configuration for each simulation
            env = self.load_environment_config(config_key)

            if self.db_manager:
                env.db_manager = self.db_manager

            if self.use_gui:
                gui = GUI(env=env)
                gui.run(run_simulation)
                gui.close()
            else:
                run_simulation(env)

            # Collect the score after each run
            scores.append(env.score)
            print("scores so far are", scores)

        return scores

    def generate_random_variables(self, random_definitions):
        random_values = {}
        for var, expression in random_definitions.items():
            # Use eval to evaluate the random generation expression in the context of the current random_values
            random_values[var] = eval(expression, {"random": random, **random_values})
        return random_values

    def __parse_action_description(self, action_description_string):
        """
        Parses an action description string into a list of Action objects.

        :param action_description_string: The action description string.
        :return: A list of Action objects.
        """
        pattern = r"- \*\*(\w+)\*\*: (.+)"
        actions = [
            Action(name=match.group(1), description=match.group(2))
            for match in re.finditer(pattern, action_description_string)
        ]
        return actions

    def setup_agents(
            self,
            num_agents: int,
            action_space: List[Action],
            grid_size: tuple[int, int],
            shared_agent_variables: dict,
            system_prompt: PromptTemplate,
            user_prompt: PromptTemplate,
            backend_model: tuple[str, str]
    ):
        agents = {}
        positions = set()

        # Generate random, unique starting positions within the grid size
        while len(positions) < num_agents:
            position = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
            positions.add(position)

        starting_positions = list(positions)

        # Shuffle the names and select as many as needed
        random.shuffle(self.name_bank)
        selected_names = self.name_bank[:num_agents]

        for i in range(num_agents):
            agent_id = i
            name = selected_names[i]

            variables = shared_agent_variables.copy()
            variables.update({
                "name": name,
                "agent_names": selected_names,
                "agent_id": agent_id,
                "memory": ""
            })

            agent_system_prompt = copy.deepcopy(system_prompt)
            agent_system_prompt.set_variables(variables)

            agent = Agent(
                agent_id=agent_id,
                name=name,
                variables=variables,
                action_space=action_space,
                backend_provider=backend_model[0],
                backend_model=backend_model[1]
            )
            agent.set_system_prompt(str(agent_system_prompt))
            agent.set_user_prompt(copy.deepcopy(user_prompt))
            agent.set_start_position(starting_positions[i])
            agents[i] = agent

        return agents

    def get_actions_from_yaml(self, actions_yaml: str):
        """
        Parses the action names from the YAML string and returns a list of corresponding Action objects.

        :param actions_yaml: A YAML string listing action names.
        :return: A list of Action objects.
        """
        # Extract action names using regex
        pattern = r"\*\*(\w+)\*\*"
        action_names = re.findall(pattern, actions_yaml)

        # Retrieve corresponding Action objects from the action bank
        actions = [self.action_bank[name.lower()] for name in action_names if name.lower() in self.action_bank]
        return actions

    def apply_random_values(self, config: dict, random_values: dict):
        for key, value in config.items():
            if isinstance(value, str):
                # Replace placeholders in string values
                for placeholder, random_value in random_values.items():
                    value = value.replace(placeholder, str(random_value))
                config[key] = value
            elif isinstance(value, list):
                # Replace placeholders in lists
                config[key] = [
                    random_values.get(item, item) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                self.apply_random_values(value, random_values)

    def load_environment_config(self, config_key: str) -> ComplexGridworld:
        """
        Loads environment configurations from a list of dictionaries and sets up everything.

        :param configs: A list of dictionaries containing environment configurations.
        """
        config = self.configs[config_key]
        yaml_file = config["yaml_file"]
        with open(yaml_file, "r") as file:
            environment_config = yaml.safe_load(file)

        # Generate random values based on the definitions
        random_definitions = environment_config.pop("random_variables")
        if random_definitions:
            random_values = self.generate_random_variables(random_definitions)
            self.apply_random_values(environment_config, random_values)  # Apply the random values

        # Extract environment properties from the YAML configuration
        num_agents = environment_config["num_agents"]
        grid_size = tuple(environment_config.get("grid_size", [5, 5]))
        output_instruction_text = environment_config["output_instruction_text"]
        max_episodes = environment_config["max_episodes"]
        env_variables = environment_config["env_variables"]
        env_variables["score"] = 0
        actions = self.get_actions_from_yaml(environment_config["actions"])

        unified_goal = environment_config["unified_goal"]
        prompt = environment_config["prompt"]

        user_prompt = environment_config["user_prompt"]

        system_prompt = PromptTemplate(initial_data=prompt + output_instruction_text)

        user_prompt_template = PromptTemplate(initial_data=user_prompt)

        shared_agent_variables = {
            "grid_size": grid_size,
            "goal": unified_goal,
            "n_agents": num_agents,
            "actions": format_actions(actions)
        }

        backend_model: str = config.get("backend_model", "")
        if backend_model == "":
            raise ValueError("Must have a backend model")

        backend_provider = config.get("backend_provider", "")
        if backend_provider == "":
            raise ValueError("Must have a backend provider")

        # Set up agents using the setup_agents method
        agents = self.setup_agents(
            num_agents=num_agents,
            action_space=actions,
            shared_agent_variables=shared_agent_variables,
            grid_size=grid_size,
            system_prompt=system_prompt,
            user_prompt=user_prompt_template,
            backend_model=(backend_provider, backend_model)
        )

        env = ComplexGridworld(agents=agents, grid_size=grid_size)
        env.max_episodes = max_episodes
        env.variables = env_variables
        env.score = 0
        # Register the environment and termination condition
        termination_condition = config["termination_condition"]
        env.register_termination_callback(termination_condition)
        return env

