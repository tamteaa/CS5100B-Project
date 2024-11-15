import yaml

from envManager import EnvManager
from src.agent.prompts import PromptLoader
from src.envwrapper.env_names import EnvironmentNames
from src.agent.actions import Action
import re

from src.storage.database import DatabaseManager


class Simulator:
    """
    This is the main simulator class. It provides a common interface to manage and run environments.
    """

    __prompt_map = {
        EnvironmentNames.GRID_WORLD.value: PromptLoader().load_prompt("gridworld_system_prompt")
    }

    def __init__(self, use_db, use_gui):
        """
        Initializes an empty dictionary to keep track of each environment.

        :param use_db: Boolean to use database

        :param use_gui: Boolean to use GUI
        """
        self.environments = {}
        self.use_db = use_db
        self.use_gui = use_gui
        self.db_manager = DatabaseManager() if use_db else None

    def add_environment(self, env_name):
        """
        Adds a new environment to the simulator.

        :param env_name         : Environment name. Must be one specified in :class:`EnvironmentNames`.

        :return                 : None
        """
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name] = EnvManager(env_name)
        print(f"{env_name} environment created.")


    def create_agents_in_env(self, env_name, agents, unified_goal, prompt, agent_starting_positions):
        """
        Spawns the agents in the specified environment.

        :param env_name: Name of the environment.

        :param agents: a list of agent names.

        :param unified_goal: The unified goal for the agents.

        :param prompt: A system prompt for the agents.

        :param agent_starting_positions: a list of tuples each containing the starting positions of the agents. Since we work with grid-world,
                                         we use (x, y) co-ordinates.

        :return: None
        """
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name].create_agents(agents, unified_goal, prompt, agent_starting_positions)


    def define_target_for_environment(self, env_name, target):
        """
        Defines a target for an environment. The agent(s) work to achieve this target. This will vary from environment
        to environment.

        :param env_name : name of the environment. Must be one specified in :class:`EnvironmentNames`.

        :param target   : Target for the environment. Agents work to get this target.

        :return         : None
        """
        self.environments[env_name].define_target(target)


    def get_agents_for_environment(self, env_name):
        """
        Returns the agents created for an environment.

        :param env_name : Name of the environment.

        :return         : A List of dicts containing agent id and agent name. { {id: 0, name: Agent_0} }.
        """
        agents = []
        for agent in self.environments[env_name].agents:
            agent_info = {"id": agent.id, "name": agent.name}
            agents.append(agent_info)
        return agents

    def set_output_instruction_text_for_env(self, env_name, output_instruction_text):
        """
        Sets the output instruction text for an environment.

        :param env_name                 : Name of the environment.

        :param output_instruction_text  : Output instruction text for the environment.

        :return                         : None
        """
        self.environments[env_name].set_output_instruction_text(output_instruction_text)

    def set_action_description_for_agent(self, env_name, agent_name, action_description):
        """
        Sets the action description for an agent in the environment.

        :param env_name             : Name of the environment.

        :param agent_name            : name of the agent.

        :param action_description   : Action description for the agent.

        :return                     : None
        """
        env = self.environments[env_name]

        for agent in env.agents:
            if agent.name in agent_name:
                agent.set_action_space(action_description)


    def remove_environment(self, env_name):
        """
        Removes a specified environment from the simulator.

        :param env_name: Environment name to remove. Must be one specified in :class:`EnvironmentNames`.

        :return        : None
        """
        del self.environments[env_name]

    def run_environment(self, env_name, num_simulations=1, max_episodes = 20):
        """
        Runs a specified environment.

        :param env_name     : Runs the specified environment for the number of episodes specified.

        :param num_simulations : Number of simulations to run for the specified environment.

        :param max_episodes : Maximum number of episodes to run. By default, it is 20.

        :return             : None
        """
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        for sim in range(num_simulations):
            self.environments[env_name].run(max_episodes)


    def run_all(self, num_simulations=1, max_episodes=20, parallel_run = False):
        """
        Runs all environments in the simulator.

        :param max_episodes : Maximum number of episodes for each environment to run. By default, it is 20.

        :param num_simulations : Number of simulations to run for each environment.

        :param parallel_run : Flag to enable parallel run of the environments.

        :return             : None
        """
        for env in self.environments.values():
            for sim in range(num_simulations):
                env.run(max_episodes, self.db_manager)


    def __parse_action_description(self, action_description_string):
        """
        This function takes an action description string and converts them into a list of Action objects.

        :param action_description_string: The action description string.

        :return: a list of Action objects.
        """
        pattern = r"- \*\*(\w+)\*\*: (.+)"

        # Parse the actions
        actions = [
            Action(name=match.group(1), description=match.group(2))
            for match in re.finditer(pattern, action_description_string)
        ]
        return actions

    def load_environment_configs(self, config_files):
        """
        Given a list of config files, loads them into the simulator.

        :param config_files: A list of config files.

        :return: None
        """
        for config_file in config_files:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)

            environments = config["environments"]
            for env_name in environments:
                print(env_name)
                grid_size = environments[env_name]["grid_size"]
                output_instruction_text = environments[env_name]["output_instruction_text"]
                actions = self.__parse_action_description(environments[env_name]["actions_description"])
                unified_goal = environments[env_name]["unified_goal"]
                agent_names = environments[env_name]["agent_names"]
                agent_starting_positions = [tuple(pos) for pos in environments[env_name]['agent_starting_positions']]
                prompt = environments[env_name]["prompt"]

                self.add_environment(env_name)
                self.create_agents_in_env(env_name, agent_names, unified_goal, prompt+output_instruction_text, agent_starting_positions)
                self.set_output_instruction_text_for_env(env_name, output_instruction_text)
                self.set_action_description_for_agent(env_name, agent_names, actions)


if __name__ == "__main__":
    simulator = Simulator(use_db=True, use_gui=False)
    simulator.load_environment_configs([
        "single_agent.yaml",
        "multi_agent.yaml"
    ])
    simulator.define_target_for_environment(EnvironmentNames.GRID_WORLD.value, (4, 4))
    #simulator.run_environment(EnvironmentNames.COMPLEX_GRID_WORLD.value)
    simulator.run_all(num_simulations=3)
