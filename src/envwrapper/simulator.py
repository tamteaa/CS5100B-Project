import yaml

from envManager import EnvManager
from src.agent.prompts import PromptLoader
from src.envwrapper.env_names import EnvironmentNames
from src.agent.actions import Action
import re


class Simulator:
    """
    This is the main simulator class. It provides a common interface to manage and run environments.
    """

    __prompt_map = {
        EnvironmentNames.GRID_WORLD.value: PromptLoader().load_prompt("gridworld_system_prompt")
    }

    def __init__(self):
        """
        Initializes an empty dictionary to keep track of each environment.
        """
        self.environments = {}

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


    def create_agents_in_env(self, env_name, agents, unified_goal, prompt):
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name].create_agents(agents, unified_goal, prompt)


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

        :param env_name                 :

        :param output_instruction_text  :

        :return                         :
        """
        self.environments[env_name].set_output_instruction_text(output_instruction_text)

    def set_action_description_for_agent(self, env_name, agent_id, action_description):
        """
        Sets the action description for an agent in the environment.

        :param env_name             : Name of the environment.

        :param agent_id             : ID of the agent.

        :param action_description   : Action description for the agent.

        :return                     : None
        """
        env = self.environments[env_name]

        for agent in env.agents:
            if agent.id == agent_id:
                agent.set_action_description(action_description)
                break


    def remove_environment(self, env_name):
        """
        Removes a specified environment from the simulator.

        :param env_name: Environment name to remove. Must be one specified in :class:`EnvironmentNames`.

        :return        : None
        """
        del self.environments[env_name]

    def run_environment(self, env_name, max_episodes = 20):
        """
        Runs a specified environment.

        :param env_name     : Runs the specified environment for the number of episodes specified.

        :param max_episodes : Maximum number of episodes to run. By default, it is 20.

        :return             : None
        """
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name].run(max_episodes)


    def run_all(self, max_episodes=20, parallel_run = False):
        """
        Runs all environments in the simulator.

        :param max_episodes : Maximum number of episodes for each environment to run. By default, it is 20.

        :param parallel_run : Flag to enable parallel run of the environments.

        :return             : None
        """
        pass

    def __parse_action_description(self, action_description_string):
        pattern = r"- \*\*(\w+)\*\*: (.+)"

        # Parse the actions
        actions = [
            Action(name=match.group(1), description=match.group(2))
            for match in re.finditer(pattern, action_description_string)
        ]
        return actions

    def load_environment_config(self, config_file):
        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)

        environments = config["environments"]
        for env_name in environments:
            print(env_name)
            grid_size = environments[env_name]["grid_size"]
            output_instruction_text = environments[env_name]["output_instruction_text"]
            actions = self.__parse_action_description(environments[env_name]["actions_description"])
            unified_goal = environments[env_name]["unified_goal"]
            agent_names = environments[env_name]["agent_names"]
            prompt = environments[env_name]["prompt"]

            self.add_environment(env_name)
            self.create_agents_in_env(env_name, agent_names, unified_goal, prompt)
            self.set_output_instruction_text_for_env(env_name, output_instruction_text)
            self.set_action_description_for_agent(env_name, agent_names, actions)







if __name__ == "__main__":
    simulator = Simulator()
    simulator.load_environment_config("env_config.yaml")
    simulator.define_target_for_environment(EnvironmentNames.GRID_WORLD.value, (4, 4))
    simulator.run_environment(EnvironmentNames.GRID_WORLD.value)



    """
    Write a config file for env. Put all things needed by env in the file. Have simulator load it and initialize env.
    Connect simulator with DB. (Priority)
    Logging.
    
    Complex grid world.
    Debug multi-env.
    Messaging layer.
    """