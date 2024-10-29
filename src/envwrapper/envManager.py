"""
This class provides manages an environment.
It defines a specific environment, number of agents in that environment,
and the actions to take.
"""
from dotenv import load_dotenv
from src.agent.base_agent import Agent
from src.agent.prompts import PromptLoader
from src.environments.custom_environments.gridworld_environment import GridworldEnvironment
from src.envwrapper.env_names import EnvironmentNames

# Loading the env file.
load_dotenv()

class EnvManager:

    __env_map = {
        EnvironmentNames.GRID_WORLD.value: GridworldEnvironment
    }
    
    __prompt_map = {
        EnvironmentNames.GRID_WORLD.value: PromptLoader().load_prompt("gridworld_system_prompt")
    }

    def __init__(self, env_name, agents, **kwargs):
        """
        The constructor of the EnvManager class.

        :param env_name : The name of the environment.

        :param agents   : Number of agents in the environment.

        :param kwargs   : used to spin up the environment object. If no kwarg is specified, it will use the
                          default settings for environment.
         """
        self.agents = []
        env_name = env_name.lower()
        if env_name not in (env.value for env in EnvironmentNames):
            raise ValueError("Invalid environment name")

        self.env = self.__env_map[env_name](**kwargs)
        for i in range(agents):
            agent_id = int(i)
            name = f"Agent_{i}"
            action_space = []
            system_prompt = self.__prompt_map[env_name]
        
            agent = Agent(agent_id, name, action_space, str(system_prompt))
            self.agents.append(agent)

    def run(self, num_episodes):
        pass
