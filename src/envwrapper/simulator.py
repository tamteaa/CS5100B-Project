"""
This is the main simulator class. It provides a common interface to manage and run environments.
"""
from dotenv import load_dotenv

from envManager import EnvManager
from src.envwrapper.env_names import EnvironmentNames

class Simulator:

    def __init__(self):
        self.environments = {}

    def add_environment(self, env_name, num_of_agents = 1):
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name] = EnvManager(env_name, num_of_agents)
        print(f"{env_name} environment created.")


    def remove_environment(self, env_name):
        del self.environments[env_name]

    def run_environment(self, env_name):
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name].run()


    def run_all(self, parallel_run = False):
        pass


if __name__ == "__main__":
    simulator = Simulator()
    simulator.add_environment(EnvironmentNames.GRID_WORLD.value)