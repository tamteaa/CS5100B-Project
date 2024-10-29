from envManager import EnvManager
from src.envwrapper.env_names import EnvironmentNames

class Simulator:
    """
    This is the main simulator class. It provides a common interface to manage and run environments.
    """

    def __init__(self):
        """
        Initializes an empty dictionary to keep track of each environment.
        """
        self.environments = {}

    def add_environment(self, env_name, num_of_agents = 1):
        """
        Adds a new environment to the simulator.

        :param env_name         : Environment name. Must be one specified in :class:`EnvironmentNames`.

        :param num_of_agents    : Number of agents per environment.

        :return                 : None
        """
        if env_name not in (env.value for env in EnvironmentNames):
            raise Exception("Environment name must be one of {0}".format(EnvironmentNames))
        self.environments[env_name] = EnvManager(env_name, num_of_agents)
        print(f"{env_name} environment created.")


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


if __name__ == "__main__":
    simulator = Simulator()
    simulator.add_environment(EnvironmentNames.GRID_WORLD.value)