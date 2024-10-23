from typing import List


# custom environments
from src.environments.custom_environments.gridworld_environment import GridworldEnvironment


class EnvironmentWrapper:
    def __init__(
            self,
            environments: List
    ):

        self.environment_map = {
            "gridworld": GridworldEnvironment
        }

        self.environments = environments

        assert len(self.environments) == 1



