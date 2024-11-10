from envManager import EnvManager
from src.agent.prompts import PromptLoader
from src.envwrapper.env_names import EnvironmentNames

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
                variables = {
                    "name": agent.name,
                    "goal": f"Reach the target position at {env.target}.",
                    "actions": action_description,
                }
                system_prompt = self.__prompt_map[env_name]
                system_prompt.set_variables(variables)
                system_prompt_str = str(system_prompt)
                system_prompt_str += env.output_instruction_text
                agent.set_system_prompt(system_prompt_str)
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


if __name__ == "__main__":
    simulator = Simulator()
    simulator.add_environment(EnvironmentNames.GRID_WORLD.value)
    simulator.define_target_for_environment(EnvironmentNames.GRID_WORLD.value, (4, 4))

    output_instruction_text = """
        You are required to respond in JSON format only.

        Your response must include the following keys:
        1. **action_name**: The name of the action you intend to perform.
        2. **action_parameters**: Any specific parameters related to the action, such as step count or target position. If there are no parameters, use an empty dictionary.
        3. **rationale**: A brief explanation of why this action was chosen, considering the current state and objectives.

        Here is an example of the expected format:

        {
          "action_name": "up",
          "action_parameters": {"steps": 1},
          "rationale": "Moving up to get closer to the target position."
        }

        Remember, you must always output a JSON response following this structure.
        """
    simulator.set_output_instruction_text_for_env(EnvironmentNames.GRID_WORLD.value, output_instruction_text)

    actions_description = """
       - **north**: Move one step upward on the grid.
       - **south**: Move one step downward on the grid.
       - **west**: Move one step to the left on the grid.
       - **east**: Move one step to the right on the grid.
       """
    simulator.set_action_description_for_agent(EnvironmentNames.GRID_WORLD.value, 0, actions_description )

    simulator.run_environment(EnvironmentNames.GRID_WORLD.value)



    """
    Write a config file for env. Put all things needed by env in the file. Have simulator load it and initialize env.
    Connect simulator with DB. (Priority)
    Logging.
    
    Complex grid world.
    Debug multi-env.
    Messaging layer.
    """