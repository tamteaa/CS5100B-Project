"""
This class provides manages an environment.
It defines a specific environment, number of agents in that environment,
and the actions to take.
"""
import time

from dotenv import load_dotenv
from src.agent.base_agent import Agent
from src.agent.prompts import PromptLoader
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld
from src.environments.custom_environments.gridworld_environment import GridworldEnvironment
from src.envwrapper.env_names import EnvironmentNames

# Loading the env file.
load_dotenv()

class EnvManager:

    __env_map = {
        EnvironmentNames.GRID_WORLD.value: GridworldEnvironment,
        EnvironmentNames.COMPLEX_GRID_WORLD.value: ComplexGridworld
    }
    
    __prompt_map = {
        EnvironmentNames.GRID_WORLD.value: PromptLoader().load_prompt("gridworld_system_prompt")
    }

    def __init__(self, env_name, **kwargs):
        """
        The constructor of the EnvManager class.

        :param env_name : The name of the environment.

        :param agents   : Number of agents in the environment.

        :param kwargs   : used to spin up the environment object. If no kwarg is specified, it will use the
                          default settings for environment.
         """
        self.agents = []
        self.target = None
        self.output_instruction_text = None
        env_name = env_name.lower()
        if env_name not in (env.value for env in EnvironmentNames):
            raise ValueError("Invalid environment name")

        self.env_name = env_name
        self.env = self.__env_map[env_name](**kwargs)

    def create_agents(self, agents, unified_goal, prompt, agent_starting_positions):
        """
        Creates the specified agents.

        :param agents: a list containing the agent names.

        :param unified_goal: The unified goal for the agent(s).

        :return: None
        """
        for i in range(len(agents)):
            agent_id = int(i)
            name = agents[i]
            action_space = []
            #system_prompt = self.__prompt_map[self.env_name]

            variables = {
                "name": agents[i],
                "goal": unified_goal,
                "agent_names": agents,
                "n_agents": len(agents)
                #"gridworld_size": grid_size
            }

            #agent = Agent(agent_id, name, action_space, str(system_prompt))

            agent = Agent(
                agent_id=agent_id,
                name=name,
                variables=variables,
                action_space=action_space
            )
            agent.set_system_prompt(prompt)
            self.agents.append(agent)

        for agent, position in zip(self.agents, agent_starting_positions):
            agent.set_start_position(position)

        if self.env_name == EnvironmentNames.COMPLEX_GRID_WORLD.value:
            self.env.set_agents_for_env(self.agents)


    def define_target(self, target):
        """
        Sets the specified target for this environment.

        :param target   :   Target for this environment.

        :return         :   None
        """
        self.target = target

    def is_target_achieved(self, agent):
        """
        Given an agent, this method checks if the target is achieved for that agent.

        :param agent    :   Agent object to be checked.

        :return         :   boolean true of false.
        """
        return self.env.get_agent_position(agent.id) == self.target

    def set_output_instruction_text(self, output_instruction_text):
        """
        Sets the output instruction text for this environment.

        :param output_instruction_text  : Output instruction text for this environment.

        :return                         : None
        """
        self.output_instruction_text = output_instruction_text

    def run(self, num_episodes):
        """
        Runs the specified number of episodes.

        :param num_episodes :   Number of episodes to run.

        :return             :   None
        """
        print("*" * 20 + f" Starting simulation of environment {self.env_name} " + "*" * 20)
        observation = self.env.reset()
        #observation_str = f"Your current position is: {observation[0]}"
        agent_reached_target = False
        for episode in range(num_episodes):
            time.sleep(1)
            print("*" * 20 + f" Episode {episode + 1} of {num_episodes} " + "*" * 20)

            for agent in self.agents:
                observation = self.env.get_agent_position(agent.id)
                agent.observation = f"Your current position is: {observation}"
                print(f"Agent {agent.id}: {agent.name}, Observation {observation}")
                action = agent.step()

                # Extract the action name from the agent's response
                action_name = action.get("action_name", "invalid")
                print(f"Agent {agent.id} Action: {action_name}")
                print(f"Rationale: {action.get('rationale', 'No rationale provided.')}")

                # Execute the action in the environment
                observation_str = self.env.step(agent.id, action_name)

                # Display the updated grid state
                print("Updated Grid State:")
                #self.env.render()

                # Get the agent's current position
                agents_position = self.env.get_agent_position(agent.id)
                print(f"Agent {agent.id} Current Position: {agents_position}")

                # Check if the agent has reached the target position
                if self.is_target_achieved(agent):
                    agent_reached_target = True
                    print(f"Agent {agent.id} has reached the target position {self.target}!")
                    break

            print("*" * 50)

            if agent_reached_target:
                break

        # Final summary
        if agent_reached_target:
            print("Simulation Complete: The agent successfully reached the target position!")
        else:
            print("Simulation Complete: The agent did not reach the target position within the maximum number of "
                  "episodes.")

