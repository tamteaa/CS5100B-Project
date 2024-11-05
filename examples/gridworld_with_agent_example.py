import time

from src.environments.custom_environments.gridworld_environment import GridworldEnvironment
from src.agent.base_agent import Agent
from src.agent.prompts import PromptLoader
from dotenv import load_dotenv

# load the GROQ API KEY from a .env file
load_dotenv("../.env")

if __name__ == '__main__':
    # Define start positions for multiple agents
    start_positions = {0: (0, 0)}

    prompts = PromptLoader()

    system_prompt = prompts.load_prompt("gridworld_system_prompt")

    agent_name = "Llama-agent-v1"

    actions_description = """
    - **north**: Move one step upward on the grid.
    - **south**: Move one step downward on the grid.
    - **west**: Move one step to the left on the grid.
    - **east**: Move one step to the right on the grid.
    """

    output_instruction_text = """
    You are required to respond in JSON format only.

    Your response must include the following keys:
    1. **action_name**: The name of the action you intend to perform.
    2. **action_parameters**: Any specific parameters related to the action, such as step count or target position. If there are no parameters, use an empty dictionary.
    3. **rationale**: A brief explanation of why this action was chosen, considering the current state and objectives.

    Here is an example of the expected format:

    {
      "action_name": "north",
      "action_parameters": {"steps": 1},
      "rationale": "Moving up to get closer to the target position."
    }

    Remember, you must always output a JSON response following this structure.
    """

    target_position = (4, 4)

    variables = {
        "name": agent_name,
        "goal": f"Reach the target position at {target_position}.",
        "actions": actions_description,
    }

    system_prompt.set_variables(variables)

    system_prompt_str = str(system_prompt)
    system_prompt_str += output_instruction_text

    print(system_prompt_str)

    agents = {
        0: Agent(
            agent_id=0,
            name=agent_name,
            action_space=[],
            system_prompt=system_prompt_str
        )
    }

    # Initialize the gridworld environment
    env = GridworldEnvironment(grid_size=(5, 5), start_positions=start_positions)
    print("Initial Grid State:")
    env.render()
    print("\n" + "=" * 50 + "\n")

    # Define the maximum number of episodes (steps)
    max_episodes = 10

    # Initial observation of the agent's position
    observation = env.get_agent_position(0)
    observation_str = f"Your current position is: {observation}"

    print("Starting Gridworld Simulation...\n")
    agent_reached_target = False

    for episode in range(max_episodes):
        time.sleep(1)
        print("=" * 20 + f" Episode {episode + 1} of {max_episodes} " + "=" * 20 + "\n")

        for agent_id, agent in agents.items():
            print(f"Agent {agent_id} Observation: {observation_str}")

            # Agent makes a decision based on the current observation
            action = agent.step(observation_str)

            # Extract the action name from the agent's response
            action_name = action.get("action_name", "invalid")
            print(f"Agent {agent_id} Action: {action_name}")
            print(f"Rationale: {action.get('rationale', 'No rationale provided.')}\n")

            # Execute the action in the environment
            observation_str = env.step(agent_id, action_name)

            # Display the updated grid state
            print("Updated Grid State:")
            env.render()

            # Get the agent's current position
            agents_position = env.get_agent_position(agent_id)
            print(f"Agent {agent_id} Current Position: {agents_position}\n")

            # Check if the agent has reached the target position
            if agents_position == target_position:
                agent_reached_target = True
                print(f"Agent {agent_id} has reached the target position {target_position}!")
                break

        print("\n" + "=" * 50 + "\n")

        if agent_reached_target:
            break

    # Final summary
    if agent_reached_target:
        print("Simulation Complete: The agent successfully reached the target position!")
    else:
        print("Simulation Complete: The agent did not reach the target position within the maximum number of episodes.")