import time
from src.environments.custom_environments.gridworld_environment import GridworldEnvironment
from src.agent.base_agent import Agent
from src.agent.prompts import PromptLoader
from dotenv import load_dotenv
from src.storage.database import DatabaseManager

# Load the GROQ API KEY from a .env file
load_dotenv("../.env")

if __name__ == '__main__':
    # Initialize DatabaseManager
    # If reset_db is False, use the exting db with db_name but the episode id needs to fixed accordingly
    db = DatabaseManager(db_name="agent_data.db", reset_db=True)

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

    # Define the maximum number of steps per episode
    max_steps = 10
    max_episode = 3 

    # Start simulation for a single episode
    print("Starting Gridworld Simulation...\n")

    for episode_id in range(max_episode): 
        # Initialize history for this episode
        episode_history = []
        agent_reached_target = False

        for step in range(max_steps):
            time.sleep(1)
            print("=" * 20 + f" Step {step + 1} of {max_steps} " + "=" * 20 + "\n")

            for agent_id, agent in agents.items():
                observation = env.get_agent_position(agent_id)
                observation_str = f"Your current position is: {observation}"
                print(f"Agent {agent_id} Observation: {observation_str}")

                # Agent makes a decision based on the current observation
                action = agent.step(observation_str)
                action_name = action.get("action_name", "invalid")
                rationale = action.get("rationale", "No rationale provided.")
                action_params = action.get("action_parameters", {})

                # Log this action and observation in the episode history
                episode_history.append({
                    "observation": observation_str,
                    "action_name": action_name,
                    "action_parameters": action_params,
                    "rationale": rationale
                })

                print(f"Agent {agent_id} Action: {action_name}")
                print(f"Rationale: {rationale}\n")

                # Execute the action in the environment
                observation_str = env.step(agent_id, action_name)
                agents_position = env.get_agent_position(agent_id)

                print("Updated Grid State:")
                env.render()
                print(f"Agent {agent_id} Current Position: {agents_position}\n")

                # Check if the agent has reached the target position
                if agents_position == target_position:
                    agent_reached_target = True
                    print(f"Agent {agent_id} has reached the target position {target_position}!")
                    break

            print("\n" + "=" * 50 + "\n")

            if agent_reached_target:
                break

        # Save the episode history in the database as a single row at the end of the episode
        db.insert_episode_history(episode_id, agent_id, episode_history)
        print(f"Episode {episode_id} saved with full action and observation history.")

    # Close the database connection
    db.close()
