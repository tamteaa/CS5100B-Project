import time

from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld, Square, Item
from src.agent.base_agent import Agent
from dotenv import load_dotenv
from src.gui.gui import GUI
from src.agent.actions import Action
import random

# Load the GROQ API KEY from a .env file
load_dotenv("../.env")


# Define the simulation logic in a function
def run_simulation(env: ComplexGridworld):

    # Initial observation of the agent's position
    print("Starting Gridworld Simulation...\n")

    # Define the maximum number of episodes (steps)
    max_episodes = 10

    for episode in range(max_episodes):
        print("=" * 20 + f" Episode {episode + 1} of {max_episodes} " + "=" * 20 + "\n")
        for agent_id, agent in env.agents.items():
            observation = env.get_agent_position(agent_id)
            agent.observation = f"Your current position is: {observation}"

        for agent_id, agent in env.agents.items():
            time.sleep(2)
            print(f"Agent {agent_id} Observation: {agent.observation}")

            # Agent makes a decision based on the current observation
            action_dict = agent.step()

            # Check if there is a message to send and distribute it to other agents
            message = action_dict.get("message", "")
            if message:
                for other_agent_id, other_agent in env.agents.items():
                    if other_agent_id != agent_id:  # Only send to other agents
                        other_agent.add_inbox_message(agent.name, message)

            # Execute the action in the environment
            agent.observation = env.step(agent.id, action_dict["action_name"])

        if env.terminated:
            break

        print("\n" + "=" * 50 + "\n")

    # Final summary
    if env.terminated:
        print("Simulation Complete: The agent successfully reached the target position!")
    else:
        print("Simulation Complete: The agent did not reach the target position within the maximum number of episodes.")


if __name__ == '__main__':
    # Define action space, including the new "skip" action
    actions = [
        Action(name="north", description="Move one step upward on the grid. Increases the y-coordinate by 1"),
        Action(name="south", description="Move one step downward on the grid. Decreases the y-coordinate by 1"),
        Action(name="west", description="Move one step to the left on the grid. Decreases the x-coordinate by 1"),
        Action(name="east", description="Move one step to the right on the grid. Increases the x-coordinate by 1"),
        Action(name="skip", description="Do nothing and skip this step"),
    ]

    unified_goal = (
        "PRIMARY GOAL: Coordinate with other agents to ensure each agent occupies a different corner of the gridworld.\n\n"
        "SPECIFIC INSTRUCTIONS:\n"
        "1. Choose an unclaimed corner to move toward\n"
        "2. Communicate your intended corner to other agents\n"
        "3. If multiple agents choose the same corner, negotiate and reassign\n"
        "4. Keep track of which corners are claimed by others\n"
        "5. Move efficiently toward your chosen corner\n\n"
        "CORNERS: There are four corners in the grid:\n"
        "- Top-left corner\n"
        "- Top-right corner\n"
        "- Bottom-left corner\n"
        "- Bottom-right corner\n\n"
        "SUCCESS CRITERIA:\n"
        "- Each agent must occupy a unique corner position\n"
        "- The simulation continues running until ALL agents are in their final corners\n"
        "- If you can still observe the environment, the task is NOT complete - keep coordinating!\n"
        "Use your observations to determine the exact corner coordinates."
    )

    # Assign normal, everyday names to the agents
    agent_names = ["Alice", "Bob", "Charlie", "Diana"]

    # Initialize grid size and define corners
    grid_size = (6, 6)
    corners = [(0, 0), (0, grid_size[1] - 1), (grid_size[0] - 1, 0), (grid_size[0] - 1, grid_size[1] - 1)]


    # Function to generate unique random start positions
    def generate_unique_positions(num_agents, grid_size):
        positions = set()
        while len(positions) < num_agents:
            pos = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
            positions.add(pos)
        return list(positions)


    # Generate random start positions for the agents
    start_positions = generate_unique_positions(4, grid_size)

    # Create four agents with normal names, random start positions, and detailed instructions
    agents = {}
    for i in range(4):
        variables = {
            "name": agent_names[i],
            "goal": unified_goal,
            "agent_names": agent_names,
            "n_agents": len(agent_names),
            "gridworld_size": grid_size
        }

        agent = Agent(
            agent_id=i,
            name=agent_names[i],
            variables=variables,
            action_space=actions,
            start_position=start_positions[i],
            color=(0, 0, 0), # does not work yet
            backend_model="llama3-groq-70b-8192-tool-use-preview"
        )

        agent.create_system_prompt("gridworld_system_prompt")
        agents[agent.id] = agent

    # Termination condition: All agents have reached a unique corner
    def termination_condition(env):
        """Check if all agents have reached a unique corner."""
        reached_corners = set()
        for agent in env.agents.values():
            if agent.position in corners:
                reached_corners.add(agent.position)
        return len(reached_corners) == len(corners)


    # Initialize the gridworld environment
    env = ComplexGridworld(grid_size=grid_size, agents=agents)
    env.register_termination_callback(termination_condition)

    # Visual representation for the GUI (e.g., marking corner positions)
    for corner in corners:
        env[corner[0], corner[1]] = Square(
            items=[Item(item_type="target", color=(0, 0, 100), shape="circle")]
        )

    gui = GUI(env)
    gui.run(run_simulation)
