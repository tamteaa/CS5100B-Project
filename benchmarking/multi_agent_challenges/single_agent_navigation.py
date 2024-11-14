import random
import pandas as pd
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld, Square, Item
from src.agent.base_agent import Agent
from dotenv import load_dotenv
from src.gui.gui import GUI
from src.agent.actions import Action
from benchmarking.benchmarking_utils import get_random_goal_position

load_dotenv("../../.env")
stats_df = pd.DataFrame()

messages_sent = {}

def track_message_in_df(agent_name, message, stats_df):
    agent_index = stats_df[stats_df["Agent Name"] == agent_name].index[0]
    current_messages = stats_df.loc[agent_index, "Messages Sent"]
    if current_messages:
        new_messages = current_messages + "\n" + message
    else:
        new_messages = message
    stats_df.loc[agent_index, "Messages Sent"] = new_messages

def initialize_stats_dataframe(agent_name):
    return pd.DataFrame({
        "Agent Name": [agent_name],
        "Steps Taken": [0],
        "Corners Visited": [0],
        "Reached Goal": [False],
        "Messages Sent": [""],
    })

def run_simulation(env: ComplexGridworld):
    print("Starting Single-Agent Gridworld Simulation...\n")

    max_episodes = 50
    agent = list(env.agents.values())[0]
    visited_corners = set()

    for episode in range(max_episodes):
        print(f"\nEpisode {episode + 1}/{max_episodes}")

        observation = env.get_agent_position(agent.id)
        agent.observation = f"Your current position is: {observation}"
        print(f"Agent Observation: {agent.observation}")

        action_dict = agent.step()

        message = action_dict.get("message", "")
        if message:
            track_message_in_df(agent.name, message, stats_df)

        agent.observation = env.step(agent.id, action_dict["action_name"])
        stats_df.loc[0, "Steps Taken"] += 1

        if observation in env.corners:
            visited_corners.add(observation)
            print(f"Visited corners: {visited_corners}")

        stats_df.loc[0, "Corners Visited"] = len(visited_corners)

        if len(visited_corners) == 4 and agent.position == agent.variables["goal_position"]:
            stats_df.loc[0, "Reached Goal"] = True
            break

        if env.terminated:
            break

    print("\nSimulation Complete!")
    print(stats_df)

if __name__ == '__main__':
    actions = [
        Action(name="north", description="Move one step upward on the grid."),
        Action(name="south", description="Move one step downward on the grid."),
        Action(name="west", description="Move one step to the left on the grid."),
        Action(name="east", description="Move one step to the right on the grid."),
        Action(name="skip", description="Do nothing and skip this step.")
    ]

    agent_name = "SoloAgent"
    grid_size = (6, 6)

    stats_df = initialize_stats_dataframe(agent_name)

    corners = [(0, 0), (0, grid_size[1] - 1), (grid_size[0] - 1, 0), (grid_size[0] - 1, grid_size[1] - 1)]
    goal_pos = get_random_goal_position(grid_size)

    start_position = random.choice(corners)

    unified_goal = (
        f"PRIMARY GOAL: Visit all four corners of the grid before heading to the goal position.\n\n"
        "SPECIFIC INSTRUCTIONS:\n"
        "1. Navigate to each of the four corners of the grid:\n"
        f"   - Top-left corner = {corners[0]}\n"
        f"   - Top-right corner = {corners[1]}\n"
        f"   - Bottom-left corner = {corners[2]}\n"
        f"   - Bottom-right corner = {corners[3]}\n"
        "2. After visiting all corners, move to the final goal position.\n"
        f"3. The goal position is: {goal_pos}\n"
        "4. Communicate your actions if needed.\n"
        "SUCCESS CRITERIA:\n"
        "1. All corners must be visited at least once.\n"
        "2. The agent must reach the goal position after visiting all corners."
    )

    variables = {
        "name": agent_name,
        "goal": unified_goal,
        "goal_position": goal_pos,
        "gridworld_size": grid_size,
        "corners": corners
    }

    agent = Agent(
        agent_id=0,
        name=agent_name,
        variables=variables,
        action_space=actions,
        start_position=start_position,
        backend_model="llama3-groq-70b-8192-tool-use-preview"
    )

    agent.create_system_prompt("gridworld_system_prompt")
    agents = {agent.id: agent}

    env = ComplexGridworld(grid_size=grid_size, agents=agents)
    env.corners = corners
    gui = GUI(env)

    gui.run(run_simulation)
