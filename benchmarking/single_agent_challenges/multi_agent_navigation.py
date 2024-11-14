import time
import pandas as pd
from benchmarking.benchmarking_utils import get_random_start_positions, get_random_goal_position
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld, Square, Item
from src.agent.base_agent import Agent
from dotenv import load_dotenv
from src.gui.gui import GUI
from src.agent.actions import Action


load_dotenv("../../.env")

messages_sent = {}
stat_df = pd.DataFrame()


def track_message(agent_name, message):
    if agent_name not in messages_sent:
        messages_sent[agent_name] = []
    messages_sent[agent_name].append(message)


def initialize_stats_dataframe(agent_names):
    return pd.DataFrame({
        "Agent Name": agent_names,
        "Steps Taken": [0] * len(agent_names),
        "Reached Goal": [False] * len(agent_names),
        "Turns to Goal": [None] * len(agent_names),
    })


def run_simulation(env: ComplexGridworld):
    print("Starting Gridworld Simulation...\n")

    max_episodes = 20

    for episode in range(max_episodes):
        print("=" * 20 + f" Episode {episode + 1} of {max_episodes} " + "=" * 20 + "\n")


        for agent_id, agent in env.agents.items():
            observation = env.get_agent_position(agent_id)
            agent.observation = f"Your current position is: {observation}"


        for agent_id, agent in sorted(env.agents.items(), key=lambda x: x[1].name):
            time.sleep(2)
            print(f"Agent {agent_id} ({agent.name}) Observation: {agent.observation}")


            action_dict = agent.step()


            message = action_dict.get("message", "")
            if message:
                for other_agent_id, other_agent in env.agents.items():
                    if other_agent_id != agent_id:
                        other_agent.add_inbox_message(agent.name, message)
                        track_message(agent.name, message)

            agent.observation = env.step(agent.id, action_dict["action_name"])

            stat_df.loc[stat_df["Agent Name"] == agent.name, "Steps Taken"] += 1


            if agent.position == agent.variables["goal_position"] and not \
            stat_df.loc[stat_df["Agent Name"] == agent.name, "Reached Goal"].values[0]:
                stat_df.loc[stat_df["Agent Name"] == agent.name, "Reached Goal"] = True
                stat_df.loc[stat_df["Agent Name"] == agent.name, "Turns to Goal"] = stat_df.loc[
                    stat_df["Agent Name"] == agent.name, "Steps Taken"]

        if env.terminated:
            break

        print("\n" + "=" * 50 + "\n")


    print("\nSimulation Complete!")
    print("=" * 30)
    print(stat_df)


    success_count = stat_df["Reached Goal"].sum()
    print(f"\nSuccess Rate: {success_count}/{len(stat_df)} agents reached the goal.")


    reached_in_order = [agent.name for agent in sorted(env.agents.values(), key=lambda x: x.name)
                        if stat_df.loc[stat_df["Agent Name"] == agent.name, "Reached Goal"].values[0]]
    print("\nOrder in which agents reached the goal (in alphabetical order):")
    print(reached_in_order)


if __name__ == '__main__':

    actions = [
        Action(name="north", description="Move one step upward on the grid."),
        Action(name="south", description="Move one step downward on the grid."),
        Action(name="west", description="Move one step to the left on the grid."),
        Action(name="east", description="Move one step to the right on the grid."),
        Action(name="skip", description="Do nothing and skip this step.")
    ]


    agent_names = ["Alice", "Bob", "Charlie", "Diana"]
    grid_size = (6, 6)
    stat_df = initialize_stats_dataframe(agent_names)


    goal_pos = get_random_goal_position(grid_size)


    start_positions = get_random_start_positions(len(agent_names), grid_size, goal_pos)


    unified_goal = (
        "PRIMARY GOAL: All agents must reach a shared goal position on the grid in alphabetical order.\n\n"
        "SPECIFIC INSTRUCTIONS:\n"
        "1. Determine your alphabetical position relative to the other agents.\n"
        "2. Wait until all agents before you in alphabetical order have reached the goal.\n"
        "3. Communicate with other agents to verify their status before making your move.\n"
        "4. Only proceed towards the goal when it is your turn.\n"
        "5. Once you reach the goal, communicate your arrival to the other agents.\n"
        "6. The simulation continues running until ALL agents reach the goal in the correct order.\n\n"
        "GOAL POSITION:\n"
        "- The shared goal position on the grid is: {goal_pos}\n\n"
        "SUCCESS CRITERIA:\n"
        "- All agents must reach the goal position.\n"
        "- Agents must arrive at the goal in alphabetical order based on their names.\n"
        "- If you can still observe the environment, the task is NOT complete - keep coordinating!"
    )


    agents = {}
    for i, name in enumerate(agent_names):
        variables = {
            "name": name,
            "goal": unified_goal.format(goal_pos=goal_pos),
            "agent_names": sorted(agent_names),
            "goal_position": goal_pos,
            "n_agents": len(agent_names),
            "gridworld_size": grid_size
        }

        agent = Agent(
            agent_id=i,
            name=name,
            variables=variables,
            action_space=actions,
            start_position=start_positions[i],
            backend_model="llama3-groq-70b-8192-tool-use-preview"
        )

        agent.create_system_prompt("gridworld_system_prompt")
        agents[agent.id] = agent



    def termination_condition(env):
        reached_order = []
        for agent_name in sorted(agent_names):
            for agent in env.agents.values():
                if agent.name == agent_name and agent.position == goal_pos:
                    reached_order.append(agent.name)


        return reached_order == sorted(agent_names)



    env = ComplexGridworld(grid_size=grid_size, agents=agents)
    env.register_termination_callback(termination_condition)


    env[goal_pos[0], goal_pos[1]] = Square(
        items=[Item(item_type="target", color=(0, 100, 0), shape="star")]
    )


    gui = GUI(env)
    gui.run(run_simulation)


    print("\nFinal Stats DataFrame:")
    print(stat_df)

    print("\nMessages sent by agents:")
    for agent_name, messages in messages_sent.items():
        print(f"{agent_name} sent messages: {messages}")