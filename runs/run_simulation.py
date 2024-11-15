from src.envwrapper.simulator import Simulator
from dotenv import load_dotenv
from src.environments.custom_environments.complex_gridworld_environment import Square, Item


def is_agent_in_position(env):
    # Retrieve the target position from the environment's variables
    target_position = tuple(env.variables.get("target_position", [0, 0]))

    # Check if all agents are in the target position
    for agent in env.agents.values():
        if agent.position != target_position:
            return False

    # If all agents are in the target position, return True
    env.score = 1
    return True



def all_agents_in_corners(env):
    # Define the four unique corner positions based on the grid size
    corners = {
        (0, 0),  # South-West Corner
        (0, env.grid_size[1] - 1),  # South-East Corner
        (env.grid_size[0] - 1, 0),  # North-West Corner
        (env.grid_size[0] - 1, env.grid_size[1] - 1)  # North-East Corner
    }

    for corner in corners:
        env[corner[0], corner[1]] = Square(
            items=[
                Item(item_type="target", color=(0, 0, 100), shape="circle")]
        )

    # Create a set to track which corners have been occupied by agents
    occupied_corners = set()

    # Iterate over all agents and check their positions
    for agent in env.agents.values():
        agent_position = tuple(agent.position)
        if agent_position in corners:
            occupied_corners.add(agent_position)

    # Check if all corners have been occupied by agents
    if len(occupied_corners) == len(corners):
        env.score = 1
        return True

    return False


def are_agents_in_alphabetical_order(env):
    # Define the starting position at the top-left corner
    start_y, start_x = 0, 0

    # Retrieve a sorted list of agents by their names in alphabetical order
    agents_sorted = sorted(env.agents.values(), key=lambda agent: agent.name)

    # Iterate over the sorted agents and check their positions
    for index, agent in enumerate(agents_sorted):
        expected_position = (start_y, start_x + index)

        # If the agent's position does not match the expected position, return False
        if agent.position != expected_position:
            return False

    # If all agents are in the correct alphabetical order, set the score and return True
    env.score = 1
    return True


# Load the GROQ API KEY from a .env file
load_dotenv("../.env")


if __name__ == "__main__":
    configs = {
        "single_agent_navigation": {
            "yaml_file": "single_agent_navigation.yaml",
            "termination_condition": is_agent_in_position,
            "backend_model_id": "llama-3.1-8b-instant"
        },
        "multi_agent_navigation": {
            "yaml_file": "multi_agent_navigation.yaml",
            "termination_condition": all_agents_in_corners,
            "backend_model_id": "llama-3.2-11b-vision-preview"
        },
        "align_alphabetically_task": {
            "yaml_file": "alphabetical_order.yaml",
            "termination_condition": are_agents_in_alphabetical_order,
            "backend_model_id": "llama-3.2-11b-vision-preview"
        },
    }
    simulator = Simulator(
        use_db=False,
        use_gui=True,
        configs=configs
    )

    result = simulator.run("align_alphabetically_task")
    print(result)
