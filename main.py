from src.envwrapper.simulator import Simulator
from dotenv import load_dotenv
from src.environments.custom_environments.complex_gridworld_environment import Square, Item
from src.agent.backend import Provider, TogetherModels, LocalModels, GroqModels


def random_points_multi_agent_navigation_scoring_function(env):
    # Get target positions from environment variables
    targets_str = env.variables["target_positions"]
    targets_list = eval(targets_str)  # Safely converts string representation to list of tuples
    targets = set(map(tuple, targets_list))

    # Add target items to the squares
    for target in targets:
        if len(env[target[0], target[1]].items) == 0:
            env[target[0], target[1]].items = [
                Item(item_type="target", color=(0, 0, 50, 128), shape="circle")
            ]

    # Track occupied targets
    occupied_targets = set()

    # Check each agent's position
    for agent in env.agents.values():
        agent_pos = tuple(agent.position)
        if agent_pos in targets:
            occupied_targets.add(agent_pos)

    # Calculate score - each unique target found is worth equal points
    points_per_target = 100 / len(targets)
    env.score = len(occupied_targets) * points_per_target

    # Return True if all targets are found
    return len(occupied_targets) == len(targets)

def single_agent_navigation_scoring_function(env):
    # Retrieve the target position from the environment's variables
    target_position = tuple(env.variables.get("target_position", None))
    if target_position is None:
        raise ValueError("targe position cannot be None")

    env[target_position[0], target_position[1]] = Square(
        items=[
            Item(item_type="target", color=(0, 0, 100), shape="circle")
        ]
    )
    # Check if all agents are in the target position
    for agent in env.agents.values():
        if agent.position != target_position:
            return False

    # If all agents are in the target position, return True
    env.score = 100
    return True


def multi_agent_navigation_scoring_function(env):
    # Define the four unique corner positions based on the grid size
    print(env.variables)
    corners = {
        (0, 0),  # South-West Corner
        (0, env.grid_size[1] - 1),  # South-East Corner
        (env.grid_size[0] - 1, 0),  # North-West Corner
        (env.grid_size[0] - 1, env.grid_size[1] - 1)  # North-East Corner
    }

    for corner in corners:
        if len(env[corner[0], corner[1]].items) == 0:
            env[corner[0], corner[1]].items = [
                    Item(item_type="target", color=(0, 0, 50, 128), shape="circle")
            ]

    # Create a set to track which corners have been occupied by agents
    occupied_corners = set()

    # Iterate over all agents and check their positions
    for agent in env.agents.values():
        agent_position = tuple(agent.position)
        if agent_position in corners:
            occupied_corners.add(agent_position)

    # Score 25 points for each unique corner occupied
    env.score = len(occupied_corners) * 25

    # Check if all corners have been occupied by agents
    if len(occupied_corners) == len(corners):
        return True

    return False


def align_alphabetically_task_scoring_function(env):
    # Define the starting position at the top-left corner
    start_y, start_x = 0, 0

    # Get agents sorted alphabetically by name
    agents_sorted = sorted(env.agents.values(), key=lambda agent: agent.name)
    correct_positions = 0

    # Check each agent's position and count correct ones
    for index, agent in enumerate(agents_sorted):
        expected_position = (start_y, start_x + index)
        if agent.position == expected_position:
            correct_positions += 1

    # Calculate percentage score
    total_agents = len(agents_sorted)
    env.score = (correct_positions / total_agents) * 100 if total_agents > 0 else 0

    # Return True only if all agents are in correct position
    return correct_positions == total_agents


# Load the GROQ API KEY from a .env file
load_dotenv(".env")


if __name__ == "__main__":
    # Usage
    backend_provider = Provider.GROQ
    backend_model = GroqModels.LLAMA_8B

    configs = {
        "single_agent_navigation": {
            "yaml_file": "./configs/single_agent_navigation.yaml",
            "termination_condition": single_agent_navigation_scoring_function,
            "backend_provider": backend_provider,
            "backend_model": backend_model
        },
        "multi_agent_navigation": {
            "yaml_file": "./configs/multi_agent_navigation.yaml",
            "termination_condition": multi_agent_navigation_scoring_function,
            "backend_provider": backend_provider,
            "backend_model": backend_model
        },
        "align_alphabetically_task": {
            "yaml_file": "./configs/alphabetical_order.yaml",
            "termination_condition": align_alphabetically_task_scoring_function,
            "backend_provider": backend_provider,
            "backend_model": backend_model
        },
        "random_points_multi_agent_navigation": {
            "yaml_file": "./configs/random_points_multi_agent_navigation.yaml",
            "termination_condition": random_points_multi_agent_navigation_scoring_function,
            "backend_provider": backend_provider,
            "backend_model": backend_model
        },
    }

    simulator = Simulator(
        use_db=False,
        use_gui=True,
        configs=configs
    )

    num_simulations = 5
    # list of length num_simulations with each score (x/100)
    scores = simulator.run("random_points_multi_agent_navigation", num_simulations)
    print(scores)
