from src.environments.custom_environments.complex_gridworld_environment import Square, Item


def random_points_multi_agent_navigation_scoring_function(env):
    targets_str = env.variables["target_positions"]
    targets_list = eval(targets_str)
    targets = set(map(tuple, targets_list))

    # Add target items to the squares using (x,y)
    for target in targets:
        x, y = target
        if len(env[x, y].items) == 0:
            env[x, y].items = [
                Item(item_type="target", color=(0, 0, 50, 128), shape="circle")
            ]

    occupied_targets = set()

    for agent in env.agents.values():
        agent_pos = tuple(agent.position)  # Already in (x,y)
        if agent_pos in targets:
            occupied_targets.add(agent_pos)

    points_per_target = 100 / len(targets)
    env.score = len(occupied_targets) * points_per_target

    return len(occupied_targets) == len(targets)


def single_agent_navigation_scoring_function(env):
    target_position = tuple(env.variables.get("target_position", None))
    if target_position is None:
        raise ValueError("target position cannot be None")

    x, y = target_position
    env[x, y] = Square(
        items=[Item(item_type="target", color=(0, 0, 100), shape="circle")]
    )

    for agent in env.agents.values():
        if agent.position != target_position:
            return False

    env.score = 100
    return True


def multi_agent_navigation_scoring_function(env):
    # Define corners in (x,y) format
    corners = {
        (0, 0),             # Bottom-left
        (env.grid_size[0]-1, 0),          # Bottom-right
        (0, env.grid_size[1]-1),          # Top-left
        (env.grid_size[0]-1, env.grid_size[1]-1)  # Top-right
    }

    for corner in corners:
        x, y = corner
        if len(env[x, y].items) == 0:
            env[x, y].items = [
                Item(item_type="target", color=(0, 0, 50, 128), shape="circle")
            ]

    occupied_corners = set()

    for agent in env.agents.values():
        agent_position = tuple(agent.position)  # Already in (x,y)
        if agent_position in corners:
            occupied_corners.add(agent_position)

    env.score = len(occupied_corners) * 25
    return len(occupied_corners) == len(corners)


def align_alphabetically_task_scoring_function(env):
    # Starting at (x=0, y=0)
    start_x, start_y = 0, 0

    agents_sorted = sorted(env.agents.values(), key=lambda agent: agent.name)
    correct_positions = 0

    for index, agent in enumerate(agents_sorted):
        expected_position = (start_x + index, start_y)  # Moving along x-axis
        if agent.position == expected_position:
            correct_positions += 1

    total_agents = len(agents_sorted)
    env.score = (correct_positions / total_agents) * 100 if total_agents > 0 else 0

    return correct_positions == total_agents


def pick_item_scoring_function(env):
    """Scoring function that handles target positions and optional permissions"""
    # Get target positions and item positions
    target_positions = env.variables.get("target_positions", None)
    item_positions = env.variables.get("item_positions", None)
    if target_positions is None or item_positions is None:
        raise ValueError("target_positions or item_positions not found")

    target_positions = eval(target_positions)
    item_positions = eval(item_positions)

    # Check if using permissions system
    use_permissions = env.variables.get("use_permissions", False)

    # Place target markers
    for target_pos in target_positions:
        x, y = target_pos
        if len(env[x, y].items) == 0:
            env[x, y].items = [
                Item(item_type="target", color=(0, 0, 50, 128), shape="circle")
            ]

    # Place items in their initial positions once
    if not hasattr(env, 'items_placed'):
        env.items_placed = True
        for i, pos in enumerate(item_positions):
            x, y = pos
            if use_permissions:
                env[x, y].items = [
                    Item(item_type="item", color=(200, 0, 0), shape="triangle", allowed_agent_id=i)
                ]
            else:
                env[x, y].items = [
                    Item(item_type="item", color=(200, 0, 0), shape="triangle")
                ]

    # Count filled targets
    total_targets = len(target_positions)
    filled_targets = 0

    # For each target position, check if there's an item in the same square
    for target_pos in target_positions:
        x, y = target_pos
        # Check if any item is in this target position
        has_item = False
        for item in env[x, y].items:
            if item.item_type == "item":
                has_item = True
                break
        if has_item:
            filled_targets += 1

    # Calculate score based on how many targets have items
    if total_targets > 0:
        env.score = (filled_targets / total_targets) * 100
    else:
        env.score = 0

    # Success when all targets have items
    return filled_targets == total_targets


DEFAULT_CONFIGS = {
    "single_agent_navigation": {
        "yaml_file": "./configs/single_agent_navigation.yaml",
        "termination_condition": single_agent_navigation_scoring_function,
    },
    "multi_agent_navigation": {
        "yaml_file": "./configs/multi_agent_navigation.yaml",
        "termination_condition": multi_agent_navigation_scoring_function,
    },
    "align_alphabetically_task": {
        "yaml_file": "./configs/alphabetical_order.yaml",
        "termination_condition": align_alphabetically_task_scoring_function,
    },
    "random_points_multi_agent_navigation": {
        "yaml_file": "./configs/random_points_multi_agent_navigation.yaml",
        "termination_condition": random_points_multi_agent_navigation_scoring_function,
    },
    "single_agent_pick_item": {
        "yaml_file": "./configs/single_agent_pick_item.yaml",
        "termination_condition": pick_item_scoring_function,
    },
    "multi_agent_pick_item": {
        "yaml_file": "./configs/multi_agent_pick_item.yaml",
        "termination_condition": pick_item_scoring_function,
    },
    "multi_agent_pick_item_permissions": {
        "yaml_file": "./configs/multi_agent_permissions_pick_up.yaml",
        "termination_condition": pick_item_scoring_function,
    },
}