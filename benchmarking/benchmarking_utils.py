import random


def get_random_goal_position(grid_size):
    return random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)


def get_random_start_positions(num_agents, grid_size, goal_pos):
    positions = set()
    while len(positions) < num_agents:
        pos = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        while pos == goal_pos:
            pos = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        positions.add(pos)
    return list(positions)