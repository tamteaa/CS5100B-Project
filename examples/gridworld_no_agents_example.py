from src.environments.custom_environments.gridworld_environment import GridworldEnvironment

if __name__ == '__main__':
    # Define start positions for multiple agents
    start_positions = {0: (0, 0), 1: (4, 4)}

    # Initialize the gridworld environment
    env = GridworldEnvironment(grid_size=(5, 5), start_positions=start_positions)

    # Render initial state
    env.render()

    observation = env.step(0, 'north')
    print(observation)

    observation = env.step(1, 'south')
    print(observation)

    # Render updated state
    env.render()

    # Get specific agent positions
    print(env.get_agent_position(0))  # Position of agent 0
    print(env.get_agent_position(1))  # Position of agent 1
