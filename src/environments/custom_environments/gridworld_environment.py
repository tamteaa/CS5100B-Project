import numpy as np
from typing import Tuple, List, Dict


class GridworldEnvironment:
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), start_positions: Dict[int, Tuple[int, int]] = None):
        """
        Initializes the Gridworld environment with multiple agents.

        :param grid_size: Tuple representing the dimensions of the grid (rows, cols).
        :param start_positions: Dictionary representing the starting positions of agents
                                where keys are agent IDs and values are (row, col) tuples.
        """
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.start_positions = start_positions if start_positions else {0: (0, 0)}
        self.agent_positions = self.start_positions.copy()

        # Place agents in their starting positions
        for agent_id, pos in self.agent_positions.items():
            self.grid[pos] = agent_id + 1  # Representing agent positions uniquely

    def reset(self):
        """
        Resets the environment to the initial state with all agents at their starting positions.
        """
        self.grid = np.zeros(self.grid_size)
        self.agent_positions = self.start_positions.copy()

        # Place agents back in their starting positions
        for agent_id, pos in self.agent_positions.items():
            self.grid[pos] = agent_id + 1
        return self.agent_positions
    def step(self, agent_id: int, action: str) -> str:
        """
        Executes a step for the specified agent based on the given action.

        :param agent_id: ID of the agent to be moved.
        :param action: A string representing the action ('north', 'south', 'east', 'west').
        :return: A string providing feedback about the action's result.
        """
        if agent_id not in self.agent_positions:
            return f"Agent ID {agent_id} not found in the environment."

        if action not in ['north', 'south', 'east', 'west']:
            return f"Invalid action: '{action}'. Valid actions are ['north', 'south', 'east', 'west']."

        # Get the current position of the agent
        current_pos = self.agent_positions[agent_id]
        row, col = current_pos

        # Determine the new position based on the action
        new_row, new_col = row, col
        if action == 'north':
            new_row = min(self.grid_size[0] - 1, row + 1)  # Move up (increase row)
        elif action == 'south':
            new_row = max(0, row - 1)  # Move down (decrease row)
        elif action == 'east':
            new_col = min(self.grid_size[1] - 1, col + 1)  # Move right (increase col)
        elif action == 'west':
            new_col = max(0, col - 1)  # Move left (decrease col)

        # Check if the new position is different from the current position
        if (new_row, new_col) == (row, col):
            return f"Agent {agent_id} tried to move '{action}', but it cannot move further in that direction."

        # Update the grid and agent's position
        self.grid[current_pos] = 0  # Clear the old position
        self.agent_positions[agent_id] = (new_row, new_col)
        self.grid[new_row, new_col] = agent_id + 1  # Mark the new position

        # Provide feedback about the action's result
        return f"Agent {agent_id} moved '{action}' from {current_pos} to {(new_row, new_col)}."

    def render(self):
        """
        Renders the current state of the grid with (0, 0) at the bottom-left.
        """
        # Flip the grid vertically for bottom-left origin display
        flipped_grid = np.flipud(self.grid)

        # Create a visual representation of the grid
        for row in flipped_grid:
            print(' '.join(f"{int(cell):2}" for cell in row))

    def get_state(self):
        """
        Returns the current state of the grid.
        """
        return self.grid.copy()

    def get_agent_position(self, agent_id: int):
        """
        Returns the current position of the specified agent.

        :param agent_id: ID of the agent.
        :return: The position of the agent.
        """
        return self.agent_positions.get(agent_id, None)

    def get_all_agent_positions(self) -> Dict[int, Tuple[int, int]]:
        """
        Returns the positions of all agents.

        :return: Dictionary with agent IDs as keys and their positions as values.
        """
        return self.agent_positions.copy()

