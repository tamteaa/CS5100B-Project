import numpy as np
from typing import Tuple, Dict, List
from src.agent.base_agent import Agent  # Make sure you have the correct import path for your Agent class


class Item:
    def __init__(self, item_type: str, color: Tuple[int, int, int], shape: str):
        self.item_type = item_type
        self.color = color
        self.shape = shape

    def __repr__(self):
        return f"Item(type={self.item_type}, color={self.color}, shape={self.shape})"


class Square:
    def __init__(self, obstacle: bool = False, items: List[Item] = None):
        self.obstacle = obstacle
        self.agent = None  # Store the entire Agent object
        self.items = items if items else []

    def is_empty(self):
        return not self.obstacle and self.agent is None and not self.items

    def has_items(self):
        return bool(self.items)

    def pick_up_item(self):
        return self.items.pop() if self.items else None


class ComplexGridworld:
    def __init__(
            self,
            grid_size: Tuple[int, int] = (10, 10),
            agents: Dict[int, Agent] = None,
            obstacles: List[Tuple[int, int]] = None,
            items: Dict[Tuple[int, int], List[Item]] = None
    ):
        self.grid_size = grid_size
        self.grid = [[Square() for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        self.agents = agents if agents else {}

        # environment termination
        self.termination_callbacks = []
        self.terminated: bool = False

        # Initialize agent positions
        for agent_id, agent in self.agents.items():
            row, col = agent.position
            self.grid[row][col].agent = agent

        # Place obstacles
        if obstacles:
            for (row, col) in obstacles:
                self.grid[row][col].obstacle = True

        # Place items
        if items:
            for (row, col), item_list in items.items():
                self.grid[row][col].items.extend(item_list)

        self.max_episodes = 0
        self.variables = {}

        self.score = 0

    def __getitem__(self, key):
        """Support both single index and tuple index access."""
        if isinstance(key, tuple):
            row, col = key
            return self.grid[row][col]
        return self.grid[key]

    def __setitem__(self, key, value):
        """Support both single index and tuple index assignment."""
        if isinstance(key, tuple):
            row, col = key
            self.grid[row][col] = value
        else:
            self.grid[key] = value

    def reset(self):
        """Resets the environment to its initial state."""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                self.grid[row][col].agent = None

        for agent_id, agent in self.agents.items():
            row, col = agent.position
            self.grid[row][col].agent = agent

        return None

    def step(self, agent_id: int, action: str) -> str:
        """Execute a step for the specified agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return f"Agent ID {agent_id} not found in the environment."

        if action not in ['north', 'south', 'east', 'west', 'pick', 'drop', 'skip']:
            return f"Invalid action: '{action}'. Valid actions are ['north', 'south', 'east', 'west', 'pick', 'drop', 'skip']."

        row, col = agent.position
        new_row, new_col = row, col

        if action == 'north':
            new_row = min(self.grid_size[0] - 1, row + 1)
        elif action == 'south':
            new_row = max(0, row - 1)
        elif action == 'east':
            new_col = min(self.grid_size[1] - 1, col + 1)
        elif action == 'west':
            new_col = max(0, col - 1)
        elif action == 'pick':
            if self.grid[row][col].has_items():
                return "You pick up the item"
            else:
                return "No item here"
        elif action == 'drop':
            return "You drop off the item"

        if action == "skip":
            return "You skipped your turn."

        # Check if the new position is different from the current position
        if (new_row, new_col) == (row, col):
            return f"Agent {agent.name} tried to move '{action}', but it cannot move further in that direction."

        target_square = self.grid[new_row][new_col]
        if target_square.obstacle:
            return "Cannot move into obstacle."
        # Update grid and agent's position
        self.grid[row][col].agent = None
        agent.position = (new_row, new_col)
        self.grid[new_row][new_col].agent = agent

        if len(self.termination_callbacks) == 0:
            raise ValueError("must have at least one termination callback")

        # Check termination conditions
        if all(callback(self) for callback in self.termination_callbacks):
            self.terminated = True
            return "The environment has reached a termination condition."

        return f"Agent  {agent.name} moved '{action}' from {(row, col)} to {(new_row, new_col)}."

    def get_agent_position(self, agent_id: int):
        """Get the position of a specific agent."""
        agent = self.agents.get(agent_id)
        return agent.position if agent else None

    def register_termination_callback(self, func):
        """Register a termination callback."""
        self.termination_callbacks.append(func)

    def iter_agents(self):
        pass

    def set_agents_for_env(self, agents):
        self.agents = {i: obj for i, obj in enumerate(agents)}
