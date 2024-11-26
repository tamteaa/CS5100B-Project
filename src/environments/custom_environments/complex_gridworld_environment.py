import numpy as np
from typing import Tuple, Dict, List
from src.agent.base_agent import Agent  # Make sure you have the correct import path for your Agent class


class Item:
    def __init__(self, item_type: str, color: Tuple[int, int, int], shape: str, allowed_agent_id: int = None):
        self.item_type = item_type
        self.color = color
        self.shape = shape
        self.allowed_agent_id = allowed_agent_id

    def __repr__(self):
        agent_str = f", allowed_agent_id={self.allowed_agent_id}" if self.allowed_agent_id is not None else ""
        return f"Item(type={self.item_type}, color={self.color}, shape={self.shape}{agent_str})"

    def can_be_picked_up_by(self, env: 'ComplexGridworld', agent_id: int) -> bool:
        if self.allowed_agent_id is None:
            return True
        agent = env.agents.get(agent_id)
        if agent is None:
            return False
        return agent_id == self.allowed_agent_id


class Square:
    def __init__(self, obstacle: bool = False, items: List[Item] = None):
        self.obstacle = obstacle
        self.agents = []
        self.items = items if items else []

    def is_empty(self):
        return not self.obstacle and self.agents is None and not self.items

    def has_items(self):
        return bool(self.items)

    def pick_up_item(self):
        return self.items.pop() if self.items else None


class ComplexGridworld:
    def __init__(
            self,
            grid_size: Tuple[int, int] = (10, 10),  # (width, height)
            agents: Dict[int, Agent] = None,
            obstacles: List[Tuple[int, int]] = None,  # List of (x,y) positions
            items: Dict[Tuple[int, int], List[Item]] = None  # Dict of (x,y) positions to items
    ):
        self.grid_size = grid_size
        self.grid = [[Square() for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        self.agents = agents if agents else {}

        self.termination_callbacks = []
        self.terminated = False

        # Initialize agent positions
        for agent_id, agent in self.agents.items():
            x, y = agent.position
            self.grid[x][y].agents.append(agent)

        # Place obstacles
        if obstacles:
            for (x, y) in obstacles:
                self.grid[x][y].obstacle = True

        # Place items with validation
        if items:
            for (x, y), item_list in items.items():
                if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
                    raise ValueError(f"Item position ({x}, {y}) is out of bounds for grid size {grid_size}")
                self.grid[x][y].items.extend(item_list)

        self.max_episodes = 0
        self.variables = {"group_messages": []}
        self.score = 0
        self.db_manager = None
        self.use_db = False
        self.sim_id = 0
        self.name = None

    def __getitem__(self, key):
        """Support both single index and tuple index access."""
        if isinstance(key, tuple):
            x, y = key
            # Add bounds checking
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                return self.grid[x][y]
            raise IndexError(f"Position ({x}, {y}) is out of bounds for grid size {self.grid_size}")
        return self.grid[key]

    def __setitem__(self, key, value):
        """Support both single index and tuple index assignment."""
        if isinstance(key, tuple):
            x, y = key
            # Add bounds checking
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.grid[x][y] = value
            else:
                raise IndexError(f"Position ({x}, {y}) is out of bounds for grid size {self.grid_size}")
        else:
            self.grid[key] = value

    def reset(self):
        """Resets the environment to its initial state."""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                self.grid[row][col].agents = []

        for agent_id, agent in self.agents.items():
            row, col = agent.position
            self.grid[row][col].agents = agent

        return None

    def step(self, agent_id: int, action: str) -> str:
        """Execute a step for the specified agent."""
        if len(self.termination_callbacks) == 0:
            raise ValueError("must have at least one termination callback")

        if all(callback(self) for callback in self.termination_callbacks):
            self.terminated = True
            return "The environment has reached a termination condition."

        agent = self.agents.get(agent_id)
        if not agent:
            return f"Agent ID {agent_id} not found in the environment."

        valid_actions = [action.name for action in agent.action_space]
        if action not in valid_actions:
            return f"Invalid action: '{action}'. Valid actions are {valid_actions}."

        x, y = agent.position
        new_x, new_y = x, y
        current_square = self.grid[x][y]

        if current_square.has_items():
            current_items_str = ", ".join(str(item) for item in current_square.items)
            item_observation = f"\nYou are in square ({x}, {y}). There are items here: {current_items_str}."
        else:
            item_observation = f"\nYou are in square ({x}, {y}). There are no items here."

        if action == "skip":
            return "You skipped your turn." + item_observation

        if action == 'north':
            new_y = min(self.grid_size[1] - 1, y + 1)
        elif action == 'south':
            new_y = max(0, y - 1)
        elif action == 'east':
            new_x = min(self.grid_size[0] - 1, x + 1)
        elif action == 'west':
            new_x = max(0, x - 1)
        elif action == 'pick':
            if self.grid[x][y].has_items():
                item = self.grid[x][y].items[-1]
                if item and item.item_type == "item":
                    if item.can_be_picked_up_by(self, agent_id):
                        agent.item = self.grid[x][y].pick_up_item()
                        return "You pick up the item"
                    else:
                        return f"You (Agent {agent.name}) are not authorized to pick up this item"
                else:
                    return "No item here"
            else:
                return "No item here"
        elif action == 'drop':
            if agent.item:
                self.grid[x][y].items.append(agent.item)
                agent.item = None
                return "You drop off the item"
            else:
                return "You are not holding any item"

        if (new_x, new_y) == (x, y):
            return f"Agent {agent.name} tried to move '{action}', but it cannot move further in that direction." + item_observation

        target_square = self.grid[new_x][new_y]
        if target_square.obstacle:
            return "Cannot move into obstacle."

        self.grid[x][y].agents.remove(agent)
        agent.position = (new_x, new_y)
        self.grid[new_x][new_y].agents.append(agent)

        if target_square.has_items():
            current_items_str = ", ".join(str(item) for item in target_square.items)
            item_observation = f"\nYou are in square ({new_x}, {new_y}). There are items here: {current_items_str}."
        else:
            item_observation = f"\nYou are in square ({new_x}, {new_y}). There are no items here."

        observation = f"Agent {agent.name} moved '{action}' from {(x, y)} to {(new_x, new_y)}." + item_observation

        if all(callback(self) for callback in self.termination_callbacks):
            self.terminated = True
            return "The environment has reached a termination condition." + item_observation

        agent.variables["score"] = self.score

        return observation

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
