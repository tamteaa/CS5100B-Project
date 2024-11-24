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
            self.grid[row][col].agents.append(agent)

        # Place obstacles
        if obstacles:
            for (row, col) in obstacles:
                self.grid[row][col].obstacle = True

        # Place items
        if items:
            for (row, col), item_list in items.items():
                self.grid[row][col].items.extend(item_list)
            
            

        self.max_episodes = 0
        self.variables = {
            "group_messages": []
        }

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
                self.grid[row][col].agents = []

        for agent_id, agent in self.agents.items():
            row, col = agent.position
            self.grid[row][col].agents = agent

        return None

    def step(self, agent_id: int, action: str) -> str:
        """Execute a step for the specified agent."""
        if len(self.termination_callbacks) == 0:
            raise ValueError("must have at least one termination callback")

        # Check termination conditions
        if all(callback(self) for callback in self.termination_callbacks):
            self.terminated = True
            return "The environment has reached a termination condition."

        agent = self.agents.get(agent_id)
        if not agent:
            return f"Agent ID {agent_id} not found in the environment."

        valid_actions = [action.name for action in agent.action_space]
        if action not in valid_actions:
            return f"Invalid action: '{action}'. Valid actions are {valid_actions}."

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
                item = self.grid[row][col].pick_up_item()
                if item and item.item_type == "item":
                    agent.item = item
                    return "You pick up the item"
                else:
                    self.grid[row][col].items.append(item)
                    return "No item here"
            else:
                return "No item here"
        elif action == 'drop':
            if agent.item:
                self.grid[row][col].items.append(agent.item)
                agent.item = None
                return "You drop off the item"
            else:
                return "You are not holding any item"

        if action == "skip":
            current_square = self.grid[row][col]
            if current_square.has_items():
                items_str = ", ".join(str(item) for item in current_square.items)
                return f"You skipped your turn. There are items here: {items_str}"
            return "You skipped your turn."

        # Check if the new position is different from the current position
        if (new_row, new_col) == (row, col):
            return f"Agent {agent.name} tried to move '{action}', but it cannot move further in that direction."

        target_square = self.grid[new_row][new_col]
        if target_square.obstacle:
            return "Cannot move into obstacle."
        # Update grid and agent's position
        self.grid[row][col].agents.remove(agent)
        agent.position = (new_row, new_col)
        self.grid[new_row][new_col].agents.append(agent)

        #
        observation = f"Agent  {agent.name} moved '{action}' from {(row, col)} to {(new_row, new_col)}."

        if target_square.has_items():
            items_str = ", ".join(str(item) for item in target_square.items)
            observation += f"\nThere are items here: {items_str}"

        # Check termination conditions
        if all(callback(self) for callback in self.termination_callbacks):
            self.terminated = True
            return "The environment has reached a termination condition."

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
