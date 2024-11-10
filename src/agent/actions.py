from typing import List, Dict, Optional, Any
import yaml

class Action:
    def __init__(self, name: str, description: str = '', parameters: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.steps = int(self.parameters.get("steps", 1))

    def format(self) -> str:
        """Format the action as a string."""
        params_formatted = ', '.join(f"{key}: {value}" for key, value in self.parameters.items())
        return f"**{self.name}**: {self.description} | Parameters: {params_formatted}"

    @classmethod
    def from_llm_output(cls, output: Dict[str, Any]) -> Optional['Action']:
        """Create an Action instance from an LLM output."""
        action_name = output.get("action_name")
        action_parameters = output.get("action_parameters", {})

        # Create Action instance with extracted name and parameters
        action = cls(name=action_name, parameters=action_parameters)
        return action

    def is_valid(self, grid_size: tuple, current_position: tuple) -> bool:
        """Check if an action is valid within the grid size and current position context."""
        x, y = current_position
        steps = int(self.parameters.get("steps", 1))

        if self.name == "north" and y + steps < grid_size[1]:
            return True
        elif self.name == "south" and y - steps >= 0:
            return True
        elif self.name == "east" and x + steps < grid_size[0]:
            return True
        elif self.name == "west" and x - steps >= 0:
            return True
        elif self.name == "skip":
            return True
        return False


class ActionManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.actions = {}

    def load_actions(self, env_name: str):
        """Load actions specific to the provided environment from the YAML configuration."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)

        action_configs = config.get(env_name, [])
        if not action_configs:
            print(f"No actions found for environment '{env_name}'.")
            return

        self.actions = {
            action_config['name']: Action(
                name=action_config['name'],
                description=action_config.get('description', ''),
                parameters=action_config.get('parameters', {})
            )
            for action_config in action_configs
        }

    def format_action(self, action_name: str) -> Optional[str]:
        """Format an action for LLM input."""
        action = self.actions.get(action_name)
        return action.format() if action else None

    def parse_llm_output(self, output: Dict[str, Any]) -> Optional[Action]:
        """Parse LLM output to get an Action instance with rationale and message."""
        action_name = output.get("action_name")
        if action_name in self.actions:
            action = Action.from_llm_output(output)
            return action
        return None

    def validate_action(self, action: Action, grid_size: tuple, current_position: tuple) -> bool:
        """Validate if an action is feasible within the grid context."""
        return action.is_valid(grid_size, current_position)


def format_actions(actions: List[Action]) -> str:
    """Format multiple actions with a 'actions:' header."""
    formatted_actions = '\n'.join(f"- {action.format()}" for action in actions)
    return f"actions:\n{formatted_actions}"


if __name__ == '__main__':
    # Creating Action instances with descriptions and empty parameters
    actions = [
        Action(name="north", description="Move one step upward on the grid."),
        Action(name="south", description="Move one step downward on the grid."),
        Action(name="west", description="Move one step to the left on the grid."),
        Action(name="east", description="Move one step to the right on the grid.")
    ]

    # Printing the formatted actions using the new function
    formatted_actions = format_actions(actions)
    print(formatted_actions)
