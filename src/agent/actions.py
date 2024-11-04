from typing import List, Dict


class Action:
    def __init__(self, name: str, description: str = '', parameters: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    def format(self) -> str:
        """Format the action as a string."""
        if self.parameters:
            params_formatted = ', '.join(f"{key}: {value}" for key, value in self.parameters.items())
            return f"**{self.name}**: {self.description} | Parameters: {params_formatted}"
        else:
            return f"**{self.name}**: {self.description}"


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
