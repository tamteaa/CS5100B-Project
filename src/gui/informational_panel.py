import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld


class InfoPanelView:
    def __init__(self, agents):
        self.agents = agents
        self.selected_agent = list(agents.keys())[0] if agents else None

        # Define colors for different roles
        self.backgrounds = {
            "system": (50, 50, 80),  # Dark blue
            "assistant": (50, 80, 50),  # Dark green
            "user": (80, 50, 50)  # Dark red
        }

    def format_message(self, message, parent):
        """Format a single message with role-specific styling"""
        role = message.get("role", "unknown")
        content = message.get("content", "")

        # Add spacer before each message
        dpg.add_spacer(height=5, parent=parent)

        # Create a colored background group
        with dpg.group(parent=parent):
            # Add colored background using a theme
            with dpg.theme() as item_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg,
                                        self.backgrounds.get(role, (60, 60, 60)))

            # Calculate dynamic height based on content
            text_length = len(content)
            # Assume roughly 50 characters per line with wrap at 360
            estimated_lines = (text_length // 45) + 1
            dynamic_height = max(85, 60 + (estimated_lines * 20))  # base height + lines * line height

            # Create child window with colored background
            with dpg.child_window(width=380, height=dynamic_height, parent=parent):
                dpg.bind_item_theme(dpg.last_item(), item_theme)

                # Add the role label with padding
                dpg.add_spacer(height=5)
                dpg.add_text(f"[{role.upper()}]", color=(255, 255, 255))
                dpg.add_spacer(height=8)
                # Add the content with padding
                dpg.add_text(content, wrap=350, color=(255, 255, 255))  # Slightly reduced wrap width
                dpg.add_spacer(height=5)

        # Add spacer after message
        dpg.add_spacer(height=5, parent=parent)

    def on_agent_selected(self, sender, value):
        self.selected_agent = value

    def update_agent_info(self, gridworld: ComplexGridworld, tag: str):
        # Clear existing text
        dpg.delete_item(tag, children_only=True)

        # Add agent selector dropdown
        dpg.add_text("Select Agent:", parent=tag)
        dpg.add_combo(
            items=list(self.agents.keys()),
            default_value=self.selected_agent,
            callback=self.on_agent_selected,
            width=200,
            parent=tag
        )

        dpg.add_spacer(height=15, parent=tag)
        dpg.add_separator(parent=tag)
        dpg.add_spacer(height=15, parent=tag)

        # Display selected agent's information
        if self.selected_agent is not None:
            agent = self.agents[self.selected_agent]
            pos = gridworld.get_agent_position(self.selected_agent)

            # Position information with styling
            dpg.add_text("Current Position", color=(255, 200, 100), parent=tag)
            dpg.add_text(f"({pos[0]}, {pos[1]})", indent=10, parent=tag)

            dpg.add_spacer(height=15, parent=tag)
            dpg.add_separator(parent=tag)
            dpg.add_spacer(height=5, parent=tag)

            # Message history header
            dpg.add_text("Message History",
                         color=(200, 200, 200),
                         parent=tag)
            dpg.add_spacer(height=10, parent=tag)

            # Add messages with visual styling
            if hasattr(agent, 'messages') and agent.messages:
                for message in agent.messages:
                    self.format_message(message, tag)
            else:
                dpg.add_text("No messages", parent=tag)