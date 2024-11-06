import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld


class InfoPanelView:
    def __init__(self, agents):
        self.agents = agents
        self.agent_names = {agent_id: agent.name for agent_id, agent in agents.items()}
        self.selected_agent = list(agents.keys())[0] if agents else None
        self.last_message_count = 0  # Track number of messages to detect changes
        self.last_position = None  # Track position to detect changes

        # Define colors for different roles
        self.backgrounds = {
            "system": (50, 50, 80),  # Dark blue
            "assistant": (50, 80, 50),  # Dark green
            "user": (80, 50, 50)  # Dark red
        }

        # Tag for the messages container
        self.messages_container_tag = "messages_container"
        self.position_tag = "agent_position"

        # Scroll threshold - consider user "at bottom" if within this many pixels of bottom
        self.SCROLL_THRESHOLD = 20

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

            # Calculate dynamic height based on content with 30% increase
            text_length = len(content)
            # Assume roughly 45 characters per line with wrap at 350
            estimated_lines = (text_length // 45) + 1
            base_height = max(85, 60 + (estimated_lines * 20))  # base height + lines * line height
            dynamic_height = int(base_height * 1.3)  # Increase height by 30%

            # Create child window with colored background
            with dpg.child_window(width=380, height=dynamic_height, parent=parent, autosize_x=True, autosize_y=False):
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

    def should_auto_scroll(self, window):
        """Determine if we should auto-scroll based on current scroll position"""
        current_scroll = dpg.get_y_scroll(window)
        max_scroll = dpg.get_y_scroll_max(window)

        # If we're within SCROLL_THRESHOLD pixels of the bottom, consider it "at bottom"
        return (max_scroll - current_scroll) <= self.SCROLL_THRESHOLD

    def on_agent_selected(self, sender, app_data):
        """Handle agent selection from dropdown"""
        for agent_id, name in self.agent_names.items():
            if name == app_data:
                self.selected_agent = agent_id
                self.last_message_count = 0  # Reset message count to force refresh
                self.last_position = None  # Reset position to force refresh
                break

    def setup_static_ui(self, tag: str):
        """Setup the static UI elements that don't need frequent updates"""
        dpg.add_text("Select Agent:", parent=tag)
        dpg.add_combo(
            items=list(self.agent_names.values()),
            default_value=self.agent_names[self.selected_agent],
            callback=self.on_agent_selected,
            width=200,
            parent=tag
        )

        dpg.add_spacer(height=15, parent=tag)
        dpg.add_separator(parent=tag)
        dpg.add_spacer(height=15, parent=tag)

        # Position information with styling
        dpg.add_text("Current Position", color=(255, 200, 100), parent=tag)
        dpg.add_text("", tag=self.position_tag, indent=10, parent=tag)

        dpg.add_spacer(height=15, parent=tag)
        dpg.add_separator(parent=tag)
        dpg.add_spacer(height=5, parent=tag)

        # Message history header
        dpg.add_text("Message History", color=(200, 200, 200), parent=tag)
        dpg.add_spacer(height=10, parent=tag)

        # Create container for messages
        dpg.add_group(tag=self.messages_container_tag, parent=tag)

    def update_agent_info(self, gridworld: ComplexGridworld, tag: str):
        # Setup static UI if it doesn't exist
        if not dpg.does_item_exist(self.messages_container_tag):
            dpg.delete_item(tag, children_only=True)
            self.setup_static_ui(tag)

        if self.selected_agent is not None:
            agent = self.agents[self.selected_agent]
            pos = gridworld.get_agent_position(self.selected_agent)

            # Update position if changed
            if pos != self.last_position:
                dpg.set_value(self.position_tag, f"({pos[0]}, {pos[1]})")
                self.last_position = pos

            # Update messages if changed
            current_message_count = len(agent.messages) if hasattr(agent, 'messages') else 0
            if current_message_count != self.last_message_count:
                # Get scroll info before updating
                parent_window = dpg.get_item_parent(tag)
                should_scroll = self.should_auto_scroll(parent_window)

                # Update messages
                dpg.delete_item(self.messages_container_tag, children_only=True)
                if hasattr(agent, 'messages') and agent.messages:
                    for message in agent.messages:
                        self.format_message(message, self.messages_container_tag)

                    if should_scroll:
                        dpg.set_y_scroll(parent_window, dpg.get_y_scroll_max(parent_window))
                else:
                    dpg.add_text("No messages", parent=self.messages_container_tag)

                self.last_message_count = current_message_count