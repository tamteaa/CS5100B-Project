import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld


class InfoPanelView:
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.agent_names = {agent_id: agent.name for agent_id, agent in env.agents.items()}
        self.selected_agent = list(env.agents.keys())[0] if env.agents else None
        self.last_message_count = 0
        self.last_position = None
        self.active_tab = "agents"

        self.backgrounds = {
            "system": (50, 50, 80),
            "assistant": (50, 80, 50),
            "user": (80, 50, 50)
        }

        self.themes = {}  # Store themes for reuse

        # Agent messages
        self.messages_container_tag = "agent_messages_container"
        self.messages_window_tag = "agent_messages_window"

        # Group messages
        self.group_messages_container_tag = "group_messages_container"
        self.group_messages_window_tag = "group_messages_window"

        self.position_tag = "agent_position"
        self.tab_bar_tag = "tab_bar"
        self.SCROLL_THRESHOLD = 20

        self.agent_colors = [
            (75, 0, 130),  # Indigo
            (0, 100, 0),  # Dark Green
            (139, 0, 0),  # Dark Red
            (0, 0, 139),  # Dark Blue
            (128, 0, 128),  # Purple
            (184, 134, 11),  # Dark Golden Rod
            (0, 139, 139),  # Dark Cyan
            (139, 69, 19),  # Saddle Brown
        ]

        # Create a mapping of agent names to color indices
        self.agent_color_map = {
            agent_name: idx % len(self.agent_colors)
            for idx, agent_name in enumerate(sorted(self.agent_names.values()))
        }

    def get_message_theme(self, role_or_agent):
        """Cache and reuse themes instead of creating new ones"""
        if role_or_agent not in self.themes:
            theme = dpg.add_theme()
            with dpg.theme_component(dpg.mvAll, parent=theme):
                if role_or_agent in self.backgrounds:
                    color = self.backgrounds[role_or_agent]
                else:
                    color_idx = self.agent_color_map.get(role_or_agent, 0)
                    color = self.agent_colors[color_idx]
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, color)
            self.themes[role_or_agent] = theme
        return self.themes[role_or_agent]

    def format_group_message(self, message, parent):
        # Similar changes as format_message, using cached themes
        from_agent = message.get("from", "unknown")
        content = message.get("message", "")

        dpg.add_spacer(height=5, parent=parent)

        with dpg.group(parent=parent):
            theme = self.get_message_theme(from_agent)
            text_length = len(content)
            estimated_lines = (text_length // 45) + 1
            base_height = max(85, 60 + (estimated_lines * 20))
            dynamic_height = int(base_height * 1.3)

            with dpg.child_window(width=380, height=dynamic_height, autosize_x=True, autosize_y=False):
                dpg.bind_item_theme(dpg.last_item(), theme)
                dpg.add_spacer(height=5)
                dpg.add_text(f"[{from_agent}]", color=(255, 255, 255))
                dpg.add_spacer(height=8)
                dpg.add_text(content, wrap=350, color=(255, 255, 255))
                dpg.add_spacer(height=5)

        dpg.add_spacer(height=5, parent=parent)

    def format_message(self, message, parent):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        dpg.add_spacer(height=5, parent=parent)

        with dpg.group(parent=parent):
            theme = self.get_message_theme(role)
            text_length = len(content)
            estimated_lines = (text_length // 45) + 1
            base_height = max(85, 300 + (estimated_lines * 20))
            dynamic_height = int(base_height * 1.1)

            with dpg.child_window(width=450, height=dynamic_height, autosize_x=True, autosize_y=False):
                dpg.bind_item_theme(dpg.last_item(), theme)
                dpg.add_spacer(height=5)
                dpg.add_text(f"[{role.upper()}]", color=(255, 255, 255))
                dpg.add_spacer(height=8)
                dpg.add_text(content, wrap=400, color=(255, 255, 255))
                dpg.add_spacer(height=5)

        dpg.add_spacer(height=5, parent=parent)

    def should_auto_scroll(self, window):
        current_scroll = dpg.get_y_scroll(window)
        max_scroll = dpg.get_y_scroll_max(window)
        return (max_scroll - current_scroll) <= self.SCROLL_THRESHOLD

    def on_agent_selected(self, sender, app_data):
        for agent_id, name in self.agent_names.items():
            if name == app_data:
                self.selected_agent = agent_id
                self.last_message_count = 0
                self.last_position = None
                break

    def on_tab_selected(self, sender, app_data):
        self.active_tab = app_data

    def setup_static_ui(self, tag: str):
        with dpg.group(parent=tag):
            with dpg.tab_bar(tag=self.tab_bar_tag):
                with dpg.tab(label="Agents", tag="agents_tab"):
                    # Agent selection
                    with dpg.group(horizontal=True):
                        dpg.add_text("Select Agent:")
                        dpg.add_combo(
                            items=list(self.agent_names.values()),
                            default_value=self.agent_names[self.selected_agent],
                            callback=self.on_agent_selected,
                            width=200
                        )

                    dpg.add_spacer(height=10)

                    # Position information
                    dpg.add_text("Current Position:", color=(255, 200, 100))
                    dpg.add_text("", tag=self.position_tag, indent=10)

                    dpg.add_spacer(height=10)

                    # Scrollable messages window
                    viewport_height = dpg.get_viewport_height()
                    remaining_height = viewport_height + 100
                    with dpg.child_window(tag=self.messages_window_tag, height=remaining_height):
                        dpg.add_group(tag=self.messages_container_tag)

                with dpg.tab(label="Group Messages", tag="messages_tab"):
                    with dpg.child_window(height=viewport_height - 50):
                        self.group_messages_container = dpg.add_group(tag="group_messages_container")

    def update_agent_info(self, gridworld: ComplexGridworld, tag: str):
        if not dpg.does_item_exist(self.messages_container_tag):
            dpg.delete_item(tag, children_only=True)
            self.setup_static_ui(tag)

        # Update agent info
        if self.selected_agent is not None:
            agent = self.agents[self.selected_agent]
            pos = gridworld.get_agent_position(self.selected_agent)

            if pos != self.last_position:
                dpg.set_value(self.position_tag, f"({pos[0]}, {pos[1]})")
                self.last_position = pos

            # Update agent messages
            current_message_count = len(agent.messages) if hasattr(agent, 'messages') else 0
            if current_message_count != self.last_message_count:
                should_scroll = self.should_auto_scroll(self.messages_window_tag)
                dpg.delete_item(self.messages_container_tag, children_only=True)

                if hasattr(agent, 'messages') and agent.messages:
                    for message in agent.messages:
                        self.format_message(message, self.messages_container_tag)
                    if should_scroll:
                        dpg.set_y_scroll(self.messages_window_tag, dpg.get_y_scroll_max(self.messages_window_tag))
                else:
                    dpg.add_text("No messages", parent=self.messages_container_tag)

                self.last_message_count = current_message_count

        # Update group messages
        dpg.delete_item(self.group_messages_container_tag, children_only=True)
        if 'group_messages' in gridworld.variables:
            for message in gridworld.variables['group_messages']:
                self.format_group_message(message, self.group_messages_container_tag)