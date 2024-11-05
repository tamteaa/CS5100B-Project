import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld
from src.gui.gridworld_view import GridWorldView
from src.gui.informational_panel import InfoPanelView
import threading
from time import sleep


class GUI:
    def __init__(self, gridworld: ComplexGridworld, agents):
        self.gridworld = gridworld
        self.is_running = True
        self.view = GridWorldView()
        self.info_panel = InfoPanelView(agents)
        self.initial_width = 800
        self.initial_height = 800

        # Initialize DearPyGUI context
        dpg.create_context()

        # Create the main window
        with dpg.window(label="Simulation", tag="primary_window", no_close=True):
            # Create horizontal layout with dynamic width
            with dpg.group(horizontal=True, tag="main_group"):
                # Left panel for grid (50% width) - no scrollbars
                with dpg.child_window(tag="left_panel",
                                    width=self.initial_width // 2,
                                    height=self.initial_height,
                                    horizontal_scrollbar=False,
                                    no_scrollbar=True):
                    with dpg.drawlist(width=self.initial_width // 2,
                                    height=self.initial_height,
                                    tag="grid_canvas"):
                        pass

                # Right panel for info (50% width) - only vertical scrollbar
                with dpg.child_window(tag="right_panel",
                                    width=self.initial_width // 2,
                                    height=self.initial_height,
                                    horizontal_scrollbar=False):
                    with dpg.group(tag="info_panel"):
                        pass

        # Handler for window resize
        def resize_callback():
            viewport_width = dpg.get_viewport_client_width()
            viewport_height = dpg.get_viewport_client_height()

            # Update window size
            dpg.configure_item("primary_window", width=viewport_width, height=viewport_height)

            # Update panels (each 50% of width)
            panel_width = viewport_width // 2
            dpg.configure_item("left_panel", width=panel_width, height=viewport_height)
            dpg.configure_item("right_panel", width=panel_width, height=viewport_height)
            dpg.configure_item("grid_canvas", width=panel_width, height=viewport_height)

        dpg.set_viewport_resize_callback(resize_callback)

    def _run_gui(self):
        # Configure and show viewport
        dpg.create_viewport(title="GridWorld Simulation",
                          width=self.initial_width,
                          height=self.initial_height,
                          resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

        # Main render loop
        while self.is_running:
            self.view.draw_gridworld(self.gridworld, "grid_canvas")
            self.info_panel.update_agent_info(self.gridworld, "info_panel")
            dpg.render_dearpygui_frame()
            sleep(0.01)

        dpg.destroy_context()

    def start(self):
        self.gui_thread = threading.Thread(target=self._run_gui)
        self.gui_thread.start()

    def close(self):
        self.is_running = False
        if hasattr(self, 'gui_thread'):
            self.gui_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()