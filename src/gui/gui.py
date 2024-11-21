import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld
from src.gui.gridworld_view import GridWorldView
from src.gui.informational_panel import InfoPanelView
import platform
import time
import threading


class GUI:
    def __init__(self, env: ComplexGridworld):
        self.env = env
        self.agents = env.agents
        self.is_running = True
        self.view = GridWorldView()
        self.info_panel = InfoPanelView(env)
        self.initial_width = 800
        self.initial_height = 800
        self.is_mac = platform.system() == "Darwin"

        # Initialize DearPyGUI context
        dpg.create_context()
        self._setup_gui()

        # Flag to track if info panel has been initialized
        self.info_panel_initialized = False

    def _setup_gui(self):
        with dpg.window(label="Simulation", tag="primary_window", no_close=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                with dpg.group(tag="left_group"):
                    dpg.add_drawlist(width=self.initial_width // 2,
                                     height=self.initial_height,
                                     tag="grid_canvas")
                with dpg.group(tag="right_group"):
                    # Create a scrollable container for the info panel
                    with dpg.child_window(tag="info_panel_container", width=self.initial_width // 2,
                                          height=self.initial_height, no_scrollbar=True):
                        dpg.add_group(tag="info_panel")

        dpg.create_viewport(title="GridWorld Simulation",
                            width=self.initial_width,
                            height=self.initial_height,
                            resizable=True,
                            )

        dpg.set_viewport_resize_callback(self._resize_callback)
        dpg.setup_dearpygui()

    def _resize_callback(self):
        viewport_width = dpg.get_viewport_client_width() * 0.98
        viewport_height = dpg.get_viewport_client_height() * 0.98
        dpg.configure_item("primary_window", width=viewport_width, height=viewport_height)
        panel_width = viewport_width // 2
        dpg.configure_item("grid_canvas", width=panel_width, height=viewport_height)
        dpg.configure_item("info_panel_container", width=panel_width, height=viewport_height)

    def _render_frame(self):
        """Render a single frame"""
        # Update grid canvas
        if dpg.does_item_exist("grid_canvas"):
            dpg.delete_item("grid_canvas", children_only=True)
            self.view.draw_gridworld(self.env, "grid_canvas")

        # Initialize info panel once or update its contents
        if dpg.does_item_exist("info_panel"):
            if not self.info_panel_initialized:
                self.info_panel.setup_static_ui("info_panel")
                self.info_panel_initialized = True
            self.info_panel.update_agent_info(self.env, "info_panel")

    def _start_dearpygui(self):
        """Start DearPyGUI main loop"""
        while dpg.is_dearpygui_running() and self.is_running:
            self._render_frame()
            dpg.render_dearpygui_frame()
            time.sleep(0.01)  # ~30 FPS

            if self.env.terminated:
                break

    def start(self):
        """Start the GUI"""
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        self._start_dearpygui()

    def close(self):
        """Close the GUI"""
        self.is_running = False
        dpg.destroy_context()

    def run(self, sim_func):
        simulation_thread = threading.Thread(target=sim_func, args=(self.env,))
        simulation_thread.start()

        # Start the GUI main loop (blocks the main thread)
        self.start()

        # Wait for the simulation thread to finish
        simulation_thread.join()

        # Close the GUI
        self.close()

