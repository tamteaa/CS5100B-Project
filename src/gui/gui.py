import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld
from src.gui.gridworld_view import GridWorldView
from src.gui.informational_panel import InfoPanelView
import platform
import time
import threading


class GUI:
    def __init__(self, env: ComplexGridworld, agents):
        self.env = env
        self.agents = agents
        self.is_running = True
        self.view = GridWorldView()
        self.info_panel = InfoPanelView(agents)
        self.initial_width = 800
        self.initial_height = 800
        self.is_mac = platform.system() == "Darwin"

        # Initialize DearPyGUI context
        dpg.create_context()
        self._setup_gui()

    def _setup_gui(self):
        with dpg.window(label="Simulation", tag="primary_window", no_close=True):
            with dpg.group(horizontal=True):
                with dpg.group(tag="left_group"):
                    dpg.add_drawlist(width=self.initial_width // 2,
                                     height=self.initial_height,
                                     tag="grid_canvas")
                with dpg.group(tag="right_group"):
                    dpg.add_group(tag="info_panel")

        dpg.create_viewport(title="GridWorld Simulation",
                            width=self.initial_width,
                            height=self.initial_height,
                            resizable=True)

        dpg.set_viewport_resize_callback(self._resize_callback)
        dpg.setup_dearpygui()

    def _resize_callback(self):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        dpg.configure_item("primary_window", width=viewport_width, height=viewport_height)
        panel_width = viewport_width // 2
        dpg.configure_item("grid_canvas", width=panel_width, height=viewport_height)

    def _render_frame(self):
        """Render a single frame"""
        if dpg.does_item_exist("grid_canvas"):
            dpg.delete_item("grid_canvas", children_only=True)
            self.view.draw_gridworld(self.env, "grid_canvas")

        if dpg.does_item_exist("info_panel"):
            dpg.delete_item("info_panel", children_only=True)
            self.info_panel.update_agent_info(self.env, "info_panel")

    def _start_dearpygui(self):
        """Start DearPyGUI main loop"""
        while dpg.is_dearpygui_running() and self.is_running:
            self._render_frame()
            dpg.render_dearpygui_frame()
            time.sleep(0.033)  # ~30 FPS

    def start(self):
        """Start the GUI"""
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        self._start_dearpygui()

    def close(self):
        """Close the GUI"""
        self.is_running = False
        dpg.destroy_context()

    def run(self, sim_func, target_function):
        simulation_thread = threading.Thread(
            target=sim_func, args=(self.env, target_function, self)
        )
        simulation_thread.start()

        # Start the GUI main loop (blocks the main thread)
        self.start()

        # Wait for the simulation thread to finish
        simulation_thread.join()

        # Close the GUI
        self.close()