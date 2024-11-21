import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld


class GridWorldView:
    def __init__(self):
        self.cell_size = 40
        self.padding = 10

    def calculate_cell_size(self, canvas_width, canvas_height, grid_size):
        """Calculate the optimal cell size based on canvas dimensions and grid size."""
        available_width = canvas_width - 2 * self.padding
        available_height = canvas_height - 2 * self.padding

        # Get the smallest dimension to ensure square cells that fit in the canvas
        return min(
            available_width // grid_size,
            available_height // grid_size
        )

    def draw_square(self, parent, top_left, bottom_right, color=(255, 255, 255), thickness=1):
        dpg.draw_rectangle(
            top_left, bottom_right, color=(200, 200, 200), thickness=thickness, fill=color, parent=parent
        )

    def draw_circle(self, parent, center, radius, color, thickness=1):
        dpg.draw_circle(center=center, radius=radius, color=(0, 0, 0), fill=color, parent=parent)

    def draw_triangle(self, parent, p1, p2, p3, color, thickness=1):
        dpg.draw_triangle(p1, p2, p3, color=(0, 0, 0), fill=color, parent=parent)

    def draw_item_square(self, parent, top_left, bottom_right, color, thickness=1):
        dpg.draw_rectangle(top_left, bottom_right, color=(0, 0, 0), fill=color, parent=parent)

    def draw_text(self, parent, position, text, color=(0, 0, 0), size=20):
        dpg.draw_text(position, text=text, color=color, size=size, parent=parent)

    def draw_gridworld(self, gridworld: ComplexGridworld, tag: str):
        # Clear existing drawings
        dpg.delete_item(tag, children_only=True)

        # Get drawable area dimensions
        canvas_width = dpg.get_item_width(tag)
        canvas_height = dpg.get_item_height(tag)

        # Calculate optimal cell size
        self.cell_size = self.calculate_cell_size(
            canvas_width,
            canvas_height,
            max(gridworld.grid_size[0], gridworld.grid_size[1])
        )

        # Calculate total grid size
        grid_width = self.cell_size * gridworld.grid_size[1]
        grid_height = self.cell_size * gridworld.grid_size[0]

        # Calculate offset to center the grid
        offset_x = (canvas_width - grid_width) // 2
        offset_y = (canvas_height - grid_height) // 2

        for row in range(gridworld.grid_size[0]):
            for col in range(gridworld.grid_size[1]):
                square = gridworld[row][col]

                flipped_row = gridworld.grid_size[0] - 1 - row
                top_left = (offset_x + col * self.cell_size, offset_y + flipped_row * self.cell_size)
                bottom_right = (offset_x + (col + 1) * self.cell_size, offset_y + (flipped_row + 1) * self.cell_size)

                fill_color = (100, 100, 100) if square.obstacle else (255, 255, 255)
                self.draw_square(tag, top_left, bottom_right, fill_color)

                if square.agents:
                    self.draw_square(tag, top_left, bottom_right, (255, 200, 200))

                    # Draw all agent names
                    num_agents = len(square.agents)
                    for i, agent in enumerate(square.agents):
                        text_size = int(self.cell_size * 0.3 / num_agents)  # Scale text size based on number of agents
                        text_width = len(agent.name) * text_size * 0.6
                        text_x = (top_left[0] + bottom_right[0]) // 2 - text_width // 2
                        text_y = top_left[1] + (i + 1) * self.cell_size // (num_agents + 1) - text_size // 2
                        text_position = (text_x, text_y)
                        self.draw_text(tag, text_position, agent.name, color=(0, 0, 0), size=text_size)

                num_items = len(square.items)
                for i, item in enumerate(square.items):
                    # Calculate offset based on number of items
                    x_offset = (i - (num_items - 1) / 2) * (self.cell_size // (num_items + 1))
                    item_center = (
                        ((top_left[0] + bottom_right[0]) // 2) + x_offset,
                        (top_left[1] + bottom_right[1]) // 2
                    )

                    if item.shape == "circle":
                        radius = self.cell_size // (4 + num_items // 2)  # Smaller radius with more items
                        self.draw_circle(tag, item_center, radius, item.color)
                    elif item.shape == "triangle":
                        size = self.cell_size // (4 + num_items // 2)
                        p1 = (item_center[0], item_center[1] - size)
                        p2 = (item_center[0] - size, item_center[1] + size)
                        p3 = (item_center[0] + size, item_center[1] + size)
                        self.draw_triangle(tag, p1, p2, p3, item.color)
                    elif item.shape == "square":
                        size = self.cell_size // (4 + num_items // 2)
                        item_top_left = (item_center[0] - size, item_center[1] - size)
                        item_bottom_right = (item_center[0] + size, item_center[1] + size)
                        self.draw_item_square(tag, item_top_left, item_bottom_right, item.color)

