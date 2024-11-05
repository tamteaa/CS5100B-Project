import dearpygui.dearpygui as dpg
from src.environments.custom_environments.complex_gridworld_environment import ComplexGridworld


class GridWorldView:
    def __init__(self):
        self.cell_size = 30
        self.padding = 30  # Increased padding for better visibility

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

        # Draw grid
        for row in range(gridworld.grid_size[0]):
            for col in range(gridworld.grid_size[1]):
                # Flip the row coordinate for north-south orientation
                flipped_row = gridworld.grid_size[0] - 1 - row

                # Calculate cell position with offset
                top_left = (
                    offset_x + col * self.cell_size,
                    offset_y + flipped_row * self.cell_size
                )
                bottom_right = (
                    offset_x + (col + 1) * self.cell_size,
                    offset_y + (flipped_row + 1) * self.cell_size
                )

                square = gridworld[row][col]

                # Draw base square
                fill_color = (100, 100, 100) if square.obstacle else (255, 255, 255)
                self.draw_square(tag, top_left, bottom_right, fill_color)

                # Draw agent or items
                if square.agent_id is not None:
                    self.draw_square(tag, top_left, bottom_right, (255, 0, 0))
                elif square.has_items():
                    item = square.items[0]
                    item_center = (
                        (top_left[0] + bottom_right[0]) // 2,
                        (top_left[1] + bottom_right[1]) // 2
                    )

                    if item.shape == "circle":
                        radius = self.cell_size // 4
                        self.draw_circle(tag, item_center, radius, item.color)
                    elif item.shape == "triangle":
                        p1 = (item_center[0], item_center[1] - self.cell_size // 4)
                        p2 = (item_center[0] - self.cell_size // 4, item_center[1] + self.cell_size // 4)
                        p3 = (item_center[0] + self.cell_size // 4, item_center[1] + self.cell_size // 4)
                        self.draw_triangle(tag, p1, p2, p3, item.color)
                    elif item.shape == "square":
                        item_top_left = (
                            item_center[0] - self.cell_size // 4,
                            item_center[1] - self.cell_size // 4
                        )
                        item_bottom_right = (
                            item_center[0] + self.cell_size // 4,
                            item_center[1] + self.cell_size // 4
                        )
                        self.draw_item_square(tag, item_top_left, item_bottom_right, item.color)