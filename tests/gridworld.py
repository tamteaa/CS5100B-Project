import unittest
from src.environments.custom_environments.gridworld_environment import GridworldEnvironment


class TestGridworldEnvironment(unittest.TestCase):

    def setUp(self):
        # Initialize a 5x5 grid environment with two agents
        self.env = GridworldEnvironment(
            grid_size=(5, 5),
            start_positions={0: (0, 0), 1: (4, 4)}
        )

    def test_initial_state(self):
        # Test that the initial state is set correctly
        initial_positions = self.env.reset()
        self.assertEqual(initial_positions, {0: (0, 0), 1: (4, 4)})
        self.assertEqual(self.env.get_agent_position(0), (0, 0))
        self.assertEqual(self.env.get_agent_position(1), (4, 4))

    def test_move_single_agent(self):
        # Test moving a single agent
        self.env.reset()
        new_pos = self.env.move(0, 'down')  # Move agent 0 down
        self.assertEqual(new_pos, (1, 0))
        self.assertEqual(self.env.get_agent_position(0), (1, 0))

        new_pos = self.env.move(1, 'left')  # Move agent 1 left
        self.assertEqual(new_pos, (4, 3))
        self.assertEqual(self.env.get_agent_position(1), (4, 3))

    def test_boundary_conditions(self):
        # Test that agents cannot move out of the grid boundaries
        self.env.reset()

        # Try to move agent 0 up (out of bounds)
        new_pos = self.env.move(0, 'up')
        self.assertEqual(new_pos, (0, 0))  # Position should not change

        # Try to move agent 1 down (out of bounds)
        new_pos = self.env.move(1, 'down')
        self.assertEqual(new_pos, (4, 4))  # Position should not change

        # Move agent 0 left (out of bounds)
        new_pos = self.env.move(0, 'left')
        self.assertEqual(new_pos, (0, 0))

        # Move agent 1 right (out of bounds)
        new_pos = self.env.move(1, 'right')
        self.assertEqual(new_pos, (4, 4))

    def test_multiple_agent_movements(self):
        # Test simultaneous movements of both agents
        self.env.reset()

        # Move agent 0 down, agent 1 up
        new_pos_0 = self.env.move(0, 'down')
        new_pos_1 = self.env.move(1, 'up')
        self.assertEqual(new_pos_0, (1, 0))
        self.assertEqual(new_pos_1, (3, 4))

        # Move agent 0 right, agent 1 left
        new_pos_0 = self.env.move(0, 'right')
        new_pos_1 = self.env.move(1, 'left')
        self.assertEqual(new_pos_0, (1, 1))
        self.assertEqual(new_pos_1, (3, 3))

    def test_agent_positions(self):
        # Test retrieval of all agent positions
        self.env.reset()
        agent_positions = self.env.get_all_agent_positions()
        self.assertEqual(agent_positions, {0: (0, 0), 1: (4, 4)})

        # Move agents and check updated positions
        self.env.move(0, 'down')
        self.env.move(1, 'left')
        agent_positions = self.env.get_all_agent_positions()
        self.assertEqual(agent_positions, {0: (1, 0), 1: (4, 3)})

    def test_render(self):
        # Test rendering of the grid
        self.env.reset()
        self.env.render()

    def test_invalid_agent_id(self):
        # Test moving an invalid agent ID
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.move(2, 'up')  # Agent ID 2 does not exist

    def test_grid_state(self):
        # Test grid state after moves
        self.env.reset()

        # Move agent 0 down and agent 1 up
        self.env.move(0, 'down')
        self.env.move(1, 'up')

        # Get grid state and verify agent positions
        grid_state = self.env.get_state()
        self.assertEqual(grid_state[1, 0], 1)  # Agent 0 at (1, 0)
        self.assertEqual(grid_state[3, 4], 2)  # Agent 1 at (3, 4)


if __name__ == '__main__':
    unittest.main()
