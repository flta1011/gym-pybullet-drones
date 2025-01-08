"""

Automatisierte Generierung von Labyrinthen mit Tiefensuche (DFS) und Auffüllen geschlossener Schleifen


Es können mit Hilfe des Random Seed-Werts beliebig viele, reproduzierbare, "zufällige" Labyrinthe generiert werden.
Die Größe des Labyrinths, die Diskretisierung, die Mindestgröße der Korridore und der Seed-Wert können angepasst werden.


- Die Größe des Labyrinths kann durch die Parameter x_width und y_height festgelegt werden.
- Die Diskretisierung des Labyrinths kann durch den Parameter discretization festgelegt werden.
- Die Mindestgröße der Korridore kann durch den Parameter min_corridor_size festgelegt werden.
- Der Seed-Wert kann durch den Parameter seed festgelegt werden.

Das Labyrinth wird als Matrix dargestellt, wobei 0 für Korridore und 1 für Wände steht.
Eine Visualisierung über die Methode visualize() ist ebenfalls möglich.


"""


import numpy as np
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET

class MazeGenerator:
    def __init__(self, width=3, height=3, discretization=0.05, min_corridor_size=0.3, seed=30):
        self.width = int(width / discretization)  # Maze width in grid units
        self.height = int(height / discretization)  # Maze height in grid units
        self.discretization = discretization  # Grid resolution
        self.min_corridor_size = int(min_corridor_size / discretization)  # Minimum corridor size in grid units
        self.grid = np.zeros((self.height, self.width), dtype=int)  # Maze grid initialized as empty
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate(self):
        """Generate maze with walls following the placement rules."""
        self._place_outer_walls()
        self._generate_internal_walls()

    def _place_outer_walls(self):
        """Place the boundary walls of the maze."""
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _generate_internal_walls(self):
        """Iteratively place internal walls while checking for placement rules."""
        attempts = 0
        max_attempts = 500  # Prevent infinite loops
        while attempts < max_attempts:
            start_x, start_y = self._find_valid_wall_start()
            if start_x is None:
                break  # No valid start position found

            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])  # Horizontal or vertical
            length = random.randint(self.min_corridor_size, self.width // 2)

            if self._can_place_wall(start_x, start_y, direction, length):
                self._place_wall(start_x, start_y, direction, length)
                print(start_x, start_y)
            attempts += 1

    def _find_valid_wall_start(self):
        """Find a valid start position for a new wall."""
        for _ in range(100):  # Limit the search attempts
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[y, x] == 0 and self._is_connected_to_wall(x, y):
                return x, y
        return None, None

    def _is_connected_to_wall(self, x, y):
        """Ensure the starting point is connected to an existing wall."""
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny, nx] == 1:
                    return True
        return False

    def _can_place_wall(self, x, y, direction, length):
        """Check if a wall can be placed starting at (x, y) in the given direction."""
        dx, dy = direction
        connected = False
        for i in range(length):
            nx, ny = x + i * dx, y + i * dy
            if not (0 < nx < self.width - 1 and 0 < ny < self.height - 1):
                return False  # Out of bounds
            if self.grid[ny, nx] == 1:
                return False  # Wall already exists
            # Ensure at least one point is connected to an existing wall
            if self._is_connected_to_wall(nx, ny):
                connected = True
            # Check for touching walls (excluding start and end points)
            if i > 0 and i < length - 1:
                if self._has_adjacent_wall(nx, ny):
                    return False
            # Ensure minimum distance orthogonal to the wall
            if not self._has_min_orthogonal_distance(nx, ny, direction):
                return False
            # Ensure minimum distance colinear with the wall
            if not self._has_min_colinear_distance(nx, ny, direction):
                return False
        return connected

    def _has_min_orthogonal_distance(self, x, y, direction):
        """Check if there is a minimum orthogonal distance to other walls."""
        orthogonal_directions = [(1, 0), (-1, 0)] if direction == (0, 1) else [(0, 1), (0, -1)]
        
        # Check orthogonal distance for the current position
        for ox, oy in orthogonal_directions:
            for dist in range(1, self.min_corridor_size + 1):
                nx, ny = x + ox * dist, y + oy * dist
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] == 1:
                        return False

        # Check orthogonal distance for the length of the corridor + min_corridor_size
        for i in range(self.min_corridor_size + 1):
            for ox, oy in orthogonal_directions:
                for dist in range(1, self.min_corridor_size + 1):
                    nx, ny = x + direction[0] * i + ox * dist, y + direction[1] * i + oy * dist
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny, nx] == 1:
                            return False

        return True
    
    def _has_min_colinear_distance(self, x, y, direction):
        """Check if there is a minimum colinear distance to other walls."""
        dx, dy = direction
        for dist in range(1, self.min_corridor_size + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny, nx] == 1:
                    return False
        return True

    def _has_adjacent_wall(self, x, y):
        """Check if a cell has adjacent walls (not diagonal).""" # adjacent = benachbart
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if self.grid[ny, nx] == 1:
                return True
        return False

    def _place_wall(self, x, y, direction, length):
        """Place a wall starting at (x, y) in the given direction."""
        dx, dy = direction
        for i in range(length):
            nx, ny = x + i * dx, y + i * dy
            self.grid[ny, nx] = 1

    # def visualize_old(self, ax=None):
    #     """Visualize the generated maze."""
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(self.grid, cmap="Greys")
    #     plt.title("Generated Maze with Walls")
    #     plt.show()

    def visualize(self, ax=None):
        """Visualize the maze using matplotlib."""
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(self.grid, cmap='binary')
            ax.set_title(f'Maze with seed {self.seed}')
            ax.axis('off')
            plt.show()
        else:
            ax.imshow(self.grid, cmap='binary')
            ax.set_title(f'Maze with seed {self.seed}')
            ax.axis('off')
    
    def visualize_range_of_mazes(self, start_seed=1, stop_seed=9):
        """Visualize a range of mazes with different seeds."""
        n = (stop_seed - start_seed) + 1 # Number of mazes to generate; +1 because range is inclusive
        max_per_row = 8
        rows = (n + max_per_row -1) // max_per_row
        cols = min(n, max_per_row)
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows)) # Create n subplots

        # Flatten axes array if it is 2D
        if rows > 1:
            axes = axes.flatten()

        for i, seed in enumerate(range(start_seed, stop_seed+1)):
            self.seed = seed
            self.generate()
            # ax = axes[i] if n > 1 else axes # Handle the case when n is 1
            ax = axes[i]
            self.visualize(ax=ax)
        plt.show()

    # def generate_urdf_from_maze(self, filename="maze_urdf_test/maze.urdf", height=0.5):
    #     """Generate a URDF file from the maze."""
    #     with open(filename, "w") as f:
    #         f.write('<?xml version="1.0" ?>\n')
    #         f.write('<robot name="maze">\n')
    #         for y in range(self.height):
    #             for x in range(self.width):
    #                 if self.grid[y, x] == 1:
    #                     f.write(f'<link name="wall_{y}_{x}">\n')
    #                     f.write('  <visual>\n')
    #                     f.write('    <geometry>\n')
    #                     f.write('      <box size="0.05 0.05 0.05"/>\n')
    #                     f.write('    </geometry>\n')
    #                     f.write('  </visual>\n')
    #                     f.write(f'  <collision>\n')
    #                     f.write('    <geometry>\n')
    #                     f.write('      <box size="0.05 0.05 0.05"/>\n')
    #                     f.write('    </geometry>\n')
    #                     f.write('  </collision>\n')
    #                     f.write('</link>\n')
    #                     f.write(f'<joint name="joint_{y}_{x}" type="fixed">\n')
    #                     f.write('  <parent link="world"/>\n')
    #                     f.write(f'  <child link="wall_{y}_{x}"/>\n')
    #                     f.write('  <origin xyz="0 0 0"/>\n')
    #                     f.write('</joint>\n')
    #         f.write('</robot>\n')

    def generate_urdf_from_maze(self, filename="gym_pybullet_drones/examples/maze_urdf_test/", height=1.0):
        """Generate a URDF file from the maze."""
        filename = filename + f"maze_seed_{self.seed}.urdf"
        root = ET.Element("robot", name="maze")

        # Add the floor
        floor_link = ET.SubElement(root, "link", name="floor")
        floor_visual = ET.SubElement(floor_link, "visual")
        floor_geometry = ET.SubElement(floor_visual, "geometry")
        floor_box = ET.SubElement(floor_geometry, "box", size=f"{self.width * self.discretization} {self.height * self.discretization} 0.1")
        floor_origin = ET.SubElement(floor_visual, "origin", xyz=f"{self.width * self.discretization / 2} {self.height * self.discretization / 2} -0.05")

        # Add the walls
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 1:  # Assuming 1 represents a wall
                    link = ET.SubElement(root, "link", name=f"wall_{x}_{y}")
                    visual = ET.SubElement(link, "visual")
                    geometry = ET.SubElement(visual, "geometry")
                    box = ET.SubElement(geometry, "box", size=f"{self.discretization} {self.discretization} {height}")
                    origin = ET.SubElement(visual, "origin", xyz=f"{x * self.discretization} {y * self.discretization} {height / 2}")

        tree = ET.ElementTree(root)
        tree.write(filename)

if __name__ == "__main__":
    maze = MazeGenerator(seed=25)
    maze.generate()
    maze.visualize()
    #maze.visualize_range_of_mazes(start_seed=1, stop_seed=48)
    maze.generate_urdf_from_maze()