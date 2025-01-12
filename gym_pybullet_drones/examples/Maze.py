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
from xml.dom import minidom

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

    def generate_urdf_from_maze(self, maze_height, filename="gym_pybullet_drones/assets/maze/maze_single_wall_link.urdf"):
        """Generate a URDF file from the maze."""

        def add_wall_link(root, name, size, xyz, rpy="0 0 0", mass=1):
            """Add a link to the URDF file."""
            link = ET.SubElement(root, "link", name=name)
            visual = ET.SubElement(link, "visual")
            geometry = ET.SubElement(visual, "geometry")
            box = ET.SubElement(geometry, "box", size=size)
            origin = ET.SubElement(visual, "origin", xyz=xyz, rpy=rpy)
            inertial = ET.SubElement(link, "inertial")
            mass_element = ET.SubElement(inertial, "mass", value=str(mass))
            inertia = ET.SubElement(inertial, "inertia", ixx="1", iyy="1", izz="1")
            collision = ET.SubElement(link, "collision")
            collision_geometry = ET.SubElement(collision, "geometry")
            collision_box = ET.SubElement(collision_geometry, "box", size=size)
            collision_origin = ET.SubElement(collision, "origin", xyz=xyz, rpy=rpy)

        def add_joint(root, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
            """Add a joint to the URDF file."""
            joint = ET.SubElement(root, "joint", name=name, type="fixed")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)

        root = ET.Element("robot", name="maze")

        # Add the base link
        base_link = ET.SubElement(root, "link", name="base_link")

        # Add the floor
        floor_size = f"{self.width * self.discretization} {self.height * self.discretization} {self.discretization}"
        floor_xyz = f"{self.width * self.discretization / 2} {self.height * self.discretization / 2} -{self.discretization / 2}"
        add_wall_link(root, "floor", floor_size, floor_xyz)
        add_joint(root, "floor_joint", "base_link", "floor")

        # Add the walls as a single link
        wall_link = ET.SubElement(root, "link", name="walls")
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 1:
                    wall_size = f"{self.discretization} {self.discretization} {maze_height}"
                    wall_xyz = f"{x * self.discretization + self.discretization / 2} {y * self.discretization + self.discretization / 2} {maze_height / 2}"
                    
                    # Add visual geometry
                    visual = ET.SubElement(wall_link, "visual")
                    geometry = ET.SubElement(visual, "geometry")
                    box = ET.SubElement(geometry, "box", size=wall_size)
                    origin = ET.SubElement(visual, "origin", xyz=wall_xyz, rpy="0 0 0")
                    
                    # Add collision geometry
                    collision = ET.SubElement(wall_link, "collision")
                    collision_geometry = ET.SubElement(collision, "geometry")
                    collision_box = ET.SubElement(collision_geometry, "box", size=wall_size)
                    collision_origin = ET.SubElement(collision, "origin", xyz=wall_xyz, rpy="0 0 0")

        # Add a joint for the walls
        add_joint(root, "walls_joint", "base_link", "walls", xyz="0 0 0")

        # Write the URDF file
        tree = ET.ElementTree(root)
        xml_str = ET.tostring(root, encoding="unicode")
        pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
        with open(filename, "w") as f:
            f.write(pretty_xml_str)


if __name__ == "__main__":
    rs = 42 # Random seed
    maze = MazeGenerator(seed=rs)
    maze.generate()
    #maze.visualize()
    #maze.visualize_range_of_mazes(start_seed=1, stop_seed=104)

    Maze_Name = f"gym_pybullet_drones/assets/maze/maze_rs_{rs}.urdf"
    maze.generate_urdf_from_maze(maze_height=1, filename=Maze_Name)



    # best usable mazes -> 1, 6, 7, 8, 12, 42, 