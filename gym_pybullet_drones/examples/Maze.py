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

class MazeGenerator:
    def __init__(self, width=3, height=3, discretization=0.05, min_corridor_size=0.3, seed=100):
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

            direction = random.choice([(0, 1), (1, 0)])  # Horizontal or vertical
            length = random.randint(self.min_corridor_size, self.width // 2)

            if self._can_place_wall(start_x, start_y, direction, length):
                self._place_wall(start_x, start_y, direction, length)
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
        """Check if a cell has adjacent walls (not diagonal)."""
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

    def visualize(self):
        """Visualize the generated maze."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap="Greys")
        plt.title("Generated Maze with Walls")
        plt.show()

if __name__ == "__main__":
    maze = MazeGenerator()
    maze.generate()
    maze.visualize()