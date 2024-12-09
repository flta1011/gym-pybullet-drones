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
from scipy.ndimage import label

class Maze:
    def __init__(self, x_width=3, y_height=3, discretization=0.05, seed=1, min_corridor_size=0.3):
        self.width = x_width
        self.length = y_height
        self.discretization = discretization
        self.seed = seed
        self.min_corridor_size = min_corridor_size

        self.rows = int(y_height / discretization)
        self.cols = int(x_width / discretization)
        self.maze = np.zeros((self.rows, self.cols), dtype=int)  # Start with all corridors
        self.generate_maze()

    def generate_maze(self):
        np.random.seed(self.seed)  # Ensure reproducibility with random seed
        
        # Start with all corridors
        self.maze = np.zeros((self.rows, self.cols), dtype=int)

        # Set the outer walls to 1 (walls)
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Define the starting point
        start_row, start_col = 6, 6
        self.maze[start_row, start_col] = 0  # Mark the starting cell as a corridor

        # Use DFS to carve out the maze
        self._dfs(start_row, start_col)

        # Fill closed loops
        self._fill_closed_loops()

    def _dfs(self, row, col):
        # Define the possible directions (up, down, left, right)
        directions = [(0, 6), (0, -6), (6, 0), (-6, 0)]
        np.random.shuffle(directions)  # Randomize the order of directions

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check if the new position is within the maze boundaries and not adjacent to outer walls
            if 1 < new_row < self.rows - 2 and 1 < new_col < self.cols - 2:
                if self.maze[new_row, new_col] == 0:  # If the new cell is a corridor
                    # Carve a path between the current cell and the new cell
                    self.maze[new_row, new_col] = 1
                    for i in range(1, 6):
                        self.maze[row + (dr // 6) * i, col + (dc // 6) * i] = 1
                    # Recursively apply DFS from the new cell
                    self._dfs(new_row, new_col)

    def _fill_closed_loops(self):
        # Label connected components
        labeled_maze, num_features = label(self.maze == 0)
        
        # Find the largest connected component (assumed to be the main path)
        largest_component = np.argmax(np.bincount(labeled_maze.flat)[1:]) + 1
        
        # Fill all other components with walls
        self.maze[labeled_maze != largest_component] = 1

    def __str__(self):
        return str(self.maze)

    def visualize(self):
        plt.imshow(self.maze, cmap='binary')
        plt.title('Generated Maze')
        plt.show()

if __name__ == "__main__":
    seed = int(input("Enter a seed value: "))
    maze = Maze(seed=seed)
    print(maze)
    maze.visualize()