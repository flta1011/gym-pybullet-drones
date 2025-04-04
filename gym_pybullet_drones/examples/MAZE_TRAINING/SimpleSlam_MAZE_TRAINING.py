import os

import matplotlib.pyplot as plt
import numpy as np
import time


class SimpleSlam:
    def __init__(self, map_size=4, resolution=0.05, init_position=None):
        """Erstellt eine leere Occupancy-Grid Map. 
        Args: 
            map_size (float): Seitengröße der Map in Metern (z. B. 8 m). 
            resolution (float): Seitengröße einer Zelle, sodass grid_size ~60 ergibt.
            init_position (tuple): (x, y)-Startposition der Drohne in Metern.
        """
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)

        # Initialisiere die Map-Werte:
        self.unbekannt_value = 0
        self.frei_value = 200
        self.wand_value = 50
        self.besucht_value = 125
        self.actual_Position_value = 255
        self.init = init_position
        self.resolution = resolution

        # Berechne die Map-Offsets basierend auf der init_position
        if init_position is not None:
            init_x, init_y = init_position
            self.offset_x = int(init_x / resolution)
            self.offset_y = int(init_y / resolution)
            if not (0 <= self.offset_x < self.grid_size and 0 <= self.offset_y < self.grid_size):
                raise ValueError("init_position muss innerhalb der Map-Grenzen liegen.")
        else:
            self.offset_x = self.grid_size // 2
            self.offset_y = self.grid_size // 2

        # Initialisiere die Map:
        self.occupancy_grid = self.unbekannt_value * np.ones((self.grid_size, self.grid_size, 1))
        self.center = self.grid_size // 2
        self.path = []  # speichert besuchte Zellen
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []
        self.counter_free_space = 0

    def reset(self):
        """Reset the SLAM map to its initial state."""
        self.occupancy_grid = self.unbekannt_value * np.ones((self.grid_size, self.grid_size, 1))
        self.path = []
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []

    def world_to_grid(self, x, y):
        init_x, init_y = self.init
        # self.offset_x = int(init_x / self.resolution)
        # self.offset_y = int(init_y / self.resolution)
        # wenn init x im dritten quadranten liegt
        if init_x <= 1.5 and init_y <= 1.5:
            grid_x = int(self.center + (x / self.resolution) - self.center)
            grid_y = int(self.center + (y / self.resolution) - self.center)
        # wenn init x im vierten quadranten liegt
        elif init_x <= 1.5 and init_y > 1.5:
            grid_x = int(self.center + (x / self.resolution) - self.center)
            grid_y = int(self.center - (y / self.resolution) + self.center)
        # wenn init x im zweiten quadranten liegt
        elif init_x > 1.5 and init_y <= 1.5:
            grid_x = int(self.center - (x / self.resolution) + self.center)
            grid_y = int(self.center + (y / self.resolution) - self.center)
        # wenn init x im ersten quadranten liegt
        elif init_x > 1.5 and init_y > 1.5:
            grid_x = int(self.center - (x / self.resolution) + self.center)
            grid_y = int(self.center - (y / self.resolution) + self.center)
        return grid_x, grid_y

    def is_within_bounds(self, grid_x, grid_y):
        """Überprüft, ob die Drohne innerhalb der Map-Grenzen bleibt."""
        return 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size

    def update(self, drone_pos, drone_yaw, raycast_results):

        #NOTE - Werte anpassen wie ich welches Szenario darstellen möchte TBD 24.03.2025
        #NOTE - 0.2: unbekannt, 0.9: frei, 0.0: Wand, 0.5: besucht (Sensor oben frei), 0.7 = aktuelle Position
      

        """
        Aktualisiert die Map anhand der Sensorwerte.
        Args:
            drone_pos (tuple): (x, y, z)-Position der Drohne.
            drone_yaw (float): Yaw-Winkel (in Radiant).
            raycast_results (dict): z. B. { 'front': d_front, 'back': d_back,
                                            'left': d_left, 'right': d_right, 'up': d_up }
        """
        x, y, _ = drone_pos

        # Iterate through 5x5 grid centered on current position --> 3x3 grid

        grid_x, grid_y = self.world_to_grid(x, y)
        # self.path.append((grid_x, grid_y))
        # Markiere aktuelle Zelle als frei:
        self.occupancy_grid[grid_x, grid_y] = self.frei_value
        self.grid_x = grid_x
        self.grid_y = grid_y
        # self.counter_free_space += 1
        

        # Definiere Richtungswinkel:
        angles = {"front": drone_yaw, "back": drone_yaw + np.pi, "left": drone_yaw + np.pi / 2, "right": drone_yaw - np.pi / 2}
        for direction in ["front", "back", "left", "right"]:
            distance = raycast_results.get(direction, 9999)
            if distance < 9999:  # Treffer – Wand erkannt
                angle = angles[direction]
                end_x = x + distance * np.cos(angle)
                end_y = y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                # Markiere Zellen entlang der Strahlbahn als frei:
                for cx, cy in cells[:-1]:
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and self.occupancy_grid[cx, cy] != self.wand_value and self.occupancy_grid[cx, cy] != self.besucht_value and self.occupancy_grid[cx, cy] != self.actual_Position_value:
                        self.occupancy_grid[cx, cy] = self.frei_value
                        self.counter_free_space += 1
                # Markiere den Endpunkt als Wand:
                if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size and self.occupancy_grid[end_grid_x, end_grid_y] != self.frei_value:
                    self.occupancy_grid[end_grid_x, end_grid_y] = self.wand_value

        self.Prev_DrohnePosition = self.DrohnePosition
        self.DrohnePosition = []

        for i in range(max(0, grid_x - 2), min(80, grid_x + 3)):
            for j in range(max(0, grid_y - 2), min(80, grid_y + 3)):
                self.path.append((i, j))
                self.DrohnePosition.append((i, j))
                if self.occupancy_grid[i, j] == self.unbekannt_value or self.occupancy_grid[i,j] == self.frei_value or self.occupancy_grid[i, j] == self.actual_Position_value and self.occupancy_grid[i, j] != self.wand_value:
                    self.occupancy_grid[i, j] = self.besucht_value
                # if self.occupancy_grid[i, j] == self.besucht_value:
                #     self.occupancy_grid[i, j] = self.actual_Position_value
        
        self.occupancy_grid[grid_x, grid_y] = self.actual_Position_value


    def get_line(self, x0, y0, x1, y1):
        """Berechnet Zellen entlang einer Linie (Bresenham-Algorithmus)."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x, y))
        return cells

    def visualize(self):
        """Visualisiert die aktuelle SLAM Map (zum Debuggen).(siehe in def step() ganz weit unten)"""
        # Create figure without displaying
        # plt.ioff()  # Turn off interactive mode
        # plt.figure(figsize=(6, 6))
        # plt.imshow(np.squeeze(self.occupancy_grid).T, cmap="gray", origin="lower")

        # # if self.Prev_DrohnePosition:
        # #     prev_position = np.array(self.Prev_DrohnePosition)
        # #     plt.plot(prev_position[:, 0], prev_position[:, 1], "r-", linewidth=2)
        # #     plt.plot(prev_position[-1, 0], prev_position[-1, 1], "ro", markersize=5)

        # # if self.path:
        # #     path = np.array(self.path)
        # #     plt.plot(path[:, 0], path[:, 1], "r-", linewidth=2)
        # #     plt.plot(path[-1, 0], path[-1, 1], "ro", markersize=5)

        # # if self.DrohnePosition:
        # #     position = np.array(self.DrohnePosition)
        # #     plt.plot(position[:, 0], position[:, 1], "g-", linewidth=2)
        # #     # plt.plot(position[-1, 0], position[-1, 1], 'go', markersize=5)

        # plt.colorbar(label="Occupancy (-1: unbekannt, 0: frei, 1: Wand, 2: besucht)")
        # plt.title("SLAM Map")
        # plt.xlabel("X (grid cells)")
        # plt.ylabel("Y (grid cells)")
        # self.OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output_SLAM_MAP")
        # # create output folder if it doesn't exist
        # if not os.path.exists(self.OUTPUT_FOLDER):
        #     os.makedirs(self.OUTPUT_FOLDER)

        # # Save plot to file with current date and time
        # current_time = time.strftime("%Y%m%d-%H%M%S")
        # self.Latest_slam_map_path = os.path.join(self.OUTPUT_FOLDER, f"slam_map_{current_time}.png")
        # plt.savefig(self.Latest_slam_map_path)
        # plt.close()
        # plt.ion()  # Turn interactive mode back on
