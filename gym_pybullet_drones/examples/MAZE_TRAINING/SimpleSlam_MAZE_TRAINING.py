import os
import time

import matplotlib.pyplot as plt
import numpy as np


class SimpleSlam:
    def __init__(self, map_size=9, cropped_map_size=6.4, resolution=0.05, init_position=None):
        """Erstellt eine leere Occupancy-Grid Map.
        Args:
            map_size (float): Seitengröße der Map in Metern (z. B. 8 m).
            resolution (float): Seitengröße einer Zelle, sodass grid_size ~60 ergibt.
            init_position (tuple): (x, y)-Startposition der Drohne in Metern.
        """
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.cropped_map_size_grid = int((cropped_map_size / resolution) / 2)

        # Initialisiere die Map-Werte:
        self.unbekannt_value = 0
        self.frei_value = 200
        self.wand_value = 50
        self.besucht_value = 125
        self.actual_Position_value = 255
        self.init = init_position
        self.resolution = resolution
        self.previous_grid_x = 0
        self.previous_grid_y = 0

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
        self.cropped_grid = self.unbekannt_value * np.ones((self.cropped_map_size_grid, self.cropped_map_size_grid, 1))
        self.center = self.grid_size // 2
        self.center_cropped = self.cropped_map_size_grid // 2
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
        grid_x = int(self.center + x / self.resolution)
        grid_y = int(self.center + y / self.resolution)
        return grid_x, grid_y

    def is_within_bounds(self, grid_x, grid_y):
        """Überprüft, ob die Drohne innerhalb der Map-Grenzen bleibt."""
        return 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size

    def update(self, drone_pos, drone_yaw, raycast_results):

        # NOTE - Werte anpassen wie ich welches Szenario darstellen möchte TBD 24.03.2025
        # NOTE - 0.2: unbekannt, 0.9: frei, 0.0: Wand, 0.5: besucht (Sensor oben frei), 0.7 = aktuelle Position

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
            if distance < 4:  # Treffer – Wand erkannt
                angle = angles[direction]
                end_x = x + distance * np.cos(angle)
                end_y = y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                # Markiere Zellen entlang der Strahlbahn als frei:
                for cx, cy in cells[:-1]:  # alles bis auf den Endpunkt frei markieren
                    if (
                        0 <= cx < self.grid_size
                        and 0 <= cy < self.grid_size
                        and self.occupancy_grid[cx, cy] != self.wand_value
                        and self.occupancy_grid[cx, cy] != self.besucht_value
                        and self.occupancy_grid[cx, cy] != self.actual_Position_value
                    ):
                        self.occupancy_grid[cx, cy] = self.frei_value
                        self.counter_free_space += 1
                # Markiere den Endpunkt als Wand:
                if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size and self.occupancy_grid[end_grid_x, end_grid_y] != self.frei_value:
                    self.occupancy_grid[end_grid_x, end_grid_y] = self.wand_value

            elif distance >= 4:  # Distanz ist auf 4 m gekappt, da das die Range des Sensors ist --> alles als frei markieren
                angle = angles[direction]
                end_x = x + distance * np.cos(angle)
                end_y = y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                # Markiere Zellen entlang der Strahlbahn als frei:
                for cx, cy in cells[:]:  # alles frei markieren
                    if (
                        0 <= cx < self.grid_size
                        and 0 <= cy < self.grid_size
                        and self.occupancy_grid[cx, cy] != self.wand_value
                        and self.occupancy_grid[cx, cy] != self.besucht_value
                        and self.occupancy_grid[cx, cy] != self.actual_Position_value
                    ):
                        self.occupancy_grid[cx, cy] = self.frei_value
                        self.counter_free_space += 1

        self.Prev_DrohnePosition = self.DrohnePosition

        self.DrohnePosition = []
        # Überschreibe die vorherige aktuelle Position mit besucht_value
        if hasattr(self, "i_value_previous") and hasattr(self, "j_value_previous"):
            for i_prev, j_prev in zip(self.i_value_previous, self.j_value_previous):
                self.occupancy_grid[i_prev, j_prev] = self.besucht_value

        # self.cropped_grid = self.occupancy_grid[]

        # Speichere die aktuelle Position als vorherige Position
        self.i_value_previous = []
        self.j_value_previous = []

        for i in range(max(0, grid_x - 1), min(self.grid_size, grid_x + 2)):
            for j in range(max(0, grid_y - 1), min(self.grid_size, grid_y + 2)):
                self.path.append((i, j))
                self.DrohnePosition.append((i, j))
                if (
                    self.occupancy_grid[i, j] == self.besucht_value
                    or self.occupancy_grid[i, j] == self.unbekannt_value
                    or self.occupancy_grid[i, j] == self.frei_value
                    or self.occupancy_grid[i, j] == self.actual_Position_value
                    and self.occupancy_grid[i, j] != self.wand_value
                ):
                    # self.occupancy_grid[i, j] = self.besucht_value
                    self.occupancy_grid[i, j] = self.actual_Position_value

                    self.i_value_previous.append(i)
                    self.j_value_previous.append(j)

                # if self.occupancy_grid[i, j] == self.besucht_value:
                #     self.occupancy_grid[i, j] = self.actual_Position_value

        # Compare previous and current positions to determine movement direction
        if self.grid_y and self.grid_x and self.previous_grid_x and self.previous_grid_y:
            # # Calculate the center of the previous and current positions
            # delta_x = self.grid_x - self.previous_grid_x
            # delta_y = self.grid_y - self.previous_grid_y

            # if abs(delta_x) > abs(delta_y):  # Movement is primarily horizontal
            #     if delta_x > 0:
            #         movement = "right"
            #     elif delta_x < 0:
            #         movement = "left"
            # else:  # Movement is primarily vertical
            #     if delta_y > 0:
            #         movement = "forward"
            #     elif delta_y < 0:
            #         movement = "backward"

            # Update cropped grid based on movement
            cropped_start_x = int(max(0, self.grid_x - self.center_cropped))
            cropped_end_x = int(min(self.grid_size, self.grid_x + self.center_cropped))
            cropped_start_y = int(max(0, self.grid_y - self.center_cropped))
            cropped_end_y = int(min(self.grid_size, self.grid_y + self.center_cropped))

            # Ensure the cropped grid is exactly 64x64
            if cropped_end_x - cropped_start_x != 64:
                cropped_end_x = cropped_start_x + 64
            if cropped_end_y - cropped_start_y != 64:
                cropped_end_y = cropped_start_y + 64

            self.cropped_grid = self.occupancy_grid[cropped_start_x:cropped_end_x, cropped_start_y:cropped_end_y]

        self.previous_grid_x = self.grid_x
        self.previous_grid_y = self.grid_y

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
        """Visualisiert die aktuelle SLAM Map und die gecroppte Map (zum Debuggen)."""
        # Visualize full SLAM map
        plt.ioff()  # Turn off interactive mode
        plt.figure(figsize=(6, 6))
        plt.imshow(np.squeeze(self.occupancy_grid).T, cmap="gray", origin="lower")
        plt.colorbar(label="Occupancy (-1: unbekannt, 0: frei, 1: Wand, 2: besucht)")
        plt.title("SLAM Map")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")
        self.OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output_SLAM_MAP")
        # Create output folder if it doesn't exist
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

        # Save full SLAM map to file
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.Latest_slam_map_path = os.path.join(self.OUTPUT_FOLDER, f"slam_map_{current_time}.png")
        plt.savefig(self.Latest_slam_map_path)
        plt.close()

        # Visualize cropped SLAM map
        plt.figure(figsize=(6, 6))
        plt.imshow(np.squeeze(self.cropped_grid).T, cmap="gray", origin="lower")
        plt.colorbar(label="Occupancy (-1: unbekannt, 0: frei, 1: Wand, 2: besucht)")
        plt.title("Cropped SLAM Map")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")

        # Save cropped SLAM map to file
        self.Latest_cropped_map_path = os.path.join(self.OUTPUT_FOLDER, f"cropped_map_{current_time}.png")
        plt.savefig(self.Latest_cropped_map_path)
        plt.close()
        plt.ion()  # Turn interactive mode back on
