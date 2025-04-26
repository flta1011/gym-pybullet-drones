import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


def _observationSpace(observation_type, cropped_map_size_grid):
    last_actions_shape = 20  # SECTION Nummer ändern für last actions

    match observation_type:
        case "O1":  # X, Y, YAW, Raycast readings
            """Returns the observation space.
            Simplified observation space with key state variables.

            10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

            Returns
            -------
            ndarray
                A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

                Information of the self._getDroneStateVector:
                    ndarray
                    1x Raycast reading (forward) [21]          -> 0 bis 4

            """

            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array(
                [-99, -99, -2 * np.pi, 0, 0, 0, 0, 0]
            )  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            obs_upper_bound = np.array([99, 99, 2 * np.pi, 4, 4, 4, 4, 4])  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        case "O2":  # 5 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (5, grid_size, grid_size) containing:
            - Channel 1: Normalized SLAM map (values in [0,1])
            - Channel 2: Normalized x position (values in [0,1])
            - Channel 3: Normalized y position (values in [0,1])
            - Channel 4: sin(yaw) (values in [-1,1])
            - Channel 5: cos(yaw) (values in [-1,1])
            """
            grid_size = int(cropped_map_size_grid)

            # Create proper shaped arrays for low and high bounds
            low = np.zeros((5, grid_size, grid_size), dtype=np.float32)
            high = np.ones((5, grid_size, grid_size), dtype=np.float32)

            # Set specific ranges for each channel
            low[3, :, :] = -1.0  # sin(yaw) lower bound
            low[4, :, :] = -1.0  # cos(yaw) lower bound

            return spaces.Box(low=low, high=high, dtype=np.float32)

        case "O3":  # 7 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (7, grid_size, grid_size) containing:
            - Channel 1: Normalized SLAM map (values in [0,1])
            - Channel 2: Normalized x position (values in [0,1])
            - Channel 3: Normalized y position (values in [0,1])
            - Channel 4: sin(yaw) (values in [-1,1])
            - Channel 5: cos(yaw) (values in [-1,1])
            - Channel 6: last Clipped Action (values in [-1,1])
            - Channel 7: second Last Clipped Action (values in [-1,1])
            - Channel 8: third Last Clipped Action (values in [-1,1])
            """
            grid_size = int(cropped_map_size_grid)

            # Create proper shaped arrays for low and high bounds
            low = np.zeros((8, grid_size, grid_size), dtype=np.float32)
            high = np.ones((8, grid_size, grid_size), dtype=np.float32)

            # Set specific ranges for each channel
            low[3, :, :] = -1.0  # sin(yaw) lower bound
            low[4, :, :] = -1.0  # cos(yaw) lower bound

            high[3, :, :] = 1.0  # sin(yaw) lower bound
            high[4, :, :] = 1.0  # cos(yaw) lower bound
            high[5, :, :] = 4.0  # last Clipped Action lower bound
            high[6, :, :] = 4.0  # second Last Clipped Action lower bound
            high[7, :, :] = 4.0  # third Last Clipped Action lower bound

            return spaces.Box(low=low, high=high, dtype=np.float32)

        case "O4":  # 1 Kanal für CNN-DQN nur das Bild
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (1, grid_size, grid_size) containing:
            - Channel 1: Grayscale SLAM map (values in [0,255])
            """
            grid_size = int(cropped_map_size_grid)

            # Create proper shaped arrays for low and high bounds
            low = np.zeros((grid_size, grid_size, 1), dtype=np.uint8)
            high = np.full((grid_size, grid_size, 1), 255, dtype=np.uint8)

            return spaces.Box(low=low, high=high, dtype=np.uint8)

        case "O5":  # X, Y, YAW, Raycast readings, last clipped action, second last clipped action, third last clipped action
            """Returns the observation space.
            Simplified observation space with key state variables.

            10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

            Returns
            -------
            ndarray
                A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

                Information of the self._getDroneStateVector:
                    ndarray
                    1x Raycast reading (forward) [21]          -> 0 bis 4
                    Last Action (values in [-1,1])
            self.number_last_actions
            """

            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array(
                [-99, -99, -2 * np.pi, 0, 0, 0, 0] + [-1] * last_actions_shape
            )  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up, last actions

            obs_upper_bound = np.array(
                [99, 99, 2 * np.pi, 4, 4, 4, 4] + [6] * last_actions_shape
            )  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up, last actions

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        case "O6":  # 7 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (7, grid_size, grid_size) containing:
            - Channel 1: Normalized SLAM map (values in [0,1])
            - Channel 2: Normalized x position (values in [0,1])
            - Channel 3: Normalized y position (values in [0,1])
            - Channel 4: sin(yaw) (values in [-1,1])
            - Channel 5: cos(yaw) (values in [-1,1])
            - Channel 6: last Clipped Action (values in [-1,1])
            - Channel 7: second Last Clipped Action (values in [-1,1])
            - Channel 8: third Last Clipped Action (values in [-1,1])
            """
            grid_size = int(cropped_map_size_grid)

            observationSpace = spaces.Dict(
                {
                    "image": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 1), dtype=np.uint8),  # Grayscale image
                    "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "sin_yaw": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "cos_yaw": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "last_action": spaces.Box(
                        low=0,
                        high=6,
                        shape=(last_actions_shape,),
                        dtype=np.float32,
                    ),
                }
            )

            return observationSpace

        case "O7":  # 7 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (7, grid_size, grid_size) containing:
            - Channel 1: Normalized SLAM map (values in [0,1])
            - Channel 2: Normalized x position (values in [0,1])
            - Channel 3: Normalized y position (values in [0,1])
            - Channel 4: sin(yaw) (values in [-1,1])
            - Channel 5: cos(yaw) (values in [-1,1])
            - Channel 6: last Clipped Action (values in [-1,1])
            - Channel 7: second Last Clipped Action (values in [-1,1])
            - Channel 8: third Last Clipped Action (values in [-1,1])
            """
            grid_size = int(cropped_map_size_grid)

            observationSpace = spaces.Dict(
                {
                    "image": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 1), dtype=np.uint8),  # Grayscale image
                    "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "sin_yaw": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "cos_yaw": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "last_action": spaces.Box(
                        low=0,
                        high=6,
                        shape=(last_actions_shape,),
                        dtype=np.float32,
                    ),
                    "raycast": spaces.Box(low=0, high=4, shape=(4,), dtype=np.float32),
                }
            )

            return observationSpace

        case "O8":  # 7 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (7, grid_size, grid_size) containing:
            - Channel 1: SLAM map (values in [0,255])
            - Channel 2: X-Position (values in [-inf,inf])
            - Channel 3: Y-Position (values in [-ing,inf])
            - Channel 4: Raycast readings (values in [0,4])
            - Channel 5: Interest Values (values in [0,32400])
            - Channel 6: n last Clipped Actions (values in [0, 3])
            """
            last_actions_size = last_actions_shape  # Number of last clipped actions

            # Define the low and high bounds for the flattened observation
            low = np.concatenate(
                (
                    np.array([-np.inf], dtype=np.float32),  # X position
                    np.array([-np.inf], dtype=np.float32),  # Y position
                    np.zeros(4, dtype=np.float32),  # Raycast readings (values in [0, 4])
                    np.zeros(4, dtype=np.float32),  # Interest values
                    np.zeros(last_actions_size, dtype=np.float32),  # Last clipped actions
                )
            )

            high = np.concatenate(
                (
                    np.array([np.inf], dtype=np.float32),  # X position
                    np.array([np.inf], dtype=np.float32),  # Y position
                    np.full(4, 4, dtype=np.float32),  # Raycast readings (values in [0, 4])
                    np.full(4, 32400, dtype=np.float32),  # Interest values
                    np.full(last_actions_size, 6, dtype=np.float32),  # Last clipped actions
                )
            )

            # Return the flattened observation space
            return spaces.Box(low=low, high=high, dtype=np.float32)

        case "O9":  # 7 Kanäle für CNN-DQN
            """
            Returns the observation space for the CNN-DQN model.
            The observation space is a Box with shape (7, grid_size, grid_size) containing:
            - Channel 1: Normalized SLAM map (values in [0,1])
            - Channel 2: Normalized x position (values in [0,1])
            - Channel 3: Normalized y position (values in [0,1])
            - Channel 6: last Clipped Action (values in [-1,1])
            - Channel 7: second Last Clipped Action (values in [-1,1])
            - Channel 8: third Last Clipped Action (values in [-1,1])
            """
            grid_size = int(cropped_map_size_grid)

            observationSpace = spaces.Dict(
                {
                    "image": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 1), dtype=np.uint8),  # Grayscale image
                    "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "raycast": spaces.Box(low=0, high=4, shape=(4,), dtype=np.float32),
                    "interest_values": spaces.Box(low=0, high=32400, shape=(4,), dtype=np.uint8),
                    "last_clipped_actions": spaces.Box(low=0, high=6, shape=(last_actions_shape,), dtype=np.float32),
                }
            )

            return observationSpace


class OBSManager:
    def __init__(self, observation_type):
        self.SLAM = SimpleSlam()
        self.observation_type = observation_type
        self.observation = _observationSpace(self.observation_type, self.SLAM.cropped_map_size_grid)
        self.interest_values = np.zeros(4, dtype=int)
        self.slam_update_callback = None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_folder = os.path.join(os.path.dirname(__file__), "observation_logs")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.observation_csv_path = os.path.join(output_folder, f"observation_log_{timestamp}.csv")

    def _save_observation_to_csv(self, file_path=None):
        """Save the current observation space to a CSV file.

        Parameters
        ----------
        file_path : str, optional
            Path to save the CSV file. If None, a default path will be created.
        """

        if file_path is None:
            # Create default file path
            timestamp = datetime.now().strftime("%Y%m%d-%H")
            output_folder = os.path.join(os.path.dirname(__file__), "observation_logs")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            file_path = os.path.join(output_folder, f"observation_log_{timestamp}.csv")

        # Check if file already exists to determine if we need to write headers
        file_exists = os.path.isfile(file_path)

        # Open the CSV file in append mode
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # If file doesn't exist, write headers based on observation type
            if not file_exists:
                if self.observation_type == "O8":
                    headers = ["x", "y"]
                    headers.extend([f"raycast_{dir}" for dir in ["front", "back", "left", "right"]])
                    headers.extend([f"interest_{i}" for i in range(4)])
                    headers.extend([f"last_action_{i}" for i in range(len(self.observation) - 10)])
                    writer.writerow(headers)
                    writer.writerow([])  # Empty line after headers

                elif self.observation_type == "O9":
                    headers = ["x", "y"]
                    headers.extend([f"raycast_{dir}" for dir in ["front", "back", "left", "right"]])
                    headers.extend([f"interest_{i}" for i in range(4)])
                    headers.extend([f"last_action_{i}" for i in range(len(self.observation["last_clipped_actions"]))])
                    writer.writerow(headers)
                    writer.writerow([])  # Empty line after headers

            # Write the current observation data
            if self.observation_type == "O8":
                # For O8, observation is a numpy array
                writer.writerow(self.observation)

            elif self.observation_type == "O9":
                # For O9, observation is a dictionary
                row_data = []
                row_data.extend(self.observation["x"])
                row_data.extend(self.observation["y"])
                row_data.extend(self.observation["raycast"])
                row_data.extend(self.observation["interest_values"])
                row_data.extend(self.observation["last_clipped_actions"])
                writer.writerow(row_data)

    def set_slam_update_callback(self, callback):
        """Set a callback function to be called after SLAM update."""
        self.slam_update_callback = callback

    def update(self, position, measurements, last_actions):
        self.SLAM.update(drone_pos=position, drone_yaw=measurements["yaw"], raycast_results=measurements)
        if self.slam_update_callback:
            self.slam_update_callback(self.SLAM.get_cropped_slam_map())
        self._compute_interest_values()

        match self.observation_type:
            case "O8":
                """X-Pos, Y-Pos, Raycast Readings, Interest Values, Last Actions"""
                # Create the observation array properly
                # First, create a list of all values
                obs_list = [position[0], position[1], measurements["front"], measurements["back"], measurements["left"], measurements["right"]]
                # Extend with interest values and last actions
                obs_list.extend(self.interest_values)
                obs_list.extend(last_actions)
                # Convert the whole list to a numpy array
                self.observation = np.array(obs_list)

            case "O9":
                """Slam-image, X-Pos, Y-Pos, Raycast Readings, Interest Values, Last Actions"""
                raycasts = np.array([measurements["front"], measurements["back"], measurements["left"], measurements["right"]])
                self.observation = dict(
                    {
                        "image": self.SLAM.cropped_grid,
                        "x": np.array([round(position[0], 3)], dtype=np.float32),
                        "y": np.array([round(position[1], 3)], dtype=np.float32),
                        "raycast": raycasts,
                        "interest_values": self.interest_values,
                        "last_clipped_actions": last_actions,
                    }
                )

        # Save the observation to CSV file
        self._save_observation_to_csv()

    def _compute_interest_values(self):
        """
        Compute interest values based on current drone position.
        """
        drone_position = np.argwhere(self.SLAM.occupancy_grid == 255)  # Get the drone position
        # Check if drone position was found
        if len(drone_position) == 0:
            print("Warning: Drone position not found in occupancy grid")
            return self.interest_values

        free_areas = np.argwhere(self.SLAM.occupancy_grid == 200)  # Get the free areas

        min_x_y = drone_position[0]
        max_x_y = [min_x_y[0] + 5, min_x_y[1] + 5]

        # Reset interest values
        self.interest_values = np.zeros(4, dtype=int)

        # Iterate through free areas and calculate their relation to the drone
        for area in free_areas:
            if area[0] < min_x_y[0]:
                self.interest_values[0] += 1  # "front"
            elif area[0] > max_x_y[0]:
                self.interest_values[1] += 1  # "back"
            if area[1] < min_x_y[1]:
                self.interest_values[3] += 1  # "left"
            elif area[1] > max_x_y[1]:
                self.interest_values[2] += 1  # "right"

        return self.interest_values

    def get_observation(self):
        return self.observation


class SimpleSlam:
    def __init__(self, map_size=18, cropped_map_size=6.4, resolution=0.05, init_position=None):
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

        #!SECTION Erkenntnis damit im Training Koordinaten in Globalen waren und hier aber die Drohne Koordinaten hat
        neues_x = x
        neues_y = y

        # Iterate through 5x5 grid centered on current position --> 3x3 grid

        grid_x, grid_y = self.world_to_grid(neues_x, neues_y)

        # FALLBACK für Errror "index 210 is out of bounds for axis 0 with size 180"
        if grid_x > self.grid_size:
            grid_x = self.grid_size
        if grid_y > self.grid_size:
            grid_y = self.grid_size
        # Fallback für Cases < 0 (sollten eigentlich nie auftreten)
        if grid_x < 0:
            grid_x = 0
        if grid_y < 0:
            grid_y = 0

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
                end_x = neues_x + distance * np.cos(angle)
                end_y = neues_y + distance * np.sin(angle)
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
                end_x = neues_x + distance * np.cos(angle)
                end_y = neues_y + distance * np.sin(angle)
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

    def get_full_slam_map(self):
        return self.occupancy_grid

    def get_cropped_slam_map(self):
        return self.cropped_grid
