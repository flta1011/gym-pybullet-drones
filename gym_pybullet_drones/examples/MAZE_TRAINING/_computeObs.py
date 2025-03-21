import numpy as np


def _computeObs(self):
    """Returns the current observation of the environment.
    10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

    28.2.25: vereinfachte Observation Space: in jede Richtung vorne, hinten, links, rechts, oben folgende Outputs möglich
    - 0: zu nahe an der Wand,
    - 1: Wand kommt näher,
    - 2: safe Distance,
    - 9999: Sensor oben frei

    Returns (28.2.25):
    -------
    ndarray
        A Box() of shape (NUM_DRONES,5) -> vorne, hinten, links, rechts, oben (1,9999)
        -> 0: zu nahe an der Wand,
        -> 1: Wand kommt näher,
        -> 2: safe Distance,
        -> 9999: Sensor oben frei
    """

    match self.OBSERVATION_TYPE:
        case "O1":  # XYZ Position, Yaw, Raycast readings
            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            # Select specific values from obs and concatenate them directly
            obs = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            # NOTE - nachfolgend auf vereinfachte Observation Space umgestellt (28.2.25):
            # Modify observation based on distance thresholds
            modified_obs = []

            # NOTE - neue Tests mit X,Y, Yaw Position der Drohne (28.2.25) übergeben
            modified_obs.append(round(state[0], 3))  # x-Position
            modified_obs.append(round(state[1], 3))  # y-Position
            modified_obs.append(round(state[9], 3))  # Yaw-Position

            # abstände anhängen mit 3 Nachkommastellen
            for distance in obs:
                modified_obs.append(round(distance, 3))

            # raycast oben noch anhängen
            if state[25] < 1:
                modified_obs.append(1)
            else:
                modified_obs.append(9999)

            return np.array(modified_obs, dtype=np.float32)  # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,9999)

        case "O2":  # 4 Kanalig Bild Slam, X, Y, Yaw Position
            """
            Baut einen 5-Kanal-Tensor auf:
            - Kanal 1: Normalisierte SLAM Map (Occupancy Map)
            - Kanal 2: Konstanter Wert des normierten x (angenommen, x ∈ [-4,4])
            - Kanal 3: Konstanter Wert des normierten y (angenommen, y ∈ [-4,4])
            - Kanal 4: sin(yaw)
            - Kanal 5: cos(yaw)
            """

            # Get current drone state
            state = self._getDroneStateVector(0)

            # Get SLAM map and normalize it
            slam_map = self.slam.occupancy_grid
            # norm_map = np.zeros_like(slam_map, dtype=np.float32)
            # norm_map[slam_map == -1] = 0.2   # unbekannt
            # norm_map[slam_map == 0] = 0.9    # frei
            # norm_map[slam_map == 1] = 0.0    # Wand
            # norm_map[slam_map == 2] = 0.5    # besucht

            # Get drone position from state
            pos = state[0:2]  # x,y position

            # Achtung: in der Simulation sind nie negative Werte zu erwarten, da die Mazes so gespant sind, das Sie immer positive Werte aufweisen. In echt kann die Drohne aber später auch negative Werte erhalten.

            # Normalisiere x und y: Angenommener Bereich [-5, 5]
            norm_x = (pos[0] + 4) / 8
            norm_y = (pos[1] + 4) / 8

            # Muss auf die Input Shape des DQN angepasst werden: (grid_size, grid_size)
            pos_x_channel = np.full((self.grid_size, self.grid_size), norm_x, dtype=np.float32)
            pos_y_channel = np.full((self.grid_size, self.grid_size), norm_y, dtype=np.float32)

            # Yaw in zwei Kanäle: sin und cos
            yaw = state[9]  # [9]=yaw-Winkel
            yaw_sin_channel = np.full((self.grid_size, self.grid_size), np.sin(yaw), dtype=np.float32)
            yaw_cos_channel = np.full((self.grid_size, self.grid_size), np.cos(yaw), dtype=np.float32)

            # Staple die 5 Kanäle zusammen: Shape = (5, grid_size, grid_size)
            # obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            self.obs = obs  # für Visualisierung in dem Dashboard

            # # Save the SLAM map as an image
            # plt.imshow(slam_map, cmap='gray', origin='lower')
            # plt.colorbar(label='Occupancy')
            # plt.title('SLAM Map')
            # plt.xlabel('X (grid cells)')
            # plt.ylabel('Y (grid cells)')
            # output_folder = os.path.join(os.path.dirname(__file__), 'output_SLAM_MAP')
            # if not os.path.exists(output_folder):
            #     os.makedirs(output_folder)
            # current_time = time.strftime("%Y%m%d-%H%M%S")
            # slam_map_path = os.path.join(output_folder, f"slam_map_{current_time}.png")
            # plt.savefig(slam_map_path)
            # plt.close()

            return obs
