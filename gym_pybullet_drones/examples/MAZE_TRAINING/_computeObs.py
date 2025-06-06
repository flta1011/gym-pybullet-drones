import numpy as np


def _computeObs(self):
    """Returns the current observation of the environment.
    10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

    28.2.25: vereinfachte Observation Space: in jede Richtung vorne, hinten, links, rechts, oben folgende Outputs möglich
    - 0: zu nahe an der Wand,
    - 1: Wand kommt näher,
    - 2: safe Distance,
    - 4: Sensor oben frei

    Returns (28.2.25):
    -------
    ndarray
        A Box() of shape (NUM_DRONES,5) -> vorne, hinten, links, rechts, oben (1,4)
        -> 0: zu nahe an der Wand,
        -> 1: Wand kommt näher,
        -> 2: safe Distance,
        -> 4: Sensor oben frei
    """

    match self.OBSERVATION_TYPE:
        case "O1":  # XYZ Position, Yaw, Raycast readings
            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Select specific values from obs and concatenate them directly
            obs = [
                state[21],
                state[22],
                state[23],
                state[24],
            ]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            # NOTE - nachfolgend auf vereinfachte Observation Space umgestellt (28.2.25):
            # Modify observation based on distance thresholds
            modified_obs = []

            # NOTE - neue Tests mit X,Y, Yaw Position der Drohne (28.2.25) übergeben
            modified_obs.append(round(relative_x, 3))  # x-Position
            modified_obs.append(round(relative_y, 3))  # y-Position
            modified_obs.append(round(state[9], 3))  # Yaw-Position

            # abstände anhängen mit 3 Nachkommastellen
            for distance in obs:
                modified_obs.append(round(distance, 3))

            # raycast oben noch anhängen
            if state[25] < 1:
                modified_obs.append(1)
            else:
                modified_obs.append(4)

            return np.array(modified_obs, dtype=np.float32)  # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,4)

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

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Get SLAM map and normalize it
            slam_map = self.slam.cropped_grid
            # norm_map = np.zeros_like(slam_map, dtype=np.float32)
            # norm_map[slam_map == -1] = 0.2   # unbekannt
            # norm_map[slam_map == 0] = 0.9    # frei
            # norm_map[slam_map == 1] = 0.0    # Wand
            # norm_map[slam_map == 2] = 0.5    # besucht

            # Get drone position from state
            pos = [relative_x, relative_y]  # x,y position

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
            obs = np.stack(
                [
                    slam_map,
                    pos_x_channel,
                    pos_y_channel,
                    yaw_sin_channel,
                    yaw_cos_channel,
                ],
                axis=0,
            )
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

        case "O3":  # 4 Kanalig Bild Slam, X, Y, Yaw Position
            """
            Baut einen 8-Kanal-Tensor auf:
            - Kanal 1: Normalisierte SLAM Map (Occupancy Map)
            - Kanal 2: Konstanter Wert des normierten x (angenommen, x ∈ [-4,4])
            - Kanal 3: Konstanter Wert des normierten y (angenommen, y ∈ [-4,4])
            - Kanal 4: sin(yaw)
            - Kanal 5: cos(yaw)
            """

            # Get current drone state
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Get SLAM map and normalize it
            slam_map = self.slam.cropped_grid
            # norm_map = np.zeros_like(slam_map, dtype=np.float32)
            # norm_map[slam_map == -1] = 0.2   # unbekannt
            # norm_map[slam_map == 0] = 0.9    # frei
            # norm_map[slam_map == 1] = 0.0    # Wand
            # norm_map[slam_map == 2] = 0.5    # besucht

            # Get drone position from state
            pos = [relative_x, relative_y]  # x,y position

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

            last_Action_1 = state[26]  # [25]=last_Action
            last_Action_2 = state[27]  # [26]=last_Action
            last_Action_3 = state[28]  # [27]=last_Action
            last_Action_channel_1 = np.full((self.grid_size, self.grid_size), last_Action_1, dtype=np.float32)
            last_Action_channel_2 = np.full((self.grid_size, self.grid_size), last_Action_2, dtype=np.float32)
            last_Action_channel_3 = np.full((self.grid_size, self.grid_size), last_Action_3, dtype=np.float32)

            # Staple die 5 Kanäle zusammen: Shape = (5, grid_size, grid_size)
            # obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            obs = np.stack(
                [
                    slam_map,
                    pos_x_channel,
                    pos_y_channel,
                    yaw_sin_channel,
                    yaw_cos_channel,
                    last_Action_channel_1,
                    last_Action_channel_2,
                    last_Action_channel_3,
                ],
                axis=0,
            )
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

        case "O4":  # 1 Kanalig Bild Slam, X, Y, Yaw Position
            """
            Baut einen 1-Kanal-Tensor auf:
            - Kanal 1: Normalisierte SLAM Map (Occupancy Map)
            """

            # Get SLAM map and normalize it
            slam_map = self.slam.cropped_grid

            obs = np.stack([slam_map], axis=0)
            self.obs = obs  # für Visualisierung in dem Dashboard

            return obs

        case "O5":  # XYZ Position, Yaw, Raycast readings, last clipped actions
            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Select specific values from obs and concatenate them directly
            obs = [
                state[21],
                state[22],
                state[23],
                state[24],
            ]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            # NOTE - nachfolgend auf vereinfachte Observation Space umgestellt (28.2.25):
            # Modify observation based on distance thresholds
            modified_obs = []

            # last_Action_1 = state[26]  # [25]=last_Action
            # last_Action_2 = state[27]  # [26]=last_Action
            # last_Action_3 = state[28]  # [27]=last_Action

            # NOTE - neue Tests mit X,Y, Yaw Position der Drohne (28.2.25) übergeben
            modified_obs.append(round(relative_x, 3))  # x-Position
            modified_obs.append(round(relative_y, 3))  # y-Position
            modified_obs.append(round(state[9], 3))  # Yaw-Position

            # abstände anhängen mit 3 Nachkommastellen
            for distance in obs:
                modified_obs.append(round(distance, 3))

            # NOTE - 7.4.25: wird nicht mehr benötigt
            # # raycast oben noch anhängen
            # if state[25] < 1:
            #     modified_obs.append(1)
            # else:
            #     modified_obs.append(4)

            # Ensure last_actions is flattened and appended correctly
            if isinstance(self.last_actions, (list, np.ndarray)):
                modified_obs.extend(self.last_actions)  # Append elements individually
            else:
                modified_obs.append(self.last_actions)  # Append directly if it's a single value

            return np.array(modified_obs, dtype=np.float32)  # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,4)

        case "O6":  # 4 Kanalig Bild Slam, X, Y, Yaw Position
            """
            Baut einen 8-Kanal-Tensor auf:
            - Kanal 1: Normalisierte SLAM Map (Occupancy Map)
            - Kanal 2: Konstanter Wert des normierten x (angenommen, x ∈ [-4,4])
            - Kanal 3: Konstanter Wert des normierten y (angenommen, y ∈ [-4,4])
            - Kanal 4: sin(yaw)
            - Kanal 5: cos(yaw)
            """

            # Get current drone state
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Get SLAM map and normalize it
            slam_map = self.slam.cropped_grid
            # norm_map = np.zeros_like(slam_map, dtype=np.float32)
            # norm_map[slam_map == -1] = 0.2   # unbekannt
            # norm_map[slam_map == 0] = 0.9    # frei
            # norm_map[slam_map == 1] = 0.0    # Wand
            # norm_map[slam_map == 2] = 0.5    # besucht

            # Get drone position from state
            # pos = state[0:2]  # x,y position

            # Achtung: in der Simulation sind nie negative Werte zu erwarten, da die Mazes so gespant sind, das Sie immer positive Werte aufweisen. In echt kann die Drohne aber später auch negative Werte erhalten.

            # Yaw in zwei Kanäle: sin und cos
            yaw = state[9]  # [9]=yaw-Winkel

            # Staple die 5 Kanäle zusammen: Shape = (5, grid_size, grid_size)
            # obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            obs = dict(
                {
                    "image": self.slam.cropped_grid,
                    "x": relative_x,
                    "y": relative_y,
                    "sin_yaw": np.sin(yaw),
                    "cos_yaw": np.cos(yaw),
                    "last_action": self.last_actions,
                }
            )
            self.obs = obs  # für Visualisierung in dem Dashboard

            return obs

        case "O7":  # 4 Kanalig Cropped Bild Slam, X, Y, Yaw Position
            """
            Baut einen 8-Kanal-Tensor auf:
            - Kanal 1: Normalisierte SLAM Map (Occupancy Map)
            - Kanal 2: Konstanter Wert des normierten x (angenommen, x ∈ [-4,4])
            - Kanal 3: Konstanter Wert des normierten y (angenommen, y ∈ [-4,4])
            - Kanal 4: sin(yaw)
            - Kanal 5: cos(yaw)
            """

            # Get current drone state
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Get SLAM map and normalize it
            slam_map = self.slam.cropped_grid

            raycasts = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            # Get drone position from state
            # pos = state[0:2]  # x,y position

            # Achtung: in der Simulation sind nie negative Werte zu erwarten, da die Mazes so gespant sind, das Sie immer positive Werte aufweisen. In echt kann die Drohne aber später auch negative Werte erhalten.

            # Yaw in zwei Kanäle: sin und cos
            yaw = state[9]  # [9]=yaw-Winkel

            # Staple die 5 Kanäle zusammen: Shape = (5, grid_size, grid_size)
            # obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            obs = dict(
                {
                    "image": self.slam.cropped_grid,
                    "x": relative_x,
                    "y": relative_y,
                    "sin_yaw": np.sin(yaw),
                    "cos_yaw": np.cos(yaw),
                    "last_action": self.last_actions,
                    "raycast": raycasts,
                }
            )
            self.obs = obs  # für Visualisierung in dem Dashboard

            return obs

        case "O8":  # X-Pos,Y-Pos, 4-raycast readings, 4-Interest Werte (Interest-Front,Back, left, right: Summe freie Flächen, die noch nicht besucht wurden), x mal last clipped actions
            # Modify observation based on distance thresholds
            modified_obs = []

            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # X- & Y-Position der Drohne
            modified_obs.append(relative_x)  # x-Position
            modified_obs.append(relative_y)  # y-Position

            # Raycast Readings
            # Select specific values from obs and concatenate them directly
            obs = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up
            # abstände anhängen mit 3 Nachkommastellen
            modified_obs.extend(obs)  # 4x raycast readings

            # Append interest values to the observation
            modified_obs.extend(self.interest_values)  # 4x interest values

            # Last clipped actions
            # Ensure last_actions is flattened and appended correctly
            if isinstance(self.last_actions, (list, np.ndarray)):
                modified_obs.extend(self.last_actions)  # Append elements individually
            else:
                modified_obs.append(self.last_actions)  # Append directly if it's a single value

            return np.array(modified_obs, dtype=np.float32)

        case "O9":  # Slam-image, x-Pos, y-Pos, racast readings,4-Interest Werte (Interest-Front,Back, left, right: Summe freie Flächen, die noch nicht besucht wurden), x mal last clipped actions
            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            absolute_x = state[0]  # x-Position
            absolute_y = state[1]  # y-Position
            initial_x = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0]
            initial_y = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][1]
            relative_x = absolute_x - initial_x  # relative x-Position
            relative_y = absolute_y - initial_y  # relative y-Position

            # Select specific values from obs and concatenate them directly
            raycasts = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            obs = dict(
                {
                    "image": self.slam.cropped_grid,
                    "x": round(relative_x, 3),
                    "y": round(relative_y, 3),
                    "raycast": raycasts,
                    "interest_values": self.interest_values,
                    "last_clipped_actions": self.last_actions,
                }
            )
            self.obs = obs  # für Visualisierung in dem Dashboard

            return obs

        case "O10":  # einfache Werte für einfachere, robusteres Training (MLP)
            # Get the current state of the drone
            state = self._getDroneStateVector(0)

            # Select specific values from obs and concatenate them directly
            obs = [
                round(state[21], 3),  # ray front
                round(state[22], 3),  # ray back
                round(state[23], 3),  # ray left
                round(state[24], 3),  # ray right
            ]
            obs.extend(self.last_actions)

            return np.array(obs, dtype=np.float32)
