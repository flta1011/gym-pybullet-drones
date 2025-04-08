import numpy as np
from gymnasium import spaces


def _observationSpace(self):
    match self.OBSERVATION_TYPE:
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
                    1x Raycast reading (forward) [21]          -> 0 bis 9999

            """

            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array(
                [-99, -99, -2 * np.pi, 0, 0, 0, 0, 0]
            )  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

            obs_upper_bound = np.array(
                [99, 99, 2 * np.pi, 9999, 9999, 9999, 9999, 9999]
            )  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

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
            grid_size = int(self.slam.cropped_map_size_grid)

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
            grid_size = int(self.slam.cropped_map_size_grid)

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
            grid_size = int(self.slam.cropped_map_size_grid)

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
                    1x Raycast reading (forward) [21]          -> 0 bis 9999
                    Last Action (values in [-1,1])
            self.number_last_actions
            """

            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array(
                [-99, -99, -2 * np.pi, 0, 0, 0, 0] + [-1] * self.number_last_actions
            )  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up, last actions

            obs_upper_bound = np.array(
                [99, 99, 2 * np.pi, 9999, 9999, 9999, 9999] + [6] * self.number_last_actions
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
            grid_size = int(self.slam.cropped_map_size_grid)

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
                        shape=(self.last_actions.shape[0],),
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
            grid_size = int(self.slam.cropped_map_size_grid)

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
                        shape=(self.last_actions.shape[0],),
                        dtype=np.float32,
                    ),
                    "raycast": spaces.Box(low=0, high=9999, shape=(4,), dtype=np.float32),
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
            - Channel 4: Raycast readings (values in [0,9999])
            - Channel 5: Interest Values (values in [0,32400])
            - Channel 6: n last Clipped Actions (values in [0, 3])
            """
            grid_size = int(self.slam.cropped_map_size_grid)

            observationSpace = spaces.Dict(
                {
                    "image": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 1), dtype=np.uint8),  # Grayscale image
                    "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # X-Position
                    "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # Y-Position
                    "raycast": spaces.Box(low=0, high=9999, shape=(4,), dtype=np.float32),  # Raycast readings
                    "interest_values": spaces.Box(low=0, high=32400, shape=(grid_size, grid_size), dtype=np.float32),  # Interest Values
                    "last_clipped_actions": spaces.Box(low=0, high=6, shape=(self.last_actions.shape[0],), dtype=np.float32),  # Last Clipped Actions
                }
            )

            return observationSpace

        case "O9":  # 7 Kanäle für CNN-DQN
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
            grid_size = int(self.slam.cropped_map_size_grid)

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
                        shape=(self.last_actions.shape[0],),
                        dtype=np.float32,
                    ),
                    "raycast": spaces.Box(low=0, high=9999, shape=(4,), dtype=np.float32),
                }
            )

            return observationSpace
