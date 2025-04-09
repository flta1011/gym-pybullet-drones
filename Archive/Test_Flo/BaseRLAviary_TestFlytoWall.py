import os
from collections import deque

import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_pybullet_drones.examples.Test_Flo.BaseAviary_TestFlytoWall import (
    BaseAviary_TestFlytoWall,
)
from gym_pybullet_drones.examples.Test_Flo.DSLPIDControl_TestFlo import DSLPIDControl
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ImageType,
    ObservationType,
    Physics,
)


class SimpleSlam:
    def __init__(self, map_size=100, resolution=0.1):
        """Initialize SLAM with an empty occupancy grid.

        Args:
            map_size (int): Size of the map in meters
            resolution (float): Size of each grid cell in meters
        """
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        # Initialize occupancy grid (-1: unknown, 0: free, 1: occupied)
        self.occupancy_grid = -1 * np.ones((self.grid_size, self.grid_size))
        # Center of the map
        self.center = self.grid_size // 2

        # Store drone's path
        self.path = []

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        grid_x = int(self.center + x / self.resolution)
        grid_y = int(self.center + y / self.resolution)
        return grid_x, grid_y

    def update(self, drone_pos, drone_yaw, raycast_results):
        """Update the map with new sensor readings.

        Args:
            drone_pos (tuple): (x, y, z) position of drone
            drone_yaw (float): Yaw angle in radians
            raycast_results (dict): Dictionary with raycast distances
                                  {'front': dist, 'back': dist, 'left': dist, 'right': dist}
        """
        x, y, _ = drone_pos
        grid_x, grid_y = self.world_to_grid(x, y)

        # Store drone's position
        self.path.append((grid_x, grid_y))

        # Mark current position as free
        self.occupancy_grid[grid_x, grid_y] = 0

        # Process each raycast
        angles = {"front": drone_yaw, "back": drone_yaw + np.pi, "left": drone_yaw + np.pi / 2, "right": drone_yaw - np.pi / 2}

        for direction, distance in raycast_results.items():
            if distance < 4:  # If ray hit something, add free and endpoint to the map
                angle = angles[direction]

                # Calculate end point of ray
                end_x = x + distance * np.cos(angle)
                end_y = y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)

                # Mark cells along ray as free
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                for cx, cy in cells[:-1]:  # alles bis auf den Endpunkt frei markieren
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                        self.occupancy_grid[cx, cy] = 0

                # Mark end point as occupied
                if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size:
                    self.occupancy_grid[end_grid_x, end_grid_y] = 1

            elif distance >= 4:  # Distanz ist auf 4 m gekappt, da das die Range des Sensors ist --> alles als frei markieren
                angle = angles[direction]

                # Calculate end point of ray
                end_x = x + distance * np.cos(angle)
                end_y = y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)

                # Mark cells along ray as free
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                for cx, cy in cells[:]:  # alles frei markieren
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                        self.occupancy_grid[cx, cy] = 0

    def get_line(self, x0, y0, x1, y1):
        """Get all grid cells along a line using Bresenham's algorithm."""
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
        """Visualize the occupancy grid."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.occupancy_grid.T, cmap="gray", origin="lower")

        # Plot drone's path
        if self.path:
            path = np.array(self.path)
            plt.plot(path[:, 0], path[:, 1], "r-", linewidth=2)
            plt.plot(path[-1, 0], path[-1, 1], "ro", markersize=10)  # Current position

        plt.colorbar(label="Occupancy (-1: unknown, 0: free, 1: occupied)")
        plt.title("SLAM Map")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")
        plt.show()

    def save_map(self, filename):
        """Save the occupancy grid and path to CSV files.

        Args:
            filename (str): Base filename to save the maps (without extension)
        """
        # Save occupancy grid
        np.savetxt(f"{filename}_grid.csv", self.occupancy_grid, delimiter=",")

        # Save path
        if self.path:
            path_array = np.array(self.path)
            np.savetxt(f"{filename}_path.csv", path_array, delimiter=",")

    def load_map(self, filename):
        """Load the occupancy grid and path from CSV files.

        Args:
            filename (str): Base filename to load the maps (without extension)
        """
        # Load occupancy grid
        self.occupancy_grid = np.loadtxt(f"{filename}_grid.csv", delimiter=",")

        # Load path if it exists
        try:
            path_array = np.loadtxt(f"{filename}_path.csv", delimiter=",")
            self.path = path_array.tolist()
        except:
            print("No path file found")

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        # In BaseAviary hinzugefügt
        #  Test_Area_Size_x: int = 10, #hoffentlich 10 Meter, später Größe der Map
        #  Test_Area_Size_y: int = 10, #hoffentlich 10 Meter, später Größe der Map
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
        reward_and_action_change_freq: int = 10,
        gui=False,
        user_debug_gui=False,
        record=False,
        act: ActionType = ActionType.VEL,
        advanced_status_plot=False,
    ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True
        and overridden with landmarks for vision applications;
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        self.reward_and_action_change_freq = reward_and_action_change_freq
        self.ACT_TYPE = act
        self.still_time = 0
        self.EPISODE_LEN_SEC = 5 * 60  # increased from 20 to 100

        #### Create integrated controllers #########################
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        else:
            print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            reward_and_action_change_freq=reward_and_action_change_freq,
            gui=gui,
            record=record,
            obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
            advanced_status_plot=advanced_status_plot,
            user_debug_gui=user_debug_gui,  # Remove of RPM sliders from all single agent learning aviaries
        )

        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

        self.slam = SimpleSlam(map_size=10, resolution=0.1)  # 10m x 10m map with 10cm resolution

    ################################################################################

    # def _addObstacles(self): # in BaseAviary_TestFlytoWall implementiert

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Discrete
        0: np.array([1, 0, 0, 0.99]), # Fly Forward
        1: np.array([-1, 0, 0, 0.99]), # Fly Backward
        2: np.array([0, 0, 0, 0.99]), # nothing

        """

        return spaces.Discrete(3)

    ################################################################################
    # ANCHOR - def preprocessAction
    def _preprocessAction(self, action):
        """Preprocesses the action from PPO to drone controls.
        Maps discrete actions to movement vectors.

        12.1.25:FT: gecheckt, ist gleich mit der Standard BaseRLAviary
        """
        # Convert action to movement vector
        # action_to_movement = {
        #     0: np.array([1, 0, 0, 0.99]),  # Forward
        #     1: np.array([-1, 0, 0, 0.99]), # Backward
        #     2: np.array([0, 0, 0, 0.99]),  # Stay
        # }

        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = action[k, :]
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(target_v[0:3]) != 0:
                v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=np.concatenate([state[0:1], np.array([0]), np.array([0.5])]),  # same as the current position on X, but should stay 0 on y and z = 0.5
                target_rpy=np.array([0, 0, 0]),  # keep orientation to base
                target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector,  # target the desired velocity vector
            )
            rpm[k, :] = temp
        return rpm

    ################################################################################

    def _observationSpace(self):
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

        #  # NOTE: wenn nicht das Alte Modell genutzt werden soll, das hier wieder auskommentieren
        # '''OLD MODELL mit 3D-Observation Rayfront, Rayback, LastAction'''
        # obs_lower_bound = np.array([0, 0, 0]) #Raycast reading forward
        # obs_upper_bound = np.array([4, 4, 2]) #Raycast reading forward, LastAction
        # return spaces.Box(
        #     low=obs_lower_bound,
        #     high=obs_upper_bound,
        #     dtype=np.float32
        #     )

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([0])  # Raycast reading forward

        obs_upper_bound = np.array([4])  # Raycast reading forward

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([0])  # Raycast reading forward

        obs_upper_bound = np.array([4])  # Raycast reading forward

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################
    # ANCHOR - computeObs
    def _computeObs(self):
        """Returns the current observation of the environment.
        10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

            Information of the self._getDroneStateVector:
                ndarray
                The state vector includes:

                1x actual raycast reading (forward)  [21]    -> 0 bis 4
        """

        state = self._getDroneStateVector(0)

        # # NOTE: wenn nicht das Alte Modell genutzt werden soll, das hier wieder auskommentieren
        # '''OLD MODELL mit 3D-Observation Rayfront, Rayback, LastAction'''
        # obs_9 = np.concatenate([
        #     state[21:23],  # actual raycast readings (forward,backward)
        #     [state[26]]   # last  action (Velocity in X-Richtung!)
        # ])
        # return obs_9
        #     ############################################################

        state = self._getDroneStateVector(0)

        # Select specific values from obs and concatenate them directly
        obs = [state[21]]  # Raycast reading forward
        return obs

        # '''für mehrere Drohnen'''
        # obs_25 = np.zeros((self.NUM_DRONES,25))
        # for i in range(self.NUM_DRONES):
        #     #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
        #     obs = self._getDroneStateVector(i)
        #     obs_25[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:20], obs[20:25]]).reshape(21,)
        #     ret = np.array([obs_25[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        # #### Add action buffer to observation #######################
        # for i in range(self.ACTION_BUFFER_SIZE):
        #     ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        # return ret
        #     ############################################################
        # else:
        #     print("[ERROR] in BaseRLAviary._computeObs()")

    ################################################################################

    # ANCHOR - computeReward
    def _computeReward(self):  # Funktioniert und die Drohne lernt, nahe an die Wand, aber nicht an die Wand zu fliegen. Problem: die Drohne bleibt nicht sauber im Sweetspot stehen.
        """Computes the current reward value.

        # _Backup_20250211_V1_with_mixed_reward_cases
        Returns
        -------
        float
            The reward.

        """
        if "lastaction" not in locals():
            lastaction = self.action

        reward = 0
        state = self._getDroneStateVector(0)  # erste Drohne

        self.Ende_Crash = 0.2
        self.Beginn_sweetspot = 0.4
        self.Ende_sweetspot = 0.5

        #####VOR DEM SWEETSPOT#############
        # wenn vorheringer Raycastreading = Actual Raycastreading = 4, dann abstand zu groß -> Vx > 0 (vorne fliegen ist gut, rückwärts fliegen ist schlecht)
        # self.action[0][0] ist der Wert der Velocity in SOLL: x-Richtung
        if self.action[0][0] == 1 and state[21] > self.Beginn_sweetspot:
            reward = 2  # von 2 auf 10 erhöht, dass
        elif self.action[0][0] == -1 and state[21] > self.Beginn_sweetspot:
            reward = (
                -2
            )  # von -5 auf -3 reduziert (9.2), von -3 auf -7 reduziert (10.2), damit Reward hacking entgegengewirkt wird (+/- immer abwechselnd), wieder zurück auf -5 (10.2), damit er beim fliegen nach vorne (was gut ist) nicht unnötig bestraft wird.

        #####Im SWEETSPOT######################
        # stillstand und im sweetspot: Belohnung

        if self.action[0][0] == 0 and state[21] > self.Ende_sweetspot and state[21] < self.Beginn_sweetspot:
            reward = 50
        # vorwärts fliegen und im sweetspot: Neutral
        elif self.action[0][0] == 1 and state[21] > self.Beginn_sweetspot and state[21] < self.Ende_sweetspot:
            reward = 0
        # Rückwärts im Sweetspot: Neutral
        elif self.action[0][0] == -1 and state[21] > self.Beginn_sweetspot and state[21] < self.Ende_sweetspot:
            reward = 0

        #####NACH DEM SWEETSPOT, zu nah an der Wand#####################
        elif state[21] < self.Beginn_sweetspot and state[21] > self.Ende_Crash:
            reward = -5

        ##############Gecrasht, aka zu nah dran########################
        elif state[21] <= self.Ende_Crash:
            reward = (
                -300
            )  # reward von -1000 auf -300 verringert, da die Drohne sonst nicht mehr lernt bzw. durch den Zusammenprall insgesamt negatives gesamtergebnis bekommt und dann ableitet, dass alles schlecht war und dann danach nur noch stehenbleibt

        # Belohnung, wenn der Abstand der Actionen 1 und nicht 2 beträgt
        if abs(self.action[0][0] - lastaction[0][0]) == 1:
            reward += 5
        elif abs(self.action[0][0] - lastaction[0][0]) == 2:
            reward += -2

        # nachdem der Unterschied verwendet wurde, nun die letzte Action mit der neusten Action überschreiben
        lastaction = self.action

        return reward

    def _computeReward_V2_Backup_20250211_1300_Squared_Reward_Function(self):
        """Ergebnis: nicht sehr gut.. und Rewardergebnisse sind nicht wirklich predictbar!!

        # V2_Backup_20250211_1300_Squared_Reward_Function

        Computes the current reward value based on distance to wall and actions.

        Returns
        -------
        float
            The reward based on the distance-reward function from the image.
        """
        state = self._getDroneStateVector(0)
        distance = state[21]  # Forward raycast distance

        # Constants
        self.DISTANCE_SENSOR_THRESHOLD = 4.0
        self.SWEET_SPOT_DISTANCE = 0.4  # Center of sweet spot
        self.SWEET_SPOT_MAX_REWARD = 3
        self.MIN_REWARD_LINEAR_DECLINE_TO_WALL = -6
        self.PUNISHMENT_WALLHIT = 3000  # muss positiv sein, da es unten abgezogen wird
        self.CRASH_DISTANCE = 0.2
        self.REWARD_FORWARD = 0.1
        self.REWARD_BACKWARD = -0.1

        # If distance is greater than threshold (4 in practice)
        if distance > self.DISTANCE_SENSOR_THRESHOLD:
            if self.action[0][0] == 1:  # Forward motion
                return self.REWARD_FORWARD
            elif self.action[0][0] == -1:  # Backward motion
                return self.REWARD_BACKWARD
            elif self.action[0][0] == 0:
                return 0

        # Distance-based reward function for distance <= 4
        if distance <= self.DISTANCE_SENSOR_THRESHOLD:
            if distance <= self.SWEET_SPOT_DISTANCE:
                # Linear decline from sweet spot to wall (0 distance)
                slope = (self.SWEET_SPOT_MAX_REWARD + self.MIN_REWARD_LINEAR_DECLINE_TO_WALL) / (self.SWEET_SPOT_DISTANCE)
                return self.SWEET_SPOT_MAX_REWARD + slope * (distance)

            else:  # distance > SWEET_SPOT_DISTANCE
                # Exponential increase from threshold to sweet spot
                normalized_dist = (self.DISTANCE_SENSOR_THRESHOLD - distance) / (self.DISTANCE_SENSOR_THRESHOLD - self.SWEET_SPOT_DISTANCE)
                return self.SWEET_SPOT_MAX_REWARD * (normalized_dist**2)

        return 0

    def _computeReward_V3_BACKUP_20250211_Discrete_Rewards(self):
        """
        #_V3_BACKUP_20250211_Discrete_Rewards: Discretized Reward-Values: depeding on the distance an Action: konkret Reward!

        Computes the current reward value based on distance to wall and actions.

        Returns
        -------
        float
            The reward based on the distance-reward function from the image.
        """
        # Constants
        self.DISTANCE_SENSOR_THRESHOLD = 4.0
        self.SWEET_SPOT_DISTANCE = 0.4  # Center of sweet spot
        self.SWEET_SPOT_DISTANCE_TOLERANCE = 0.03
        self.CRASH_DISTANCE = 0.2

        state = self._getDroneStateVector(0)
        distance = state[21]  # Forward raycast distance
        last_action = state[26]

        # starte einen Timer, wenn die Drohne im sweet spot ist
        if abs(state[21] - self.SWEET_SPOT_DISTANCE) < 0.03 and state[26] == 0:
            self.still_time += 1 / self.reward_and_action_change_freq  # Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
        else:
            self.still_time = 0.0  # Reset timer to 0 seconds

        # Ziel erreicht (5s im Sweetspot)
        if self.still_time >= 5:
            return 5000

        # NOTE: nicht mehr nötig, da die Drohne sich nicht mehr an die Wand traut und den Sweetspot nicht mehr entdeckt
        if distance < self.CRASH_DISTANCE:
            return (
                -150
            )  # von -300 auf -150 erhöht, damit die Strafe niedriger ist und die Drohne sich noch mehr an die Wand traut und den Sweetspot entdeckt, sonst scheint es eine lokales Minimum bei ca 0,9 m zu geben

        # zu nah an der Wand
        if distance < self.SWEET_SPOT_DISTANCE - self.SWEET_SPOT_DISTANCE_TOLERANCE:
            if last_action == 1:
                return -1  # Bestrafung für vorne muss leicht größer sein, damit ein Fliegen zu nah bestraft ist und sich ins Gedächtnis brennt
            elif last_action == -1:
                return +0.5
            elif last_action == 0:
                return -0.1

        # Distance innerhalb der Sweetspot-Range
        if abs(distance - self.SWEET_SPOT_DISTANCE) <= self.SWEET_SPOT_DISTANCE_TOLERANCE:
            # Belohnung fürs erreichen des Ziels
            if last_action == 1:
                return -0.25
            elif last_action == -1:
                return -0.25
            elif last_action == 0:
                return +10

        # zu weit entfernt
        if distance > self.SWEET_SPOT_DISTANCE + self.SWEET_SPOT_DISTANCE_TOLERANCE:
            if last_action == 1:
                return 0.2
            elif last_action == -1:
                return -0.2
            elif last_action == 0:
                return -0.02

        print("Distanz, die Probleme macht:", distance)

        return 0

    ################################################################################

    def _computeTerminated(self):
        terminated, reason = super()._computeTerminated()
        if terminated:
            # Save final map
            self.slam.save_map(f"slam_map_final_{self.step_counter}")
        return terminated, reason

    ################################################################################

    def _computeTruncated(self):  # coppied from HoverAviary_TestFlo.py
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the drone is too tilted or has crashed into a wall.

        """
        # Truncate when the drone is too tilted
        state = self._getDroneStateVector(0)
        if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
            Grund_Truncated = "Zu tilted"
            return True, Grund_Truncated

        # TBD wenn die Drone abstürzt, dann auch truncaten
        if state[2] < 0.1:  # state[2] ist z_position der Drohne
            Grund_Truncated = "Crash"
            return True, Grund_Truncated

        # Wenn an einer Wand gecrashed wird, beenden!
        if state[21] <= 0.2 or state[22] <= 0.2 or state[23] <= 0.2 or state[24] <= 0.2:
            Grund_Truncated = "Zu nah an der Wand"
            return True, Grund_Truncated

        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            Grund_Truncated = "Zeit abgelaufen"
            return True, Grund_Truncated

        Grund_Truncated = None

        return False, Grund_Truncated

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.



        """
        # state = self._getDroneStateVector(0) #getDroneStateVector braucht die 0

        # print("Reward:", self.reward_buffer)
        # #Plotting infos zum Zeitpunkt der Episode, Raycasts(vorne) der Drohne und Geschwindigkeiten der Drohne
        # print("Abstand zur Wand:", state[21])
        # print("Linear velocity Vx:", state[10])
        # #print raycasts
        # print("Raycast vorne:", state[21])

        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

    #########################################################################################

    def step(self, action):
        # Your existing step code...

        # Update SLAM
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        yaw = state[9]  # Assuming this is the yaw angle

        # Get raycast results
        sensor_readings = self.check_distance_sensors(0)
        raycast_results = {"front": sensor_readings[0], "back": sensor_readings[1], "left": sensor_readings[2], "right": sensor_readings[3]}

        self.slam.update(pos, yaw, raycast_results)

        # Save map every 1000 steps
        if self.step_counter % 1000 == 0:
            self.slam.save_map(f"slam_map_{self.step_counter}")

        # Visualize every 100 steps
        if self.step_counter % 100 == 0:
            self.slam.visualize()

        # Your existing step code...

        return self.step_counter

    #########################################################################################
