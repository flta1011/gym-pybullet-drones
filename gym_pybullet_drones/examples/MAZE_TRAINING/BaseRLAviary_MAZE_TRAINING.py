import contextlib
import csv
import heapq
import logging
import os
import socket
import sys
import time
import webbrowser  # Add this import
from collections import deque
from threading import Thread

import dash
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pybullet as p
from dash import dcc, html
from dash.dependencies import Input, Output
from gymnasium import spaces
from plotly.subplots import make_subplots
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_pybullet_drones.examples.MAZE_TRAINING.BaseAviary_MAZE_TRAINING import (
    BaseAviary_MAZE_TRAINING,
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
    def __init__(self, map_size=8, resolution=0.05):  # map size 8x8m, damit, egal in welche Richtung die Drohne fliegt, in jeden Quadranten ein komplettes Labyrinth dargestellt werden kann
        """Erstellt eine leere Occupancy-Grid Map. Args: map_size (float): Seitengröße der Map in Metern (z. B. 8 m). resolution (float): Seitengröße einer Zelle, sodass grid_size ~60 ergibt."""
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        # Initialisiere die Map:
        # # -1: unbekannt, 0: frei, 1: Wand, 2: besucht (Sensor oben frei)
        # # 0.2: unbekannt, 0.9: frei, 0.0: Wand, 0.5: besucht (Sensor oben frei)

        self.occupancy_grid = 0.2 * np.ones((self.grid_size, self.grid_size))
        self.center = self.grid_size // 2
        self.path = []  # speichert besuchte Zellen
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []
        self.counter_free_space = 0

    def reset(self):
        """Reset the SLAM map to its initial state."""
        self.occupancy_grid = 0.2 * np.ones((self.grid_size, self.grid_size))
        self.path = []
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []

    def world_to_grid(self, x, y):
        grid_x = int(self.center + x / self.resolution)
        grid_y = int(self.center + y / self.resolution)
        return grid_x, grid_y

    def update(self, drone_pos, drone_yaw, raycast_results):
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
        self.occupancy_grid[grid_x, grid_y] = 0.5
        self.grid_x = grid_x
        self.grid_y = grid_y
        # self.counter_free_space += 1
        # Falls der "up"-Sensor keinen Treffer hat (z. B. Wert 9999), markiere als besucht:
        if "up" in raycast_results and raycast_results["up"] == 9999:
            self.occupancy_grid[grid_x, grid_y] = 0.5

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
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size and self.occupancy_grid[cx, cy] != 0.0:
                        self.occupancy_grid[cx, cy] = 0.9
                        self.counter_free_space += 1
                # Markiere den Endpunkt als Wand:
                if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size and self.occupancy_grid[end_grid_x, end_grid_y] != 0.9:
                    self.occupancy_grid[end_grid_x, end_grid_y] = 0.0

        self.Prev_DrohnePosition = self.DrohnePosition
        self.DrohnePosition = []

        for i in range(max(0, grid_x - 2), min(160, grid_x + 2)):
            for j in range(max(0, grid_y - 2), min(160, grid_y + 2)):
                self.path.append((i, j))
                self.DrohnePosition.append((i, j))
                if self.occupancy_grid[i, j] == 0.2:
                    self.occupancy_grid[i, j] = 0.5
                elif self.occupancy_grid[i, j] == 0.5:
                    self.occupancy_grid[i, j] = 0.7

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
        plt.ioff()  # Turn off interactive mode
        plt.figure(figsize=(6, 6))
        plt.imshow(self.occupancy_grid.T, cmap="gray", origin="lower")

        if self.Prev_DrohnePosition:
            prev_position = np.array(self.Prev_DrohnePosition)
            plt.plot(prev_position[:, 0], prev_position[:, 1], "r-", linewidth=2)
            plt.plot(prev_position[-1, 0], prev_position[-1, 1], "ro", markersize=5)

        if self.path:
            path = np.array(self.path)
            plt.plot(path[:, 0], path[:, 1], "r-", linewidth=2)
            plt.plot(path[-1, 0], path[-1, 1], "ro", markersize=5)

        if self.DrohnePosition:
            position = np.array(self.DrohnePosition)
            plt.plot(position[:, 0], position[:, 1], "g-", linewidth=2)
            # plt.plot(position[-1, 0], position[-1, 1], 'go', markersize=5)

        plt.colorbar(label="Occupancy (-1: unbekannt, 0: frei, 1: Wand, 2: besucht)")
        plt.title("SLAM Map")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")
        self.OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "output_SLAM_MAP")
        # create output folder if it doesn't exist
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

        # Save plot to file with current date and time
        # current_time = time.strftime("%Y%m%d-%H%M%S")
        # self.Latest_slam_map_path = os.path.join(self.OUTPUT_FOLDER, f"slam_map_{current_time}.png")
        # plt.savefig(self.Latest_slam_map_path)
        plt.close()
        plt.ion()  # Turn interactive mode back on


class BaseRLAviary_MAZE_TRAINING(BaseAviary_MAZE_TRAINING):
    """Base single and multi-agent environment class for reinforcement learning."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
        reward_and_action_change_freq: int = 10,
        gui=False,
        user_debug_gui=False,
        record=False,
        act: ActionType = ActionType.VEL,
        advanced_status_plot=False,
        target_position=np.array([0, 0, 0]),
        Danger_Threshold_Wall=0.15,
        EPISODE_LEN_SEC=10 * 60,
        map_size_slam=8,  # map size 8x8m, damit, egal in welche Richtung die Drohne fliegt, in jeden Quadranten ein komplettes Labyrinth dargestellt werden kann
        resolution_slam=0.05,
        Dash_active=True,
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
        # NOTE - Episoden länge
        self.EPISODE_LEN_SEC = EPISODE_LEN_SEC  # increased from 5 auf 20 Minuten um mehr zu sehen (4.3.25)
        self.TARGET_POSITION = target_position
        self.Danger_Threshold_Wall = Danger_Threshold_Wall
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys
        self.port = 8051
        self.reward_components = {"collision_penalty": 0, "distance_reward": 0, "explore_bonus_new_field": 0, "explore_bonus_visited_field": 0, "Target_Hit_Reward": 0}
        self.Dash_active = Dash_active

        # Historie der Reward-Komponenten für Balkendiagramm
        self.reward_distribution_history = deque(maxlen=50)  # speichere 50 Einträge

        # Initialize reward and best_way map
        self.reward_map = np.zeros((60, 60), dtype=int)
        self.best_way_map = np.zeros((60, 60), dtype=int)

        # Counter for the amount of wall pixel in map
        self.wall_pixel_counter = 0
        self.amount_of_pixel_in_map = 60 * 60
        self.ratio_previous_step = 0
        # self.ratio_current_step = 0
        self.amount_of_pixel_in_map_without_walls = 0
        self.distance_10_step_ago = 0
        self.distance_50_step_ago = 0
        self.differnece_threshold = 0.05

        # Initialize SLAM before calling the parent constructor
        self.slam = SimpleSlam(map_size=map_size_slam, resolution=resolution_slam)  # 10m x 10m map with 10cm resolution
        self.grid_size = int(map_size_slam / resolution_slam)
        # Call the parent class constructor
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
            target_position=target_position,
            Danger_Threshold_Wall=Danger_Threshold_Wall,
        )

        #### Create integrated controllers #########################
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        else:
            print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")

        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

        if self.Dash_active:
            # ANCHOR - Create DASH Graph
            self.app = dash.Dash(__name__)
            self.app.layout = html.Div(
                [
                    dcc.Graph(id="live-map"),
                    dcc.Graph(id="observation-channels"),
                    dcc.Graph(id="reward-bar-chart"),
                    html.Div(id="current-total-reward"),
                    dcc.Interval(id="interval-component", interval=200, n_intervals=0),
                ]
            )

            @self.app.callback(
                [Output("live-map", "figure"), Output("observation-channels", "figure"), Output("reward-bar-chart", "figure"), Output("current-total-reward", "children")],
                [Input("interval-component", "n_intervals")],
            )
            def update_graph(n):
                # Create reward/best way map figure
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Reward Map", "Best Way Map"))
                fig.add_trace(go.Heatmap(z=self.reward_map, colorscale="Viridis", showscale=True, name="Reward Map"), row=1, col=1)
                fig.add_trace(go.Heatmap(z=self.best_way_map, colorscale="Viridis", showscale=True, name="Best Way Map"), row=1, col=2)
                fig.update_layout(height=600, title_text="Maze Training Visualization", showlegend=True)

                # Get current observation channels
                obs = self._computeObs()

                # Create observation channels figure with SLAM map on left and values on right
                obs_fig = make_subplots(
                    rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("Normalized SLAM Map", "Legend & Values"), specs=[[{"type": "heatmap"}, {"type": "table"}]]  # Make columns equal width
                )

                # Add SLAM map heatmap
                obs_fig.add_trace(
                    go.Heatmap(
                        z=obs[0],
                        colorscale=[[0, "rgb(0,0,0)"], [0.2, "rgb(128,128,128)"], [0.5, "rgb(255,165,0)"], [0.9, "rgb(255,255,255)"]],  # Wall (0.0)  # Unknown (0.2)  # Visited (0.5)  # Free (0.9)
                        showscale=False,
                        name="SLAM Map",
                    ),
                    row=1,
                    col=1,
                )

                # Create table with legend and observation values
                obs_fig.add_trace(
                    go.Table(
                        header=dict(values=["Type", "Description"]),
                        cells=dict(
                            values=[
                                ["Wall", "Unknown", "Visited", "Free", "", "Position X", "Position Y", "sin(yaw)", "cos(yaw)"],
                                ["Black (0.0)", "Gray (0.2)", "Orange (0.5)", "White (0.9)", "", f"{obs[1][0][0]:.3f}", f"{obs[2][0][0]:.3f}", f"{obs[3][0][0]:.3f}", f"{obs[4][0][0]:.3f}"],
                            ]
                        ),
                    ),
                    row=1,
                    col=2,
                )

                obs_fig.update_layout(height=600, title_text="Observation Channels", showlegend=False)  # Make it square

                # Create reward components bar chart
                bar_chart = go.Figure(go.Bar(x=list(self.reward_components.keys()), y=list(self.reward_components.values()), marker_color="royalblue"))
                bar_chart.update_layout(title_text="Current Reward Components", xaxis_title="Reward Type", yaxis_title="Reward Value")

                # Initialize last_total_reward if not set
                if not hasattr(self, "last_total_reward"):
                    self.last_total_reward = 0

                current_reward_text = f"Last Reward: {self.last_total_reward:.2f}"

                return fig, obs_fig, bar_chart, current_reward_text

            def is_port_in_use(port):
                with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    return s.connect_ex(("localhost", port)) == 0

            # Start Dash server in background thread
            def run_dash_app():
                logging.getLogger("werkzeug").setLevel(logging.ERROR)
                self.app.run_server(debug=False, port=self.port)

            if not is_port_in_use(self.port):
                self.dashboard_thread = Thread(target=run_dash_app, daemon=True)
                self.dashboard_thread.start()

                # Open web browser after a short delay to ensure server is running
                time.sleep(1)  # Wait for server to start
                webbrowser.open(f"http://localhost:{self.port}")
            else:
                print(f"Port {self.port} is already in use, cannot start Dash server.")

    ################################################################################

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Discrete
        DREHUNG MATHEMATISCH POSITIV (GEGEN DEN UHRZEIGER)
        1: np.array([[0, 0, 0, 0.99, 0]]), # Fly 0° (Stay)
        2: np.array([[1, 0, 0, 0.99, 0]]), # Fly 90° (Forward)
        3: np.array([[-1, 0, 0, 0.99, 0]]), # Fly 180° (Backward)
        4: np.array([[0, 1, 0, 0.99, 0]]), # Fly 90° (Left)
        5: np.array([[0, -1, 0, 0.99, 0]]), # Fly 270° (Right)
        6: np.array([[0, 0, 0, 0.99, 1/4*np.pi]]), # 45° Left-Turn
        7: np.array([[0, 0, 0, 0.99, -1/4*np.pi]]), # 45° Right-Turn




        """

        return spaces.Discrete(7)

    ################################################################################
    # ANCHOR - def preprocessAction
    def _preprocessAction(self, action):
        """Preprocesses the action from PPO to drone controls.
        Maps discrete actions to movement vectors.

        12.1.25:FT: gecheckt, ist gleich mit der Standard BaseRLAviary
        """
        # Convert action to movement vector
        # action_to_movement = {
        #     0: np.array([[1, 0, 0, 0.99, 0]]),  # Forward
        #     1: np.array([[-1, 0, 0, 0.99, 0]]), # Backward
        #     2: np.array([[0, 0, 0, 0.99, 0]]),  # Stay
        # }

        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = action[k, :4]
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(target_v[0:3]) != 0:
                v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
            else:
                v_unit_vector = np.zeros(3)

            # NOTE - neu hinzueefügt, dass die Drohne sich auch drehen kann
            current_yaw = state[9]
            change_value_yaw = action[k, 4]
            Calculate_new_yaw = current_yaw + change_value_yaw

            temp, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=np.array([state[0], state[1], 0.5]),  # same as the current position on X, and same on y (not as in fly to wall scenario) and z = 0.5
                target_rpy=np.array([0, 0, Calculate_new_yaw]),  # neue Yaw-Werte durch Drehung der Drohne
                target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector,  # target the desired velocity vector
            )
            rpm[k, :] = temp
        return rpm

    ################################################################################

    def _observationSpace_Backup(self):
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

        #  # NOTE: wenn nicht das Alte Modell genutzt werden soll, das hier wieder auskommentieren
        # '''OLD MODELL mit 3D-Observation Rayfront, Rayback, LastAction'''
        # obs_lower_bound = np.array([0, 0, 0]) #Raycast reading forward
        # obs_upper_bound = np.array([9999, 9999, 2]) #Raycast reading forward, LastAction
        # return spaces.Box(
        #     low=obs_lower_bound,
        #     high=obs_upper_bound,
        #     dtype=np.float32
        #     )

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([-99, -99, -2 * np.pi, 0, 0, 0, 0, 0])  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

        obs_upper_bound = np.array(
            [99, 99, 2 * np.pi, 9999, 9999, 9999, 9999, 9999]
        )  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([0])  # Raycast reading forward

        obs_upper_bound = np.array([9999])  # Raycast reading forward

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ###################################################################################
    # ANCHOR - OBSERVATIONSPACE Für CNN-DQN
    def _observationSpace(self):
        """
        Returns the observation space for the CNN-DQN model.
        The observation space is a Box with shape (5, grid_size, grid_size) containing:
        - Channel 1: Normalized SLAM map (values in [0,1])
        - Channel 2: Normalized x position (values in [0,1])
        - Channel 3: Normalized y position (values in [0,1])
        - Channel 4: sin(yaw) (values in [-1,1])
        - Channel 5: cos(yaw) (values in [-1,1])
        """
        grid_size = self.slam.grid_size

        # Create proper shaped arrays for low and high bounds
        low = np.zeros((5, grid_size, grid_size), dtype=np.float32)
        high = np.ones((5, grid_size, grid_size), dtype=np.float32)

        # Set specific ranges for each channel
        low[3, :, :] = -1.0  # sin(yaw) lower bound
        low[4, :, :] = -1.0  # cos(yaw) lower bound

        return spaces.Box(low=low, high=high, dtype=np.float32)

    ###################################################################################
    # ANCHOR compute_OBS_NORM_MAP für CNN-DQN
    def _computeObs(self):
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

    ################################################################################
    # ANCHOR - computeObs_Backup
    def _computeObs_Backup(self):
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

        state = self._getDroneStateVector(0)

        # # NOTE: wenn nicht das Alte Modell genutzt werden soll, das hier wieder auskommentieren
        # '''OLD MODELL mit 3D-Observation Rayfront, Rayback, LastAction'''
        # obs_9 = np.concatenate([
        #     state[21:23],  # actual raycast readings (forward,backward)
        #     [state[26]]   # last  action (Velocity in X-Richtung!)
        # ])
        # return obs_9

        state = self._getDroneStateVector(0)

        # Select specific values from obs and concatenate them directly
        obs = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

        # NOTE - nachfolgend auf vereinfachte Observation Space umgestellt (28.2.25):
        # Modify observation based on distance thresholds
        modified_obs = []

        # NOTE - Distanz zum Übergeben hat gar nicht so viel gebracht, weil es sich langfristig nicht stabilieren konnte (28.2.25)
        # drone_pos = state[0:3]  # XYZ position from state vector
        # target_pos = self.TARGET_POSITION
        # # Calculate distance to target
        # distance = np.linalg.norm(drone_pos - target_pos)
        # modified_obs.append(distance)

        # NOTE - neue Tests mit X,Y, Yaw Position der Drohne (28.2.25) übergeben
        modified_obs.append(round(state[0], 3))  # x-Position
        modified_obs.append(round(state[1], 3))  # y-Position
        modified_obs.append(round(state[9], 3))  # Yaw-Position

        # abstände anhängen mit 3 Nachkommastellen
        for distance in obs:
            modified_obs.append(round(distance, 3))

        # for distance in obs:
        #     if distance <= (self.Danger_Threshold_Wall):  # Too close to wall, Safetyalgorithmus wird gegensteuern
        #         modified_obs.append(0)
        #     elif distance <= (self.Danger_Threshold_Wall+0.1):  # Vorwarnung: gleich passiert was
        #         modified_obs.append(1)
        #     else:  # Safe distance
        #         modified_obs.append(2)

        # raycast oben noch anhängen
        if state[25] < 1:
            modified_obs.append(1)
        else:
            modified_obs.append(9999)

        return np.array(modified_obs, dtype=np.float32)  # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,9999)

        ############################################################

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

    ##############################################################################!SECTION

    def _compute_potential_fields(self):
        # Parameter
        state = self._getDroneStateVector(0)
        
        k_rep = 0.01  # Repulsion-Skalierun
        d0 = 0.4       # Einflussradius für Wände
        Scale_Grid = 0.05


        # Erstelle ein Raster mit Potentialwerten
        potential_map = np.zeros_like(self.reward_map, dtype=float)

        # Extrahiere Wandpositionen (Indizes der Wandpositionen)
        walls = np.argwhere(self.reward_map == 6)

        # Berechne Potentialfeld für jedes Pixel im Grid
        for x in range(potential_map.shape[0]):
            for y in range(potential_map.shape[1]):
                pos = np.array([x, y])

                # Abstoßungs-Potential (von Wänden)
                U_rep = 0
                for wall in walls:
                    d = np.linalg.norm(pos - wall)*Scale_Grid
                    if 0 < d < d0:
                        U_rep += k_rep * (1/d - 1/d0) ** 2

                potential_map[x, y] = U_rep
                
        # Visualisiere das Potentialfeld
        # Create output folder if it doesn't exist
        output_folder = os.path.join(os.path.dirname(__file__), 'potenzial_fields')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Create and save the plot without displaying
        plt.ioff() # Turn off interactive mode
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(potential_map, cmap='viridis', origin='lower')
        plt.colorbar(label='Potential')
        plt.title('Potentialfeld')
        plt.xlabel('x')
        
        # Generate timestamp and save
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(output_folder, f'potential_field_{timestamp}.png'))
        plt.close()
        
                
        return potential_map
                    
################################################################################
    def _initialize_Reward_Map_and_Best_Way_Map(self, Maze_Number):
        """Initializes the reward map and the best way map.
        uses potential fields to find the best way from start to goal.
        """

        # NOTE - Reward Map

        # 0 = Unbesucht,
        # 1 = Einmal besucht,
        # 2 = Zweimal besucht,
        # 3 = Dreimal besucht,
        # 4 = Startpunkt,
        # 5 = Zielpunkt,
        # 6 = Wand

        # Initializing Reward Map
        self.reward_map = np.zeros((60, 60), dtype=int)
        # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
        reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_0.csv"

        with open(reward_map_file_path, "r") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                for j, value in enumerate(row):
                    if value == "1":
                        self.reward_map[j, i] = 6  # Wand
                        self.wall_pixel_counter += 1

        # with open(reward_map_file_path, 'r') as file:
        #     reader = csv.reader(file)
        #     for i, row in enumerate(reader):
        #         for j, value in enumerate(row):
        #             if value == "1":
        #                 self.reward_map[j, i] = 6 # Wand

        # Mirror the reward map vertically
        # self.reward_map = np.flipud(self.reward_map)
        # Rotate the reward map 90° mathematically negative
        # self.reward_map = np.rot90(self.reward_map, k=4)

        # Amount of pixel in the map without walls
        self.amount_of_pixel_in_map_without_walls = self.amount_of_pixel_in_map - self.wall_pixel_counter

        # Save the best way map to a CSV file
        # self._compute_potential_fields([0, 0], only_forces=False)  # aktuell noch buggy, Übergabe [0, 0]
        with open("best_way_map_DQN.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.best_way_map)

        # Save the reward map to a CSV file
        with open("reward_map_DQN.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.reward_map)

    ######################################################################################

    # ANCHOR - computeReward
    def _computeReward(
        self, Maze_Number, random_number_Start, random_number_Target
    ):  # Funktioniert und die Drohne lernt, nahe an die Wand, aber nicht an die Wand zu fliegen. Problem: die Drohne bleibt nicht sauber im Sweetspot stehen.
        """Computes the current reward value.

        18.3.25:
        - Best way reward entfernt
        - Target reward entfernt
        - Collision reward entfernt
        - Distance reward entfernt
        - Explore bonus new field reward entfernt
        - Explore bonus visited field reward entfernt
        Bleibt nur die Belohnung für das Erkunden neuer Felder.(5x5 grid um die Drohne herum)
        Returns
        -------
        float
            The reward.

        """
        
        
        
        
        if self.step_counter == 0:
            self._initialize_Reward_Map_and_Best_Way_Map(Maze_Number)
    
    
    
        Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
        End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

        # print (Start_position, "Start-REward")
        # print (End_Position, "Target-reward")
        # Set the Startpoint of the Drone
        initial_position = [Start_position[1] / 0.05, Start_position[0] / 0.05]  # Startpunkt der Drohne
        self.reward_map[int(initial_position[0]), int(initial_position[1])] = 4  # Startpunkt

        # Set the Targetpoint of the Drone
        target_position = [End_Position[1]/0.05, End_Position[0]/0.05] # Zielpunkt der Drohne
        self.reward_map[int(target_position[0]), int(target_position[1])] = 5 # Zielpunkt
            
         
            
            
            
            
       

        # Save the reward map to a CSV file
        with open('reward_map.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.reward_map)
            
        #NOTE Test mit Potentialfeld-Plot (wird aber im Code noch nicht benutzt)
        # if self.step_counter == 0:
        #     potential_map = self._compute_potential_fields()
    
        reward = 0
        state = self._getDroneStateVector(0) #erste Drohne


            
        #### Rewards initialisieren ####
        self.reward_components["collision_penalty"] = 0
        self.reward_components["distance_reward"] = 0
        #self.reward_components["best_way_bonus"] = 0
        self.reward_components["explore_bonus_new_field"] = 0
        self.reward_components["explore_bonus_visited_field"] = 0
        self.reward_components["Target_Hit_Reward"] = 0
        
        ###### 1.PUNISHMENT FOR COLLISION ######
        if self.action_change_because_of_Collision_Danger == True:
            self.reward_components["collision_penalty"] = -1.0

        # NOTE - 18.3: Ziel hat keinen Einfluss mehr, soll aufs erkunden belohnt werden
        # ###### 2.REWARD FOR DISTANCE TO TARGET (line of sight) ######
        # # Get current drone position and target position # TODO STIMMT DER TARGET POSITION?

        # drone_pos = state[0:2]  # XY position from state vector
        # target_pos = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target][0:2] # XY position of the target
        # Target_Value_1 = target_pos[0]
        # Target_Value_2 = target_pos[1]
        # target_pos = [Target_Value_2, Target_Value_1] # X und Y Werte vertauschen, weil die Drohne die Werte vertauscht
        # # print(drone_pos, "Drone Position")s
        # # print(target_pos, "Target Position")
        # # Calculate distance to target
        # self.distance = np.linalg.norm(drone_pos - target_pos)

        # # print(self.distance, "Distance")
        # # print(drone_pos, "Drone Position")
        # # print(target_pos, "Target Position")
        
        # # Define max distance and max reward
        # MAX_DISTANCE = 3.0  # Maximum expected distance in meters
        # MAX_REWARD = 0.5    # Maximum reward for distance (excluding target hit bonus)
        
        # # Linear reward that scales from 0 (at MAX_DISTANCE) to MAX_REWARD (at distance=0)
        # distance_ratio = min(self.distance/MAX_DISTANCE, 1.0)
        # self.reward_components["distance_reward"] = MAX_REWARD * (1 - distance_ratio) ## 4.3.25: auf Linear umgestellt, damit auch in weiter entfernten Feldern noch ein Gradient erkannt werden kann
        
        # # Add huge reward if target is hit (within 0.05m) and top sensor shows no obstacle
        # if self.distance < 0.15 and state[25] < 1: # 0.15 = Radius Scheibe
        #     self.reward_components["Target_Hit_Reward"] += 1000.0
        #     print(f"Target hit. Zeitstempel (min:sek) {time.strftime('%M:%S', time.localtime())}")
        
        # Get current position
        current_position = [int(state[0]/0.05), int(state[1]/0.05)]
        
        
        ###### 3. REWARD FOR BEING ON THE BEST WAY ######
        # Get the current position of the drone
        
        # Check if the drone is on the best way
        # if self.best_way_map[current_position[0], current_position[1]] == 1:
        #     self.reward_components["best_way_bonus"] = 10
        
        
        
        ###### 4. REWARD FOR EXPLORING NEW AREAS ######
        # Vereinfachung 18.3: 5x5 grid um die Drohne herum
        x, y = current_position[0], current_position[1]
    
        # Iterate through 5x5 grid centered on current position --> 3x3 grid
        for i in range(max(0, x-2), min(60, x+2)):
            for j in range(max(0, y-2), min(60, y+2)):
                if self.reward_map[i, j] == 0:
                    self.reward_map[i, j] = 1
                    self.reward_components["explore_bonus_new_field"] += 1
                
        
        # Only give reward if any new cells were explored
        # if reward_given:
        #     self.reward_components["explore_bonus_new_field"] = 1
        # # Area visited once
        # elif self.reward_map[current_position[0], current_position[1]] == 1:
        #     self.reward_components["explore_bonus_visited_field"] = 0.1
        #     self.reward_map[current_position[0], current_position[1]] = 2
        # # Area visited twice
        # elif self.reward_map[current_position[0], current_position[1]] >=2:
        #     self.reward_components["explore_bonus_visited_field"] = -0.1# darf keine Bestrafung geben, wenn er noch mal auf ein bereits besuchtes Feld fliegt, aber auch keine Belohnung
        #     self.reward_map[current_position[0], current_position[1]] = 3
        
        
        
        reward = (self.reward_components["collision_penalty"] +
                self.reward_components["distance_reward"] +
                self.reward_components["explore_bonus_new_field"] +
                self.reward_components["Target_Hit_Reward"]
                )
        
        # update the buffer
        self.last_total_reward = reward  # Save the last total reward for the dashboard
        

        # Save the reward map to a CSV file
        with open("gym_pybullet_drones/examples/MAZE_TRAINING/reward_map.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.reward_map)

        # COMPUTE TOTAL REWARD
        reward = (
            self.reward_components["collision_penalty"]
            + self.reward_components["distance_reward"]
            + self.reward_components["explore_bonus_new_field"]
            + self.reward_components["explore_bonus_visited_field"]
            + self.reward_components["Target_Hit_Reward"]
        )
        self.last_total_reward = reward  # Save the last total reward for the dashboard

        return reward

    ################################################################################

    def _computeTerminated(self):
        """Terminated when the drone is in the sweet spot for 5 seconds."""
        state = self._getDroneStateVector(0)
        # starte einen Timer, wenn die Drohne im sweet spot ist
        if state[25] < 1:  # 0.15 = Radius Scheibe
            self.still_time += 1 / self.reward_and_action_change_freq  # Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
        else:
            self.still_time = 0.0  # Reset timer to 0 seconds

        # Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
        if self.still_time >= 5:
            current_time = time.localtime()
            Grund_Terminated = f"Drohne ist 5 s lang unter dem Objekt gewesen. Zeitstempel (min:sek) {time.strftime('%M:%S', current_time)}"
            return True, Grund_Terminated

        Grund_Terminated = None

        return False, Grund_Terminated

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

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        state = self._getDroneStateVector(0)
        # starte einen Timer, wenn die Drohne im sweet spot ist
        if state[25] < 1:  # 0.15 = Radius Scheibe
            self.still_time += 1 / self.reward_and_action_change_freq  # Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
        else:
            self.still_time = 0.0  # Reset timer to 0 seconds

            # Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
            if self.still_time >= 5:
                current_time = time.localtime()
                Grund_Terminated = f"Drohne ist 5 s lang unter dem Objekt gewesen. Zeitstempel (min:sek) {time.strftime('%M:%S', current_time)}"
                return True, Grund_Terminated

            Grund_Terminated = None

            return False, Grund_Terminated

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
            Grund_Truncated = "Crash, Abstand < 0.1 m"
            return True, Grund_Truncated

        # Wenn an einer Wand gecrashed wird, beenden!
        Abstand_truncated = self.Danger_Threshold_Wall - 0.05
        if state[21] <= Abstand_truncated or state[22] <= Abstand_truncated or state[23] <= Abstand_truncated or state[24] <= Abstand_truncated:
            Grund_Truncated = f"Zu nah an der Wand (<{Abstand_truncated} m)"
            return True, Grund_Truncated

        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
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
