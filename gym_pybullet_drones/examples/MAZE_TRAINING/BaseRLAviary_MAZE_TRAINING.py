import os
import numpy as np
import time
import pybullet as p
from gymnasium import spaces
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from threading import Thread
import webbrowser  # Add this import
import contextlib
import matplotlib.pyplot as plt

from gym_pybullet_drones.examples.MAZE_TRAINING.BaseAviary_MAZE_TRAINING import BaseAviary_MAZE_TRAINING
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.examples.Test_Flo.DSLPIDControl_TestFlo import DSLPIDControl

from stable_baselines3.common.policies import ActorCriticPolicy
import csv
import heapq
import sys
import logging
from collections import deque


class SimpleSlam: 
    def __init__(self, map_size=8, resolution=0.05): # map size 8x8m, damit, egal in welche Richtung die Drohne fliegt, in jeden Quadranten ein komplettes Labyrinth dargestellt werden kann
        """ Erstellt eine leere Occupancy-Grid Map. Args: map_size (float): Seitengröße der Map in Metern (z. B. 8 m). resolution (float): Seitengröße einer Zelle, sodass grid_size ~60 ergibt. """ 
        self.resolution = resolution 
        self.grid_size = int(map_size / resolution) 
        # Initialisiere die Map: 
        # # -1: unbekannt, 0: frei, 1: Wand, 2: besucht (Sensor oben frei) 
        # # 0.2: unbekannt, 0.9: frei, 0.0: Wand, 0.5: besucht (Sensor oben frei) 


        
        self.occupancy_grid = 0.2 * np.ones((self.grid_size, self.grid_size)) 
        self.center = self.grid_size // 2 
        self.path = [] # speichert besuchte Zellen
        self.counter_free_space = 0
        
        
    def reset(self):
        """Reset the SLAM map to its initial state."""
        self.occupancy_grid = 0.2 * np.ones((self.grid_size, self.grid_size))
        self.path = []
        
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
        grid_x, grid_y = self.world_to_grid(x, y)
        self.path.append((grid_x, grid_y))
        # Markiere aktuelle Zelle als frei:
        self.occupancy_grid[grid_x, grid_y] = 0.9
        #self.counter_free_space += 1
        # Falls der "up"-Sensor keinen Treffer hat (z. B. Wert 9999), markiere als besucht:
        if 'up' in raycast_results and raycast_results['up'] == 9999:
            self.occupancy_grid[grid_x, grid_y] = 0.5

        

        # Definiere Richtungswinkel:
        angles = {
            'front': drone_yaw,
            'back': drone_yaw + np.pi,
            'left': drone_yaw + np.pi/2,
            'right': drone_yaw - np.pi/2
        }
        for direction in ['front', 'back', 'left', 'right']:
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
        plt.ioff() # Turn off interactive mode
        plt.figure(figsize=(6,6))
        plt.imshow(self.occupancy_grid.T, cmap='gray', origin='lower')
        if self.path:
            path = np.array(self.path)
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)
            plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=5)
        plt.colorbar(label='Occupancy (-1: unbekannt, 0: frei, 1: Wand, 2: besucht)')
        plt.title('SLAM Map')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        self.OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output_SLAM_MAP')
        #create output folder if it doesn't exist
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
        
        # Save plot to file
        # self.Latest_slam_map_path = os.path.join(self.OUTPUT_FOLDER, "latest_slam_map.png")
        # plt.savefig(self.Latest_slam_map_path)
        plt.close()
        plt.ion() # Turn interactive mode back on


class BaseRLAviary_MAZE_TRAINING(BaseAviary_MAZE_TRAINING):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 60,
                 reward_and_action_change_freq: int = 10,
                 gui=False,
                 user_debug_gui=False,
                 record=False,
                 act: ActionType=ActionType.VEL,
                 advanced_status_plot=False,
                 target_position=np.array([0, 0, 0]),
                 Danger_Threshold_Wall=0.15,
                 map_size_slam=8, #map size 8x8m, damit, egal in welche Richtung die Drohne fliegt, in jeden Quadranten ein komplettes Labyrinth dargestellt werden kann
                 resolution_slam=0.05
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
        # Initialize SLAM before calling the parent constructor
        self.slam = SimpleSlam(map_size=map_size_slam, resolution=resolution_slam)  # 10m x 10m map with 10cm resolution
        self.grid_size = int(map_size_slam / resolution_slam) 
        # Call the parent class constructor
        super().__init__(drone_model=drone_model,
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         advanced_status_plot=advanced_status_plot,
                         user_debug_gui=user_debug_gui, # Remove of RPM sliders from all single agent learning aviaries
                         target_position=target_position,
                         Danger_Threshold_Wall=Danger_Threshold_Wall
                         )

        self.environment_active = True # 05032025: Um den Callback zu stoppen wenn die Umgebung nicht aktiv ist

        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE) 
        ####
        self.reward_and_action_change_freq = reward_and_action_change_freq
        self.ACT_TYPE = act
        self.still_time = 0
        self.EPISODE_LEN_SEC = 5*60 #increased from 5 auf 20 Minuten um mehr zu sehen (4.3.25) auf 5 Min (5.3.25)
        self.TARGET_POSITION = target_position
        self.Danger_Threshold_Wall = Danger_Threshold_Wall
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys
        self.port = 8051
        self.reward_components = {
            'collision_penalty': 0,
            'distance_reward': 0,
            'best_way_bonus': 0,
            'explore_bonus_new_field': 0,
            'Target_Hit_Reward': 0,
        }
        # Historie der Reward-Komponenten für Balkendiagramm
        self.reward_distribution_history = deque(maxlen=50)  # speichere 50 Einträge

        # Initialize reward and best_way map
        self.reward_map = np.zeros((60, 60), dtype=int)
        self.best_way_map = np.zeros((60, 60), dtype=int)
                
        # Counter for the amount of wall pixel in map
        self.wall_pixel_counter = 0
        self.amount_of_pixel_in_map = 60*60
        self.ratio_previous_step = 0
        #self.ratio_current_step = 0
        self.amount_of_pixel_in_map_without_walls = 0
        self.distance_10_step_ago = 0
        self.distance_50_step_ago = 0
        self.differnece_threshold = 0.05
        
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        else:
            print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        
        #### Set a limit on the maximum target speed ###############
        if  act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
            
            
        # ANCHOR - Create DASH Graph
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div([
            dcc.Graph(id='live-map'),
            dcc.Graph(id='observation-channels'), 
            dcc.Graph(id='reward-bar-chart'),
            html.Div(id='current-total-reward'),
            dcc.Interval(
                id='interval-component',
                interval=200,
                n_intervals=0
            )
        ])

        @self.app.callback(
            [Output('live-map', 'figure'),
             Output('observation-channels', 'figure'),
             Output('reward-bar-chart', 'figure'),
             Output('current-total-reward', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph(n):
            if self.environment_active == False:
                # Falls die Umgebung schon beendet wurde,
                # liefere einfach leere oder statische Diagramme,
                # um Fehler zu vermeiden
                empty_fig = go.Figure()
                current_reward_text = "Last Reward: 0.00"
                return empty_fig, empty_fig, empty_fig, current_reward_text
            
            # Create reward/best way map figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Reward Map', 'Best Way Map')
            )
            fig.add_trace(
                go.Heatmap(
                    z=self.reward_map,
                    colorscale='Viridis',
                    showscale=True,
                    name='Reward Map'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Heatmap(
                    z=self.best_way_map,
                    colorscale='Viridis',
                    showscale=True,
                    name='Best Way Map'
                ),
                row=1, col=2
            )
            fig.update_layout(
                height=600,
                title_text="Maze Training Visualization",
                showlegend=True
            )

            # Get current observation channels
            obs = self._computeObs()
            
            # Create observation channels figure with SLAM map on left and values on right
            obs_fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.5, 0.5],  # Make columns equal width
                subplot_titles=('Normalized SLAM Map', 'Legend & Values'),
                specs=[[{"type": "heatmap"}, {"type": "table"}]]
            )

            # Add SLAM map heatmap
            obs_fig.add_trace(
                go.Heatmap(
                    z=obs[0],
                    colorscale=[
                        [0, 'rgb(0,0,0)'],      # Wall (0.0)
                        [0.2, 'rgb(128,128,128)'],  # Unknown (0.2)
                        [0.5, 'rgb(255,165,0)'],    # Visited (0.5)
                        [0.9, 'rgb(255,255,255)']   # Free (0.9)
                    ],
                    showscale=False,
                    name='SLAM Map'
                ),
                row=1, col=1
            )

            # Create table with legend and observation values
            obs_fig.add_trace(
                go.Table(
                    header=dict(values=['Type', 'Description']),
                    cells=dict(
                        values=[
                            ['Wall', 'Unknown', 'Visited', 'Free', '', 'Position X', 'Position Y', 'sin(yaw)', 'cos(yaw)'],
                            ['Black (0.0)', 'Gray (0.2)', 'Orange (0.5)', 'White (0.9)', '',
                             f'{obs[1][0][0]:.3f}', f'{obs[2][0][0]:.3f}', 
                             f'{obs[3][0][0]:.3f}', f'{obs[4][0][0]:.3f}']
                        ]
                    )
                ),
                row=1, col=2
            )

            obs_fig.update_layout(
                height=600,  # Make it square
                title_text="Observation Channels",
                showlegend=False
            )

            
            # Create reward components bar chart
            bar_chart = go.Figure(
                go.Bar(
                    x=list(self.reward_components.keys()),
                    y=list(self.reward_components.values()),
                    marker_color='royalblue'
                )
            )
            bar_chart.update_layout(
                title_text="Current Reward Components",
                xaxis_title="Reward Type",
                yaxis_title="Reward Value"
            )

            # Initialize last_total_reward if not set
            if not hasattr(self, 'last_total_reward'):
                self.last_total_reward = 0
                
            current_reward_text = f"Last Reward: {self.last_total_reward:.2f}"

            return fig, obs_fig, bar_chart, current_reward_text

        # Start Dash server in background thread
        def run_dash_app():
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
            self.app.run_server(debug=False, port=self.port)

        self.dashboard_thread = Thread(target=run_dash_app, daemon=True)
        self.dashboard_thread.start()

        # Open web browser after a short delay to ensure server is running
        def open_browser():
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f'http://localhost:{self.port}')

        Thread(target=open_browser, daemon=True).start()

        # def close():
        #     self.environment_active = False
    
        
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
            change_value_yaw = action[k,4]
            Calculate_new_yaw = current_yaw + change_value_yaw
            
            
            temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=np.array([state[0], state[1], 0.5]), # same as the current position on X, and same on y (not as in fly to wall scenario) and z = 0.5
                                                    target_rpy=np.array([0,0,Calculate_new_yaw]), # neue Yaw-Werte durch Drehung der Drohne
                                                    target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
            rpm[k,:] = temp
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
        obs_lower_bound = np.array([-99, -99,-2*np.pi,0,0,0,0,0]) #x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up
    
        obs_upper_bound = np.array([99, 99, 2*np.pi, 9999,9999,9999,9999,9999]) #Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up
                                    
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float32
            )
        
        
        
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([0]) #Raycast reading forward
    
        obs_upper_bound = np.array([9999]) #Raycast reading forward
                                    
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float32
            )


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
        
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

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
        obs = [state[21],state[22],state[23],state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up
        
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
        modified_obs.append(round(state[0],2)) #x-Position
        modified_obs.append(round(state[1],2)) #y-Position
        modified_obs.append(round(state[9],2)) #Yaw-Position
        
        for distance in obs:
            if distance <= (self.Danger_Threshold_Wall):  # Too close to wall, Safetyalgorithmus wird gegensteuern
                modified_obs.append(0)
            elif distance <= (self.Danger_Threshold_Wall+0.1):  # Vorwarnung: gleich passiert was
                modified_obs.append(1) 
            else:  # Safe distance
                modified_obs.append(2)
        
        #raycast oben noch anhängen        
        if state[25] < 1:
            modified_obs.append(1)
        else:
            modified_obs.append(9999)
                
        return np.array(modified_obs, dtype=np.float32) # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,9999)
        
    
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

################################################################################
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
        try:
            if self.environment_active == False:
                self.restart_environment()

            # if self.environment_active == False:
            #     p.disconnect(physicsClientId=self.CLIENT)
            #     time.sleep(2000)
            #     p.connect(p.GUI)
            #     self.environment_active = True

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
                    
            #Achtung: in der Simulation sind nie negative Werte zu erwarten, da die Mazes so gespant sind, das Sie immer positive Werte aufweisen. In echt kann die Drohne aber später auch negative Werte erhalten.
            
            # Normalisiere x und y: Angenommener Bereich [-5, 5]
            norm_x = (pos[0] + 4) / 8
            norm_y = (pos[1] + 4) / 8

            # Muss auf die Input Shape des DQN angepasst werden: (grid_size, grid_size)
            pos_x_channel = np.full((self.grid_size, self.grid_size), norm_x, dtype=np.float32)
            pos_y_channel = np.full((self.grid_size, self.grid_size), norm_y, dtype=np.float32)

            # Yaw in zwei Kanäle: sin und cos
            yaw = state[9] # [9]=yaw-Winkel
            yaw_sin_channel = np.full((self.grid_size, self.grid_size), np.sin(yaw), dtype=np.float32)
            yaw_cos_channel = np.full((self.grid_size, self.grid_size), np.cos(yaw), dtype=np.float32)
            


            # Staple die 5 Kanäle zusammen: Shape = (5, grid_size, grid_size)
            obs = np.stack([slam_map, pos_x_channel, pos_y_channel, yaw_sin_channel, yaw_cos_channel], axis=0)
            self.obs = obs # für Visualisierung in dem Dashboard

            return obs

        except Exception as e:
            self.logger.error(f"Error in _computeObs: {e}")
            self.environment_active = False
            return None
 
    

#####################################################################################
    
    # ANCHOR - computeReward
    def _computeReward(self, Maze_Number): # Funktioniert und die Drohne lernt, nahe an die Wand, aber nicht an die Wand zu fliegen. Problem: die Drohne bleibt nicht sauber im Sweetspot stehen.
        """Computes the current reward value.

        # _Backup_20250211_V1_with_mixed_reward_cases
        Returns
        -------
        float
            The reward.

        """
        if self.step_counter == 0:
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
            # Loading the Walls of the CSV Maze into the reward map as ones
            reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
            #reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"
            
            with open(reward_map_file_path, 'r') as file:
                reader = csv.reader(file)
                for i, row in enumerate(reader):
                    for j, value in enumerate(row):
                        if value == "1":
                            self.reward_map[i, j] = 6 # Wand
                            self.wall_pixel_counter += 1
            # Mirror the reward map vertically
            #self.reward_map = np.flipud(self.reward_map)
            # Rotate the reward map 90° mathematically negative
            #self.reward_map = np.rot90(self.reward_map, k=4)
            
            # Set the Startpoint of the Drone
            initial_position = [self.INIT_XYZS[0][0]/0.05, self.INIT_XYZS[0][1]/0.05] # Startpunkt der Drohne
            self.reward_map[int(initial_position[0]), int(initial_position[1])] = 4 # Startpunkt

            # Set the Targetpoint of the Drone
            target_position = [self.TARGET_POSITION[0]/0.05, self.TARGET_POSITION[1]/0.05] # Zielpunkt der Drohne
            self.reward_map[int(target_position[0]), int(target_position[1])] = 5 # Zielpunkt

            # Amount of pixel in the map without walls
            self.amount_of_pixel_in_map_without_walls = self.amount_of_pixel_in_map - self.wall_pixel_counter

            # Best way to fly via A* Algorithm
            self.best_way_map = np.zeros((60, 60), dtype=int)
            def heuristic(a, b):
                return np.linalg.norm(np.array(a) - np.array(b))  # Euklidische Distanz
            
            

            def a_star_search(reward_map, start, goal):
                neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                close_set = set()
                came_from = {}
                gscore = {start: 0}
                fscore = {start: heuristic(start, goal)}
                oheap = []

                heapq.heappush(oheap, (fscore[start], start))

                while oheap:
                    current = heapq.heappop(oheap)[1]

                    if current == goal:
                        path = [goal]
                        while current in came_from:
                            current = came_from[current]
                            path.append(current)
                        path.reverse()
                        return path

                    close_set.add(current)
                    for i, j in neighbors:
                        neighbor = (current[0] + i, current[1] + j)

                        if not (0 <= neighbor[0] < reward_map.shape[0] and 0 <= neighbor[1] < reward_map.shape[1]):
                            continue  # Außerhalb des Grids

                        if reward_map[neighbor[0], neighbor[1]] == 6:
                            continue  # Hindernis überspringen

                        tentative_g_score = gscore[current] + reward_map[neighbor[0], neighbor[1]]  # Kosten berücksichtigen

                        if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                            continue

                        if tentative_g_score < gscore.get(neighbor, float('inf')) or all(n[1] != neighbor for n in oheap):
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            heapq.heappush(oheap, (fscore[neighbor], neighbor))

                return None  # Kein Pfad gefunden

            start = (int(initial_position[0]), int(initial_position[1]))
            goal = (int(target_position[0]), int(target_position[1]))
            path = a_star_search(self.reward_map, start, goal)

            # Initializing the best way map
            if path:
                for position in path:
                    self.best_way_map[position[0], position[1]] = 1
            
            # Save the best way map to a CSV file
            with open('best_way_map.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.best_way_map)

            # Save the reward map to a CSV file
            with open('reward_map.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.reward_map)
                
        
        reward = 0
        state = self._getDroneStateVector(0) #erste Drohne


        
        #### Rewards initialisieren ####
        self.reward_components["collision_penalty"] = 0
        self.reward_components["distance_reward"] = 0
        self.reward_components["best_way_bonus"] = 0
        self.reward_components["explore_bonus_new_field"] = 0
        #self.reward_components["explore_bonus_visited_field"] = 0
        self.reward_components["Target_Hit_Reward"] = 0
        
        ###### 1.PUNISHMENT FOR COLLISION ######
        if self.action_change_because_of_Collision_Danger == True:
            self.reward_components["collision_penalty"] = -10.0

        ###### 2.REWARD FOR DISTANCE TO TARGET (line of sight) ######
        # Get current drone position and target position
        drone_pos = state[0:2]  # XY position from state vector
        target_pos = self.TARGET_POSITION[0:2]
        
        # Calculate distance to target
        self.distance = np.linalg.norm(drone_pos - target_pos)
        # Define max distance and max reward
        MAX_DISTANCE = 2.0  # Maximum expected distance in meters
        MAX_REWARD = 2   # Maximum reward for distance (excluding target hit bonus)
        
        # Linear reward that scales from 0 (at MAX_DISTANCE) to MAX_REWARD (at distance=0)
        distance_ratio = min(self.distance/MAX_DISTANCE, 1.0)
        self.reward_components["distance_reward"] = MAX_REWARD * (1 - distance_ratio) ## 4.3.25: auf Linear umgestellt, damit auch in weiter entfernten Feldern noch ein Gradient erkannt werden kann
        
        # Add huge reward if target is hit (within 0.05m) and top sensor shows no obstacle
        if self.distance < 0.15 and state[25] < 1: # 0.15 = Radius Scheibe
            self.reward_components["Target_Hit_Reward"] += 1000.0
            print(f"Target hit. Zeitstempel (min:sek) {time.strftime('%M:%S', time.localtime())}")

        if self.step_counter%10 == 0:
        # Check if the distance to the target has not changed
            difference = np.abs(self.distance - self.distance_10_step_ago)
            # print(f"Difference to 10 steps ago: {difference}")
            # print(f"Distance to target: {self.distance}")
            # print(f"Distance 10 steps ago: {self.distance_10_step_ago}")

            if self.distance < 0.20:
                self.reward_components["distance_reward"] = self.reward_components["distance_reward"] + 1
            
            
            elif difference >= 0.5:
                self.reward_components["distance_reward"] = self.reward_components["distance_reward"] + 1

            self.distance_10_step_ago = self.distance
        
        if self.step_counter%100 == 0:

            difference_50 = np.abs(self.distance - self.distance_50_step_ago)

            if self.distance < 0.20:
                self.reward_components["distance_reward"] = self.reward_components["distance_reward"] + 1
            elif difference_50 < self.differnece_threshold:
                self.reward_components["distance_reward"] = self.reward_components["distance_reward"] - 2

            self.distance_50_step_ago = self.distance
        
        current_position = [int(state[0]/0.05), int(state[1]/0.05)]    
        # ###### 3. REWARD FOR BEING ON THE BEST WAY ######
        # Get the current position of the drone
        # Check if the drone is on the best way
        if self.best_way_map[current_position[0], current_position[1]] == 1:
            self.reward_components["best_way_bonus"] = 1
        
        # ###### 4. REWARD FOR EXPLORING NEW AREAS ######
        # # Check if the drone is in a new area
        # New area
        # if self.reward_map[current_position[1], current_position[0]] == 0:
        #     self.reward_components["explore_bonus"] = 1
        #     self.reward_map[current_position[1], current_position[0]] = 1
        # # Area visited once
        # elif self.reward_map[current_position[1], current_position[0]] == 1:
        #     self.reward_components["explore_bonus"] = 0.1
        #     self.reward_map[current_position[1], current_position[0]] = 2
        # # Area visited twice
        # elif self.reward_map[current_position[1], current_position[0]] == 2:
        #     self.reward_components["explore_bonus"] = 0.1
        #     self.reward_map[current_position[1], current_position[0]] = 3
        if self.reward_map[current_position[0], current_position[1]] == 0:
            #self.reward_components["explore_bonus_new_field"] = 2
            self.reward_map[current_position[0], current_position[1]] = 1
        # Area visited once
        elif self.reward_map[current_position[0], current_position[1]] == 1:
            #self.reward_components["explore_bonus_visited_field"] = 0.1
            self.reward_map[current_position[0], current_position[1]] = 2
        # Area visited twice
        elif self.reward_map[current_position[0], current_position[1]] >=2:
            #self.reward_components["explore_bonus_visited_field"] = -0.01# darf keine Bestrafung geben, wenn er noch mal auf ein bereits besuchtes Feld fliegt, aber auch keine Belohnung
            self.reward_map[current_position[0], current_position[1]] = 3
        

        # Save the best way map to a CSV file
        # with open('gym_pybullet_drones/examples/MAZE_TRAINING/best_way_map.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.best_way_map)

        # Save the reward map to a CSV file
        with open('gym_pybullet_drones/examples/MAZE_TRAINING/reward_map.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.reward_map)   


        ##### 5. REWARD FOR DIFFERENCE IN AMOUNT OF EXPLOERED FIELDS #####
        # Calculate the ratio of visited fields
        ratio_current_step = self.slam.counter_free_space/self.amount_of_pixel_in_map_without_walls
        difference_ratio = ratio_current_step - self.ratio_previous_step

        #print(f"Ratio current step: {ratio_current_step}")


        if difference_ratio < 0.01:
            self.reward_components["explore_bonus_new_field"] = -2  # 0 Punkte für bereits erkundete Felder
        elif difference_ratio > 0.010:
            self.reward_components["explore_bonus_new_field"] = difference_ratio * 100 # 10 Punkte für 100% erkundete Felder
        else:
            self.reward_components["explore_bonus_new_field"] = 0

        self.slam.counter_free_space = 0 # Reset the counter of free space for the next step

        ##### 


        # COMPUTE TOTAL REWARD
        reward = self.reward_components["collision_penalty"] + self.reward_components["distance_reward"] + self.reward_components["best_way_bonus"] + self.reward_components["explore_bonus_new_field"] + self.reward_components["Target_Hit_Reward"]
        self.last_total_reward = reward  # Save the last total reward for the dashboard

        self.ratio_previous_step = ratio_current_step

        return reward
        
    ################################################################################
    
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        state = self._getDroneStateVector(0)
        #starte einen Timer, wenn die Drohne im sweet spot ist
        if self.distance < 0.15 and state[25] < 1: #0.15 = Radius Scheibe
            self.still_time += (1/self.reward_and_action_change_freq)# Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
        else:
            self.still_time = 0.0 # Reset timer to 0 seconds

        #Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
        if self.still_time >= 5:
            current_time = time.localtime()
            Grund_Terminated = f"Drohne ist 5 s lang unter dem Objekt gewesen. Zeitstempel (min:sek) {time.strftime('%M:%S', current_time)}"
            self.environment_active = False
            return True, Grund_Terminated
        
        Grund_Terminated = None
        
        return False, Grund_Terminated
    
    ################################################################################
    
    def _computeTruncated(self): #coppied from HoverAviary_TestFlo.py
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the drone is too tilted or has crashed into a wall.

        """
        # Truncate when the drone is too tilted
        state = self._getDroneStateVector(0)
        if abs(state[7]) > .4 or abs(state[8]) > .4: 
            Grund_Truncated = "Zu tilted"
            self.environment_active = False
            return True, Grund_Truncated
        
        # TBD wenn die Drone abstürzt, dann auch truncaten
        if state[2] < 0.1: #state[2] ist z_position der Drohne
            Grund_Truncated = "Crash, Abstand < 0.1 m"
            self.environment_active = False
            return True, Grund_Truncated

        #Wenn an einer Wand gecrashed wird, beenden!
        Abstand_truncated = self.Danger_Threshold_Wall-0.05
        if (state[21] <= Abstand_truncated  or state[22] <= Abstand_truncated or state[23] <= Abstand_truncated or state[24] <= Abstand_truncated):
            Grund_Truncated = f"Zu nah an der Wand (<{Abstand_truncated} m)"
            self.environment_active = False
            return True, Grund_Truncated
        
        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            Grund_Truncated = "Zeit abgelaufen"
            self.environment_active = False
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
 
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    #########################################################################################
        
    def restart_environment(self):
        try:
            self.close()
            self._housekeeping()
            self._updateAndStoreKinematicInformation()
            self._startVideoRecording()
            self.environment_active = True
            self.logger.info("Environment restarted successfully")
        except Exception as e:
            self.logger.error(f"Error restarting environment: {e}")
            self.environment_active = False
    
    