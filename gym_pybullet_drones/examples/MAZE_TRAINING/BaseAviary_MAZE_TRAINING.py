import collections
import contextlib
import csv
import heapq
import logging
import os
import socket
import sys
import time
import webbrowser  # Add this import
import xml.etree.ElementTree as etxml
from collections import deque
from datetime import datetime
from sys import platform
from threading import Thread

import dash
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import plotly.graph_objects as go
import pybullet as p
import pybullet_data
from dash import dcc, html
from dash.dependencies import Input, Output
from gymnasium import spaces
from PIL import Image
from plotly.subplots import make_subplots
from SimpleSlam_MAZE_TRAINING import SimpleSlam
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.examples.MAZE_TRAINING._actionSpace import (
    _actionSpace as _actionSpace_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._computeObs import (
    _computeObs as _computeObs_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._computeReward import (
    _computeReward as _computeReward_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._computeTerminated import (
    _computeTerminated as _computeTerminated_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._computeTruncated import (
    _computeTruncated as _computeTruncated_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._observationSpace import (
    _observationSpace as _observationSpace_outsource,
)
from gym_pybullet_drones.examples.MAZE_TRAINING._preprocessAction import (
    _preprocessAction as _preprocessAction_outsource,
)
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ImageType,
    ObservationType,
    Physics,
)


class BaseRLAviary_MAZE_TRAINING(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 100,  # simulation frequency
        ctrl_freq: int = 50,
        reward_and_action_change_freq: int = 10,
        gui=False,
        user_debug_gui=False,
        record=False,
        act: ActionType = ActionType.VEL,
        advanced_status_plot=False,
        obstacles=True,
        vision_attributes=False,
        output_folder="results_maze_training" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        target_position=np.array([0, 0, 0]),
        Danger_Threshold_Wall=0.20,
        EPISODE_LEN_SEC=10 * 60,
        dash_active=False,
        map_size_slam=8,  # map size 8x8m, damit, egal in welche Richtung die Drohne fliegt, in jeden Quadranten ein komplettes Labyrinth dargestellt werden kann
        resolution_slam=0.05,
        REWARD_VERSION="R1",
        ACTION_TYPE="A1",
        OBSERVATION_TYPE="O1",
        Pushback_active=False,
        DEFAULT_Multiplier_Collision_Penalty=2,
        VelocityScale = 1,
        Procent_Step = 0.05
    ):
        """Initialization of a generic aviary environment.

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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        advanced_status_plot : bool, optional
            Whether to plot the advanced status of the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        #### Constants #############################################
        self.REWARD_VERSION = REWARD_VERSION
        self.ACTION_SPACE_VERSION = ACTION_TYPE
        self.OBSERVATION_TYPE = OBSERVATION_TYPE
        self.PUSHBACK_ACTIVE = Pushback_active
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        self.REWARD_AND_ACTION_CHANGE_FREQ = reward_and_action_change_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError("[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.")
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.PYB_STEPS_PER_REWARD_AND_ACTION_CHANGE = int(self.PYB_FREQ / self.REWARD_AND_ACTION_CHANGE_FREQ)
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1.0 / self.PYB_FREQ
        self.REWARD_AND_ACTION_CHANGE_TIMESTEP = 1.0 / self.REWARD_AND_ACTION_CHANGE_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.TARGET_POSITION = target_position
        self.Danger_Threshold_Wall = Danger_Threshold_Wall
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.ADVANCED_STATUS_PLOT = advanced_status_plot
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder

        # SECTION - INIT aus alter RLAviary
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        self.ACT_TYPE = act
        self.still_time = 0
        # NOTE - Episoden länge
        self.EPISODE_LEN_SEC = EPISODE_LEN_SEC  # increased from 5 auf 20 Minuten um mehr zu sehen (4.3.25)
        self.port = 8080
        self.reward_components = {"collision_penalty": 0, "distance_reward": 0, "explore_bonus_new_field": 0, "explore_bonus_visited_field": 0, "Target_Hit_Reward": 0}

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
        self.Multiplier_Collision_Penalty = DEFAULT_Multiplier_Collision_Penalty
        self.VelocityScale = VelocityScale
        self.Area_counter = 0
        self.Area_counter_Max = 0
        self.previous_Procent = 1
        self.Procent_Step = Procent_Step

        # Initialize SLAM before calling the parent constructor
        self.slam = SimpleSlam(map_size=map_size_slam, resolution=resolution_slam)  # 10m x 10m map with 10cm resolution
        self.grid_size = int(map_size_slam / resolution_slam)

        #### Create integrated controllers #########################
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        else:
            print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        self.DASH_ACTIVE = dash_active
        #### Load the drone properties from the .urdf file #########
        (
            self.M,
            self.L,
            self.THRUST2WEIGHT_RATIO,
            self.J,
            self.J_INV,
            self.KF,
            self.KM,
            self.COLLISION_H,
            self.COLLISION_R,
            self.COLLISION_Z_OFFSET,
            self.MAX_SPEED_KMH,
            self.GND_EFF_COEFF,
            self.PROP_RADIUS,
            self.DRAG_COEFF,
            self.DW_COEFF_1,
            self.DW_COEFF_2,
            self.DW_COEFF_3,
        ) = self._parseURDFParameters()
        # NOTE - Maze Anzahl Wechsel
        self.Maze_number = 21
        self.New_Maze_number = 5
        self.New_Maze_number_counter = 0
        # The random number to generate the init and target position
        self.random_number_Start = 1
        self.random_number_Target = 2
        print(
            "[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
                self.M,
                self.L,
                self.J[0, 0],
                self.J[1, 1],
                self.J[2, 2],
                self.KF,
                self.KM,
                self.THRUST2WEIGHT_RATIO,
                self.MAX_SPEED_KMH,
                self.GND_EFF_COEFF,
                self.PROP_RADIUS,
                self.DRAG_COEFF[0],
                self.DRAG_COEFF[2],
                self.DW_COEFF_1,
                self.DW_COEFF_2,
                self.DW_COEFF_3,
            )
        )
        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = 4 * self.KF * self.MAX_RPM**2
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM**2) / np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = self.L * self.KF * self.MAX_RPM**2
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM**2) / np.sqrt(2)
        self.MAX_Z_TORQUE = 2 * self.KM * self.MAX_RPM**2
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ % self.PYB_STEPS_PER_CTRL != 0:
                print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[1.5, 1.5, 0], physicsClientId=self.CLIENT)
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1 * np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller " + str(i) + " RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH = int(640)
                self.VID_HEIGHT = int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3, yaw=-30, pitch=-30, roll=0, cameraTargetPosition=[0, 0, 0], upAxisIndex=2, physicsClientId=self.CLIENT)
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0, aspect=self.VID_WIDTH / self.VID_HEIGHT, nearVal=0.1, farVal=1000.0)

        if self.ADVANCED_STATUS_PLOT:
            # Matplotlib Setup für Live-Plot
            self.fig, self.ax = plt.subplots()
            self.time_vals = []
            self.wall_distance_vals = []  # Abstand zur Wand
            self.action_vals = []  # Gewählte Aktion (-1, 0, 1)
            self.step_reward_vals = []  # Reward des aktuellen Schritts
            self.total_reward_vals = []  # Gesamtreward

            (self.line_wall_distance,) = self.ax.plot([], [], "b-", label="Abstand zur Wand [m]")
            (self.line_action,) = self.ax.plot([], [], "g-", label="Aktion (1, 0, -1)")
            (self.line_step_reward,) = self.ax.plot([], [], "r-", label="Step Reward")
            (self.line_total_reward,) = self.ax.plot([], [], "orange", label="Gesamtreward")

            self.ax.set_xlim(0, 100)  # X-Achse für 100 Zeitschritte
            self.ax.set_ylim(-5, 5)  # Skalierung für die verschiedenen Werte
            self.ax.set_xlabel("Zeitschritt")
            self.ax.set_ylabel("Werte")
            self.ax.legend()
            plt.ion()  # Interaktiver Modus für Live-Update

        #### Set initial poses #####################################
        self.INIT_XYZS = initial_xyzs
        if self.INIT_XYZS is None:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        print(self.Maze_number, "-------------------------------INIT MAZE NUMMER-----------------------------")
        self._housekeeping()

        #### Update and store the drones kinematic information #####

        self._updateAndStoreKinematicInformation()

        #### Start video recording #################################
        self._startVideoRecording()

        self._update_camera()

        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

        #### Start Dash server #####################################
        if self.DASH_ACTIVE:
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
                ##NOTE - ValueError:  HIER  KOMMT ERRORs
                # Invalid value of type 'numpy.float32' received for the 'z' property of heatmap
                #     Received value: 1.8

                # The 'z' property is an array that may be specified as a tuple,
                # list, numpy array, or pandas Series
                obs_fig.add_trace(
                    go.Heatmap(
                        z=obs[0].tolist(),
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

    def reset(self, seed: int = None, options: dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        #### Reset the simulation ##################################

        if self.New_Maze_number_counter == self.New_Maze_number:
            # self.Maze_number = np.random.randint(1, 20)
            # solang nicht alle csv datei erstellt dann ändern auf 20
            # Erstellen Sie eine Liste der zulässigen Zahlen

            # Wählen Sie eine Zufallszahl aus der Liste der zulässigen Zahlen
            # self.Maze_number = np.random.choice((1, 20))
            self.Maze_number = 21

            print(f"--------------------------MAZE_NUMBER_NEWWWWWWWWW: {self.Maze_number}---------------------------------------")
            print(f"--------------------------MAZE_NUMBER_NEWWWWWWWWW: {self.Maze_number}---------------------------------------")
            print(f"--------------------------MAZE_NUMBER_NEWWWWWWWWW: {self.Maze_number}---------------------------------------")
            self.New_Maze_number_counter = 0
        else:
            self.New_Maze_number_counter += 1
            print(f"--------------------------MAZE_NUMBER: {self.Maze_number}---------------------------------------")
            print(f"--------------------------MAZE_NUMBER_Counter: { self.New_Maze_number_counter}---------------------------------------")

        p.resetSimulation(physicsClientId=self.CLIENT)

        # if self.USER_DEBUG:
        #     #### Housekeeping ##########################################
        #     print("Start housekeeping - INIT_XYZS:", self.INIT_XYZS)
        #     self._housekeeping()
        #     print("End housekeeping - INIT_XYZS:", self.INIT_XYZS)

        #     #### Start video recording #################################
        #     self._startVideoRecording()
        #     #### Update and store the drones kinematic information #####
        #     print("Start kinematic update - INIT_XYZS:", self.INIT_XYZS)
        #     print(f"self.pos vor Update: {self.pos}")
        #     self._updateAndStoreKinematicInformation()
        #     print(f"self.pos nach Update: {self.pos}")
        #     print("End kinematic update - INIT_XYZS:", self.INIT_XYZS)
        #     #### Start video recording #################################
        #     self._startVideoRecording()
        #     #### Return the initial observation ########################
        #     initial_obs = None
        #     initial_obs = self._computeObs()
        #     nth_drone = 0 #weil das so standardmäßig im computeObs festegelegt ist
        #     print(f"Getting state for nth_drone: {nth_drone}")

        #     # Get all bodies in simulation
        #     all_bodies = p.getNumBodies(physicsClientId=self.CLIENT)
        #     print(f"Bodies in simulation: {all_bodies}")

        #     # Get the actual drone ID
        #     drone_id = nth_drone + 1  # If this is the issue, you might need to adjust this
        #     print(f"Accessing drone ID: {drone_id}")

        #     # Verify the body is actually a drone
        #     body_info = p.getBodyInfo(drone_id, physicsClientId=self.CLIENT)
        #     print(f"Body info for ID {drone_id}: {body_info}")

        #     print(f"Environment resettet, initial_obs: {initial_obs}\n")

        #     initial_info = self._computeInfo()

        #     return initial_obs, initial_info

        #### Housekeeping ##########################################
        self._housekeeping()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = None
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        self.slam.reset()  # TODO - Reset SLAM evtl. nicht in allen Modellen

        return initial_obs, initial_info

    ################################################################################
    # ANCHOR - STEP
    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Preprocess the action and translate it into RPMs

        # # Find wheter and what sort of action is taken
        # print(action)

        actual_action_0_bis_8 = int(action.item())

        # print(f"actual_action_0_bis_8: {actual_action_0_bis_8}")

        # Lokale Koordinaten auf der Drohne
        # Für die Maze-Trainings-Umgebung 8 Möglichkeiten
        action_to_movement_direction_local = {
            # 0: np.array([[0, 0, 0, 0.5, 0]]), # Fly 0° (Stay)
            0: np.array([[1, 0, 0, self.VelocityScale, 0]]),  # Fly 90° (Forward)
            1: np.array([[-1, 0, 0, self.VelocityScale, 0]]),  # Fly 180° (Backward)
            2: np.array([[0, 1, 0, self.VelocityScale, 0]]),  # Fly 90° (Left)
            3: np.array([[0, -1, 0, self.VelocityScale, 0]]),  # Fly 270° (Right)
            4: np.array(
                [[0, 0, 0, self.VelocityScale, 1 / 72 * np.pi]]
            ),  # 45° Left-Turn # NOTE - Tests mit 1/36*np.pi waren nicht so gut, da die Drohne scheinbar nicht verstanden hat, dass bei einer Drehung vorwärtsfliegen bedeutet
            5: np.array([[0, 0, 0, self.VelocityScale, -1 / 72 * np.pi]]),  # 45° Right-Turn # NOTE - Ausgesetzt für Testzweicke 28.02.25
        }

        if self.step_counter == 0:
            self.time_start_trainrun = 0
            self.timestamp_previous = 0
            self.timestamp_actual = 0
            self.RewardCounterActualTrainRun = 0
            self.List_Of_Tuples_Of_Reward_And_Action = []

        input_action_local = action_to_movement_direction_local[actual_action_0_bis_8]

        action = input_action_local
        self.action = input_action_local

        # Übersetzten in World-Koordinaten
        state = self._getDroneStateVector(0)

        if self.PUSHBACK_ACTIVE == True:
            # self.action = np.zeros((self.NUM_DRONES, 4))
            self.action_change_because_of_Collision_Danger = False

            # Get movement direction based on action
            input_action_local = action_to_movement_direction_local[actual_action_0_bis_8]

            # Convert quaternion to rotation matrix using NumPy
            rot_matrix = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)

            # New Function to check for Collision Danger and change the action if necessary
            self.action_change_because_of_Collision_Danger, action_with_or_without_Collision_Danger_correction = self._check_for_Collision_Danger(input_action_local)

            input_action_Velocity_World = rot_matrix.dot(action_with_or_without_Collision_Danger_correction[0][0:3])

            input_action_complete_world = np.array(
                [np.concatenate((input_action_Velocity_World[0:2], np.array([0]), input_action_local[0][3:5]))]
            )  # gedrehte x,y-Werte, z-Wert 0, yaw-Wert bleibt gleich

            action = input_action_complete_world
            self.action = input_action_complete_world

        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(
                width=self.VID_WIDTH,
                height=self.VID_HEIGHT,
                shadow=1,
                viewMatrix=self.CAM_VIEW,
                projectionMatrix=self.CAM_PRO,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.CLIENT,
            )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), "RGBA")).save(os.path.join(self.IMG_PATH, "frame_" + str(self.FRAME_NUM) + ".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(
                        img_type=ImageType.RGB,  # ImageType.BW, ImageType.DEP, ImageType.SEG
                        img_input=self.rgb[i],
                        path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.PYB_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [
                    p.addUserDebugText(
                        "Using GUI RPM",
                        textPosition=[0, 0, 0],
                        textColorRGB=[1, 0, 0],
                        lifeTime=1,
                        textSize=2,
                        parentObjectUniqueId=self.DRONE_IDS[i],
                        parentLinkIndex=-1,
                        replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                        physicsClientId=self.CLIENT,
                    )
                    for i in range(self.NUM_DRONES)
                ]

        #### Save, preprocess, and clip the action to the max. RPM #

        if self.CTRL_FREQ <= self.REWARD_AND_ACTION_CHANGE_FREQ:
            print("Ctrl-Frequenz ist kleiner oder gleich als die Reward-/Action-Änderungsfrequenz -> Reward-Frequenz wird auf Ctrl-Frequenz gesetzt")

            if not self.USE_GUI_RPM:
                clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

            #### Repeat for as many as the aggregate physics steps #####
            # loope Nachfolgend so oft, dass genau 1x die Ctrl-Frequenz erreicht wird und 1x der Controll aufgerufen wird
            for _ in range(self.PYB_STEPS_PER_CTRL):
                #### Update and store the drones kinematic info for certain
                #### Between aggregate steps for certain types of update ###
                if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                    self._updateAndStoreKinematicInformation()
                #### Step the simulation using the desired physics update ##
                for i in range(self.NUM_DRONES):
                    if self.PHYSICS == Physics.PYB:
                        self._physics(clipped_action[i, :], i)
                    elif self.PHYSICS == Physics.DYN:
                        self._dynamics(clipped_action[i, :], i)
                    elif self.PHYSICS == Physics.PYB_GND:
                        self._physics(clipped_action[i, :], i)
                        self._groundEffect(clipped_action[i, :], i)
                    elif self.PHYSICS == Physics.PYB_DRAG:
                        self._physics(clipped_action[i, :], i)
                        self._drag(self.last_clipped_action[i, :], i)
                    elif self.PHYSICS == Physics.PYB_DW:
                        self._physics(clipped_action[i, :], i)
                        self._downwash(i)
                    elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                        self._physics(clipped_action[i, :], i)
                        self._groundEffect(clipped_action[i, :], i)
                        self._drag(self.last_clipped_action[i, :], i)
                        self._downwash(i)
                #### PyBullet computes the new state, unless Physics.DYN ###
                if self.PHYSICS != Physics.DYN:
                    p.stepSimulation(physicsClientId=self.CLIENT)
                #### Save the last applied action (e.g. to compute drag) ###
                self.last_clipped_action = clipped_action
            #### Update and store the drones kinematic information #####
            self._updateAndStoreKinematicInformation()

            #### Kamera-Einstellungen, scheint aber nicht zu funktionieren (9.2)####
            # p.resetDebugVisualizerCamera(cameraDistance=3,
            #                              cameraYaw=self.rpy[0][2]-30,
            #                              cameraPitch=self.rpy[0][1]-30,
            #                              cameraRoll=self.rpy[0][0],
            #                              cameraTargetPosition=self.pos,
            #                              physicsClientId=self.CLIENT
            #                              )

        else:  # Reward-/Action-Änderungsfrequenz ist kleiner als die Ctrl-Frequenz --> Loop muss öfters durch die Contrl-Freq bzw. Physics-Frequenz durchlaufen werden
            # setzte den Step-Counter für die Zählung der Physics-Frequenz auf 0
            self.PYB_STEPS_IN_ACTUAL_STEP_CALL = 0

            # wiederhole so lange die nachfolgende Loop, wie wir noch nicht an dem Zeitpunkt angekommen sind, dass wir den Reward an das RL zurückgeben
            while self.PYB_STEPS_IN_ACTUAL_STEP_CALL < self.PYB_STEPS_PER_REWARD_AND_ACTION_CHANGE:  # Get new control action if needed

                if not self.USE_GUI_RPM:
                    clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

                # Loop for physics frequency so oft, dass wir nach X-Physics-Schritten die Control-Frequenz erreichen und den Controller erneut aufrufen (eins höher in der Loop)
                for _ in range(self.PYB_STEPS_PER_CTRL):
                    #### Update and store the drones kinematic info for certain
                    #### Between aggregate steps for certain types of update ###
                    if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                        self._updateAndStoreKinematicInformation()
                    #### Step the simulation using the desired physics update ##
                    for i in range(self.NUM_DRONES):
                        if self.PHYSICS == Physics.PYB:
                            self._physics(clipped_action[i, :], i)
                        elif self.PHYSICS == Physics.DYN:
                            self._dynamics(clipped_action[i, :], i)
                        elif self.PHYSICS == Physics.PYB_GND:
                            self._physics(clipped_action[i, :], i)
                            self._groundEffect(clipped_action[i, :], i)
                        elif self.PHYSICS == Physics.PYB_DRAG:
                            self._physics(clipped_action[i, :], i)
                            self._drag(self.last_clipped_action[i, :], i)
                        elif self.PHYSICS == Physics.PYB_DW:
                            self._physics(clipped_action[i, :], i)
                            self._downwash(i)
                        elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                            self._physics(clipped_action[i, :], i)
                            self._groundEffect(clipped_action[i, :], i)
                            self._drag(self.last_clipped_action[i, :], i)
                            self._downwash(i)
                    #### PyBullet computes the new state, unless Physics.DYN ###
                    if self.PHYSICS != Physics.DYN:
                        p.stepSimulation(physicsClientId=self.CLIENT)
                    #### Save the last applied action (e.g. to compute drag) ###
                    self.last_clipped_action = clipped_action

                    self.PYB_STEPS_IN_ACTUAL_STEP_CALL += 1
                    if self.PYB_STEPS_IN_ACTUAL_STEP_CALL == self.PYB_STEPS_PER_REWARD_AND_ACTION_CHANGE:
                        #### Update and store the drones kinematic information #####
                        self._updateAndStoreKinematicInformation()
                        break
                #### Update and store the drones kinematic information #####
                self._updateAndStoreKinematicInformation()

        #########################################################################################
        # SLAM-Update
        pos = state[0:3]
        yaw = state[9]  # Assuming this is the yaw angle

        # Get raycast results
        raycast_results = {"front": state[21], "back": state[22], "left": state[23], "right": state[24], "up": state[25]}

        self.slam.update(pos, yaw, raycast_results)
        # Optional: Zum Debuggen kann man die Map visualisieren (aber im Training besser deaktiviert)

        # NOTE - hier: SLAM Map visualisieren (im Training besser deaktiviert)
        # Achtung: bei AKtivierung wird ein Bild pro Step gespeichert!
        self.slam.visualize()
        #########################################################################################

        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward(self.Maze_number, self.random_number_Start, self.random_number_Target)
        terminated, Grund_Terminated = self._computeTerminated()
        truncated, Grund_Truncated = self._computeTruncated()
        info = self._computeInfo()
        ###Debugging Plots
        state = self._getDroneStateVector(0)  # Einführung neuste

        self.timestamp_actual = self.step_counter * self.PYB_TIMESTEP  # Use simulation time instead of real time
        if reward is not None:
            self.RewardCounterActualTrainRun += reward

        # ANCHOR - Debugging-Plots STEP

        # NOTE - plotting im Trainings-Plot mit zusätzlichen Informationen deaktiviert (28.2.25)
        if self.GUI and self.USER_DEBUG:
            # 19.3.25: EVENTUELL auskommentieren, da das ganze Ding rießig wird :0
            self.List_Of_Tuples_Of_Reward_And_Action.append((action[0][0], reward))

            print("Trainingszeit aktueller Run(s):", "{:.3f}".format(self.timestamp_actual))
            print(f" Observationspace (forward,backward, letzte Action (Velocity in X-Richtung!)):\t {state[21]} \t{state[22]} \t{state[26]}")
            print(f"aktuelle Action (Velocity in X-Richtung!) / Reward für Action: {self.action[0][0]} / {reward}")
            print(f"Reward aktueller Trainingslauf: {self.RewardCounterActualTrainRun}")
            print(f"current Physics-Step / Reward-Steps: {self.step_counter} / {self.timestamp_actual/(1/self.REWARD_AND_ACTION_CHANGE_FREQ)}")
            if truncated:
                print(f"Grund für Truncated: {Grund_Truncated}")
                print(f"List of Tuples of Reward and Action: {self.List_Of_Tuples_Of_Reward_And_Action}\n")
            if terminated:
                print(f"Grund für Terminated: {Grund_Terminated}")
                print(f"List of Tuples of Reward and Action: {self.List_Of_Tuples_Of_Reward_And_Action}\n")

        # if self.GUI: #deaktiviert, damit der nachfolgende Plot immer kommt, auch wenn keine GUI eingeschaltet ist
        if truncated:
            # Zusammenfassung Trainingslauf
            print(f"Zusammenfassung Trainingslauf Truncated (Grund: {Grund_Truncated}):")
            # Remove the redundant print(obs[0]) line
            print(
                f"Observations: x,y,yaw: {state[0]:.3f}, {state[1]:.3f}, {state[9]:.3f}, RayFront/Back: {state[21]:.1f}, {state[22]:.1f}, RayLeft/Right: {state[23]:.1f}, {state[24]:.1f}, RayUp: {state[25]:.1f}"
            )
            print(f"Summe Reward am Ende: {self.RewardCounterActualTrainRun}\n")
        if terminated:
            print(f"Zusammenfassung Trainingslauf Terminated (Grund: {Grund_Terminated}):")
            print(
                f"Observations: x,y,yaw: {state[0]:.3f}, {state[1]:.3f}, {state[9]:.3f}, RayFront/Back: {state[21]:.1f}, {state[22]:.1f}, RayLeft/Right: {state[23]:.1f}, {state[24]:.1f}, RayUp: {state[25]:.1f}"
            )
            print(f"Summe Reward am Ende: {self.RewardCounterActualTrainRun}\n")

        # nachfolgendes war nur zum Debugging der getDroneStateVector Funktion genutzt worden
        # ray_cast_readings = self.check_distance_sensors(0)
        # print(f"Sensor Readings: \n forward {ray_cast_readings[0]} \n backwards {ray_cast_readings[1]} \n left {ray_cast_readings[2]} \n right {ray_cast_readings[3]} \n up {ray_cast_readings[4]} \n down {ray_cast_readings[5]}")

        #### Advance the step counter (for physics-steps) ##############################
        if self.CTRL_FREQ <= self.REWARD_AND_ACTION_CHANGE_FREQ:
            self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        else:
            self.step_counter = self.step_counter + (
                1 * self.PYB_STEPS_IN_ACTUAL_STEP_CALL
            )  # umgeändert auf den neuen Zähler, weil wir in einem Step in diesem Fall mehr Physics-Schritte durchlaufen, als PYB_STEPS_PER_CTRL

        if self.ADVANCED_STATUS_PLOT:
            # Daten für den Plot speichern
            self.time_vals.append(self.step_counter)
            self.wall_distance_vals.append(state[21])
            self.action_vals.append(self.action[0][0])
            self.step_reward_vals.append(reward)
            self.total_reward_vals.append(self.RewardCounterActualTrainRun)

            # Graph aktualisieren
            self.line_wall_distance.set_data(self.time_vals, self.wall_distance_vals)
            self.line_action.set_data(self.time_vals, self.action_vals)
            self.line_step_reward.set_data(self.time_vals, self.step_reward_vals)
            self.line_total_reward.set_data(self.time_vals, self.total_reward_vals)

            self.ax.set_xlim(0, max(100, len(self.time_vals)))  # Dynamische X-Achse

            plt.pause(0.01)  # Kurze Pause für das Update
            plt.draw()  # Manuelles Neuzeichnen erzwingen

        #### Update the camera position to follow the drone ####
        self._update_camera()

        return obs, reward, terminated, truncated, info

    #############################################################################+

    def _update_camera(self):
        """Updates the camera to follow the drone."""
        state = self._getDroneStateVector(0)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[state[0], state[1], state[2]], physicsClientId=self.CLIENT)

    #############################################################################

    # ANCHOR - check_for_Collision_Danger
    def _check_for_Collision_Danger(self, action):
        """Checks for collision danger and changes the action if necessary. Takes the action and checks the distances to the walls in all directions. If a wall is close (<=0.3) and the alogrithm picked an action in the direction of the wall, the action is changed to an halt. The idea is to punish the drone for the try moving into the wall and giving it a chance to correct its action. (Action not going into the direction of a wall will be passed through with change)

        Input:
            action: np.array([1, 0, 0, 0.5])
        Output:
            action_change_because_of_Collision_Danger: bool
            action: If no danger, input action is returned, otherwise a new action (halts) is returned
        """
        actionInput = action

        state = self._getDroneStateVector(0)
        action_change_because_of_Collision_Danger = False

        # wenn front,back,left,right distance kleiner als Threshold und action in die Richtung der Wand -> action = halt
        if state[21] <= self.Danger_Threshold_Wall and actionInput[0][0] == 1:
            action = np.array([[-1, 0, 0, 0.25, 0]])  # Move backward
            action_change_because_of_Collision_Danger = True
            # print(f"Drohne würde vorwärts gegen die Wand fliegen: Distance to Wall_front: {state[21]} --> angepasst\n")
        elif state[22] <= self.Danger_Threshold_Wall and actionInput[0][0] == -1:
            action = np.array([[1, 0, 0, 0.25, 0]])  # Move forward
            action_change_because_of_Collision_Danger = True
            # print(f"Drohne würde rückwärts gegen die Wand fliegen: Distance to Wall_back: {state[22]} --> angepasst\n")
        elif state[23] <= self.Danger_Threshold_Wall and actionInput[0][1] == 1:
            action = np.array([[0, -1, 0, 0.25, 0]])  # Move right
            action_change_because_of_Collision_Danger = True
            # print(f"Drohne würde rechts gegen die Wand fliegen: Distance to Wall_left: {state[23]} --> angepasst\n")
        elif state[24] <= self.Danger_Threshold_Wall and actionInput[0][1] == -1:
            action = np.array([[0, 1, 0, 0.25, 0]])  # Move left
            action_change_because_of_Collision_Danger = True
            # print(f"Drohne würde links gegen die Wand fliegen: Distance to Wall_right: {state[24]} --> angepasst\n")

        return action_change_because_of_Collision_Danger, action

    #############################################################################

    def render(self, mode="human", close=False):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        drone_state = self._getDroneStateVector(0)

        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print(
            "\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
            "——— wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
            "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter * self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter * self.PYB_TIMESTEP) / (time.time() - self.RESET_TIME)),
        )
        for i in range(self.NUM_DRONES):
            print(
                "[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0] * self.RAD2DEG, self.rpy[i, 1] * self.RAD2DEG, self.rpy[i, 2] * self.RAD2DEG),
                "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]),
                "——— ray_front {:+06.4f}\t ray_back {:+06.4f}\t ray_left {:+06.4f}\t ray_right {:+06.4f}\t ray_top {:+06.4f} ——— ".format(
                    drone_state[20], drone_state[21], drone_state[22], drone_state[23], drone_state[24]
                ),
            )

    ################################################################################

    def close(self):
        """Terminates the environment."""
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT

    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS

    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.action = np.zeros((self.NUM_DRONES, 4))
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))  # 4 Werte da das die PRMs der 4 Motoren der letzten Aktion sind
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        if self.New_Maze_number_counter == self.New_Maze_number:
            # self.Maze_number = np.random.randint(1, 21)
            # solang nicht alle csv datei erstellt dann ändern auf 21
            while True:
                self.random_number_Start = np.random.randint(0, 10)
                if self.random_number_Start != self.random_number_Target:
                    break

            Start_Position_swapped = [0, 0, 0.5]  # NOTE - TARGET POSITION FIX
            if self.Maze_number == 0:
                Start_Position = self.INIT_XYZS[f"map{self.Maze_number+1}"][0][self.random_number_Start][0:2]
            else:
                Start_Position = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0:2]
            Start_Position_swapped[1] = Start_Position[0]
            Start_Position_swapped[0] = Start_Position[1]
        else:
            Start_Position_swapped = [0, 0, 0.5]
            if self.Maze_number == 0:
                Start_Position = self.INIT_XYZS[f"map{self.Maze_number+1}"][0][self.random_number_Start][0:2]
            else:
                Start_Position = self.INIT_XYZS[f"map{self.Maze_number}"][0][self.random_number_Start][0:2]
            Start_Position_swapped[1] = Start_Position[0]
            Start_Position_swapped[0] = Start_Position[1]

        # print(f"Start_Position_swapped: {Start_Position_swapped}")
        self.DRONE_IDS = np.array(
            [
                p.loadURDF(
                    pkg_resources.resource_filename("gym_pybullet_drones", "assets/" + self.URDF),
                    Start_Position_swapped,
                    p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=self.CLIENT,
                )
                for i in range(self.NUM_DRONES)
            ]
        )
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
        # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)

    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=os.path.join(self.OUTPUT_FOLDER, "video-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".mp4"), physicsClientId=self.CLIENT
            )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), "")
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    ################################################################################
    # ANCHOR - getDroneStateVector
    def _getDroneStateVector(self, nth_drone):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray
            (26,)-shaped array of floats containing the state vector of the n-th drone.
            The state vector includes:
            - 3x Position (x, y, z) [0:3]                -> -np.inf to np.inf
            - 4x Quaternion (qx, qy, qz, qw) [3:7]       -> unused
            - 3x Roll, pitch, yaw (r, p, y) [7:10]       -> -np.inf to np.inf
            - 3x Linear velocity (vx, vy, vz) [10:13]    -> -np.inf to np.inf
            - 3x Angular velocity (wx, wy, wz) [13:16]   -> -np.inf to np.inf
            - 5x Null [16:21] -> 0 to 9999
            - 4x actual raycast readings (front, back, left, right,up) [21:26] -> 0 to 9999
            - 1x Last action [26]             -> -1, 0, or 1 (velocity in x direction)
        """
        # match MODEL_Version:
        # case "M1":  # M1: PPO
        self.ray_results_actual = self.check_distance_sensors(nth_drone)  # get new actual raycast readings

        if hasattr(self, "action"):
            last_action_VEL_1 = self.action[0][0]
            last_action_VEL_2 = self.action[0][1]
            last_action_VEL_3 = self.action[0][2]
        else:
            last_action_VEL_1 = 0
            last_action_VEL_2 = 0
            last_action_VEL_3 = 0

        state = np.hstack(
            [
                self.pos[nth_drone, :],  # [0:3]
                self.quat[nth_drone, :],  # [3:7]
                self.rpy[nth_drone, :],  # [7:10]
                self.vel[nth_drone, :],  # [10:13]
                self.ang_v[nth_drone, :],  # [13:16]
                0,  # [16]
                0,  # [17]
                0,  # [18]
                0,  # [19]
                0,  # [20]
                self.ray_results_actual[0],  # forward [21]
                self.ray_results_actual[1],  # backward [22]
                self.ray_results_actual[2],  # left [23]
                self.ray_results_actual[3],  # right [24]
                self.ray_results_actual[4],  # up [25]
                last_action_VEL_1, 
                last_action_VEL_2,
                last_action_VEL_3
            ]
        )  # last clipped action [26]: jetzt nur noch 1 Wert (10.2.25)
        return state.reshape(
            29,
        )  # von 30 auf 27 geändert, da nur 1 Wert in lastClipppedACtion (10.2.25)
        # auf 28 geändert, da 2 Werte in lastClippedAction (24.03.25)

    ################################################################################

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]), cameraTargetPosition=target, cameraUpVector=[0, 0, 1], physicsClientId=self.CLIENT)
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0, aspect=1.0, nearVal=self.L, farVal=1000.0)
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0], height=self.IMG_RES[1], shadow=1, viewMatrix=DRONE_CAM_VIEW, projectionMatrix=DRONE_CAM_PRO, flags=SEG_FLAG, physicsClientId=self.CLIENT
        )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(self, img_type: ImageType, img_input, path: str, frame_num: int = 0):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype("uint8"), "RGBA")).save(os.path.join(path, "frame_" + str(frame_num) + ".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype("uint8")
        elif img_type == ImageType.SEG:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype("uint8")
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype("uint8")
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path, "frame_" + str(frame_num) + ".png"))

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES - 1):
            for j in range(self.NUM_DRONES - i - 1):
                if np.linalg.norm(self.pos[i, :] - self.pos[j + i + 1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j + i + 1] = adjacency_mat[j + i + 1, i] = 1
        return adjacency_mat

    ################################################################################

    def _physics(self, rpm, nth_drone):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone], i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)
        p.applyExternalTorque(self.DRONE_IDS[nth_drone], 4, torqueObj=[0, 0, z_torque], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

    ################################################################################

    def _groundEffect(self, rpm, nth_drone):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(self.DRONE_IDS[nth_drone], linkIndices=[0, 1, 2, 3, 4], computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self.CLIENT)
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone], i, forceObj=[0, 0, gnd_effects[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

    ################################################################################

    def _drag(self, rpm, nth_drone):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot.T, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone], 4, forceObj=drag, posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

    ################################################################################

    def _downwash(self, nth_drone):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone], 4, forceObj=downwash, posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

    ################################################################################

    def _dynamics(self, rpm, nth_drone):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone], pos, quat, physicsClientId=self.CLIENT)
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone], vel, np.dot(rotation, rpy_rates), physicsClientId=self.CLIENT)
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([[0, r, -q, p], [-r, 0, p, q], [q, -p, 0, r], [-p, -q, -r, 0]]) * 0.5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    ################################################################################

    def _normalizedActionToRPM(self, action):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action + 1) * self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM) * action)  # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`

    ################################################################################

    def _showDroneLocalAxes(self, nth_drone):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )
            self.Y_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )
            self.Z_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )

    ################################################################################

    def _addObstacles(self):  ## Trainings
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        if self.New_Maze_number_counter == self.New_Maze_number:
            # self.Maze_number = np.random.randint(1, 21)
            # solang nicht alle csv datei erstellt dann ändern auf 21
            while True:
                self.random_number_Target = np.random.randint(0, 10)
                if self.random_number_Target != self.random_number_Start:
                    break

            targetPosition_swapped = [0, 0, 1]  # NOTE - TARGET POSITION FIX
            if self.Maze_number == 0:
                targetPosition = self.TARGET_POSITION[f"map{self.Maze_number+1}"][0][self.random_number_Target][0:2]
            else:
                targetPosition = self.TARGET_POSITION[f"map{self.Maze_number}"][0][self.random_number_Target][0:2]
            targetPosition_swapped[1] = targetPosition[0]
            targetPosition_swapped[0] = targetPosition[1]
        else:
            targetPosition_swapped = [0, 0, 1]  # NOTE - TARGET POSITION FIX
            if self.Maze_number == 0:
                targetPosition = self.TARGET_POSITION[f"map{self.Maze_number+1}"][0][self.random_number_Target][0:2]
            else:
                targetPosition = self.TARGET_POSITION[f"map{self.Maze_number}"][0][self.random_number_Target][0:2]
            targetPosition_swapped[1] = targetPosition[0]
            targetPosition_swapped[0] = targetPosition[1]

        # print(f"Target_Position_swapped: {targetPosition_swapped}")

        # p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/train_square.urdf'), physicsClientId=self.CLIENT, useFixedBase=True)
        p.loadURDF(pkg_resources.resource_filename("gym_pybullet_drones", f"assets/maze/map_{self.Maze_number}.urdf"), physicsClientId=self.CLIENT, useFixedBase=True)

        # NOTE - 19.3.2025: TARGET ENTFERNT
        # p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/target.urdf'),
        #           targetPosition_swapped, # x,y,z position
        #           p.getQuaternionFromEuler([0, 0, 0]), # rotation
        #           physicsClientId=self.CLIENT,
        #           useFixedBase=True)
        # Wände mit Variablen.. läuft aber irgendwie nicht
        # p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/train_quader.urdf'), physicsClientId=self.CLIENT)

    ################################################################################

    def _perform_raycast(self):
        # Perform raycasting in four directions
        ray_length = 10  # Define the length of the rays
        front_ray = p.rayTest([0, 0, 0], [ray_length, 0, 0])
        back_ray = p.rayTest([0, 0, 0], [-ray_length, 0, 0])
        left_ray = p.rayTest([0, 0, 0], [0, ray_length, 0])
        right_ray = p.rayTest([0, 0, 0], [0, -ray_length, 0])

        return {
            "front": front_ray[0][2],  # Distance to the first hit object
            "back": back_ray[0][2],
            "left": left_ray[0][2],
            "right": right_ray[0][2],
        }

    ################################################################################

    # ANCHOR - get Distance Sensors
    def check_distance_sensors(self, nth_drone):
        """
        Check the distance sensors of the Crazyflie drone.
        Args:
            crazyflie_id (int): The PyBullet body ID of the Crazyflie drone.
        Returns:
            list: Sensor readings for each direction (forward, backward, left, right, up, down).
                Each reading is the distance to the nearest obstacle or max_distance if no obstacle is detected.
        """

        drone_id = nth_drone + 1  # nth_drone is 0-based, but the drone IDs are 1-based
        pos, ori = p.getBasePositionAndOrientation(drone_id, physicsClientId=self.CLIENT)

        local_directions = np.array(
            [
                [1, 0, 0],  # Forward
                [-1, 0, 0],  # Backward
                [0, 1, 0],  # Left
                [0, -1, 0],  # Right
                [0, 0, 1],  # Up
                [0, 0, -1],  # Down
            ]
        )

        max_distance = 4  # meters
        sensor_readings = []

        # Convert quaternion to rotation matrix using NumPy
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)

        for direction in local_directions:
            # Transform local direction to world direction
            world_direction = rot_matrix.dot(direction)

            to_pos = pos + world_direction * max_distance

            ray_result = p.rayTest(pos, to_pos)
            hit_object_id = ray_result[0][0]
            hit_fraction = ray_result[0][2]

            if hit_object_id != -1 and hit_fraction > 0:
                distance = round(hit_fraction * max_distance, 5)
            else:
                distance = 9999  # No obstacle detected within max_distance

            sensor_readings.append(distance)

        return sensor_readings

    ################################################################################

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename("gym_pybullet_drones", "assets/" + self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib["value"])
        L = float(URDF_TREE[0].attrib["arm"])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])
        IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
        IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
        IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib["kf"])
        KM = float(URDF_TREE[0].attrib["km"])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib["length"])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib["radius"])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib["xyz"].split(" ")]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib["max_speed_kmh"])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib["gnd_eff_coeff"])
        PROP_RADIUS = float(URDF_TREE[0].attrib["prop_radius"])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib["drag_coeff_xy"])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib["drag_coeff_z"])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib["dw_coeff_1"])
        DW_COEFF_2 = float(URDF_TREE[0].attrib["dw_coeff_2"])
        DW_COEFF_3 = float(URDF_TREE[0].attrib["dw_coeff_3"])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    ################################################################################

    def _actionSpace(self):
        return _actionSpace_outsource(self)

    ################################################################################

    def _observationSpace(self):
        return _observationSpace_outsource(self)

    ################################################################################

    def _computeObs(self):
        return _computeObs_outsource(self)

    ################################################################################

    def _preprocessAction(self, action):
        return _preprocessAction_outsource(self, action)

    ################################################################################

    def _computeReward(self, Maze_Number, random_number_Start, random_number_Target):
        return _computeReward_outsource(self, Maze_Number, random_number_Start, random_number_Target)

    ################################################################################

    def _computeTerminated(self):
        return _computeTerminated_outsource(self)

    ################################################################################

    def _computeTruncated(self):
        return _computeTruncated_outsource(self)

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """

        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int
            intermediate waypoint for drone

        """
        direction = destination - current_position  # Calculate the direction vector
        distance = np.linalg.norm(direction)  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = direction / distance  # Normalize the direction vector
        next_step = current_position + normalized_direction * step_size  # Calculate the next step
        return next_step

    ##############################################################################!SECTION

    def _compute_potential_fields(self):
        # Parameter
        state = self._getDroneStateVector(0)

        k_rep = 0.01  # Repulsion-Skalierung
        d0 = 1  # Einflussradius für Wände
        Scale_Grid = 0.05  # Skalierung des Grids

        # Erstelle ein Raster mit Potentialwerten
        self.potential_map = np.zeros_like(self.reward_map, dtype=float)

        # Extrahiere Wandpositionen (Indizes der Wandpositionen)
        walls = np.argwhere(self.reward_map == 6)

        Empty_Fields = self.reward_map.size - (walls.size/2)
        self.Area_counter_Max = Empty_Fields
        # Berechne Potentialfeld für jedes Pixel im Grid
        for x in range(self.potential_map.shape[0]):
            for y in range(self.potential_map.shape[1]):
                pos = np.array([x, y])

                # Abstoßungs-Potential (von Wänden)
                U_rep = 0
                for wall in walls:
                    d = np.linalg.norm(pos - wall) * Scale_Grid
                    if 0 < d < d0:
                        U_rep += k_rep * (1 / d - 1 / d0) ** 2

                self.potential_map[x, y] = U_rep

        # Visualisiere das Potentialfeld
        # Create output folder if it doesn't exist
        # output_folder = os.path.join(os.path.dirname(__file__), "potenzial_fields")
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)

        # # # Create and save the plot without displaying
        # plt.ioff()  # Turn off interactive mode
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(self.potential_map, cmap="viridis", origin="lower")
        # plt.colorbar(label="Potential")
        # plt.title("Potentialfeld")
        # plt.xlabel("x")

        # # Generate timestamp and save
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(output_folder, f"potential_field_{timestamp}.png"))
        # plt.close()

        # return self.potential_map

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
        reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_21.csv"

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
        # with open("best_way_map_DQN.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.best_way_map)

        # Save the reward map to a CSV file
        # with open("reward_map_DQN.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.reward_map)
