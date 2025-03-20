import contextlib
import csv
import heapq
import logging
import os
import sys
import time
import webbrowser  # Add this import
from collections import deque
from threading import Thread

import dash
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
        target_position=np.array([0, 0, 0]),
        Danger_Threshold_Wall=0.20,
        EPISODE_LEN_SEC=10 * 60,
        dash_active=True,
        REWARD_VERSION="REWARD_VERSION_1",
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

        # Historie der Reward-Komponenten für Balkendiagramm
        self.reward_distribution_history = deque(maxlen=50)  # speichere 50 Einträge

        # Initialize reward and best_way map
        self.reward_map = np.zeros((60, 60), dtype=int)
        self.best_way_map = np.zeros((60, 60), dtype=int)
        self.DASH_ACTIVE = dash_active

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
            target_position=target_position,
            Danger_Threshold_Wall=Danger_Threshold_Wall,
        )

        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

        if self.DASH_ACTIVE:
            self.app = dash.Dash(__name__)
            self.app.layout = html.Div(
                [
                    dcc.Graph(id="live-map"),
                    dcc.Graph(id="reward-bar-chart"),  # <--- Neues Balkendiagramm
                    html.Div(id="current-total-reward"),  # <--- Zeigt den letzten Reward als Text
                    dcc.Interval(id="interval-component", interval=100, n_intervals=0),  # Aktualisierung in ms
                ]
            )

            @self.app.callback([Output("live-map", "figure"), Output("reward-bar-chart", "figure"), Output("current-total-reward", "children")], Input("interval-component", "n_intervals"))
            def update_graph(n):
                # ... existierender Code ...
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Reward Map", "Best Way Map"))
                fig.add_trace(go.Heatmap(z=self.reward_map, colorscale="Viridis", showscale=True, name="Reward Map"), row=1, col=1)
                fig.add_trace(go.Heatmap(z=self.best_way_map, colorscale="Viridis", showscale=True, name="Best Way Map"), row=1, col=2)
                fig.update_layout(height=600, title_text="Maze Training Visualization", showlegend=True)

                bar_chart = go.Figure()
                # zeige die zuletzt gespeicherten Reward-Komponenten
                bar_chart.add_trace(go.Bar(x=list(self.reward_components.keys()), y=list(self.reward_components.values()), marker_color="royalblue"))
                bar_chart.update_layout(title_text="Aktuelle Reward-Komponenten", xaxis_title="Reward-Typ", yaxis_title="Reward-Wert")

                current_reward_text = f"Letzter Reward: {self.last_total_reward:.2f}"

                return fig, bar_chart, current_reward_text

            # Start the Dash server but redirect only the dash logs
            def run_dash_app():
                logging.getLogger("werkzeug").setLevel(logging.ERROR)
                self.app.run_server(debug=False, port=self.port)

            self.dashboard_thread = Thread(target=run_dash_app, daemon=True)
            self.dashboard_thread.start()

            # Open web browser after a short delay to ensure server is running
            def open_browser():
                time.sleep(1)  # Wait for server to start
                webbrowser.open(f"http://localhost:{self.port}")

            Thread(target=open_browser, daemon=True).start()

            ################################################################################

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Discrete
        DREHUNG MATHEMATISCH POSITIV (GEGEN DEN UHRZEIGER)

        1: np.array([[1, 0, 0, 0.99, 0]]), # Fly 90° (Forward)
        2: np.array([[-1, 0, 0, 0.99, 0]]), # Fly 180° (Backward)
        3: np.array([[0, 1, 0, 0.99, 0]]), # Fly 90° (Left)
        4: np.array([[0, -1, 0, 0.99, 0]]), # Fly 270° (Right)
        5: np.array([[0, 0, 0, 0.99, 1/4*np.pi]]), # 45° Left-Turn
        6: np.array([[0, 0, 0, 0.99, -1/4*np.pi]]), # 45° Right-Turn




        """

        return spaces.Discrete(6)

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

    ################################################################################
    # ANCHOR - computeObs
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

    ################################################################################

    # compute Reward siehe BaseAviary

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
