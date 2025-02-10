import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.examples.Test_Flo.BaseAviary_TestFlytoWall import BaseAviary_TestFlytoWall
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.examples.Test_Flo.DSLPIDControl_TestFlo import DSLPIDControl

from stable_baselines3.common.policies import ActorCriticPolicy

class BaseRLAviary_TestFlytoWall(BaseAviary_TestFlytoWall):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 # In BaseAviary hinzugefügt
                #  Test_Area_Size_x: int = 10, #hoffentlich 10 Meter, später Größe der Map
                #  Test_Area_Size_y: int = 10, #hoffentlich 10 Meter, später Größe der Map
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 60,
                 reward_and_action_change_freq: int = 10,
                 gui=False,
                 user_debug_gui=False,
                 record=False,
                 act: ActionType=ActionType.VEL
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
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        self.reward_and_action_change_freq = reward_and_action_change_freq
        self.ACT_TYPE = act
        self.still_time = 0
        self.EPISODE_LEN_SEC = 5*60 #increased from 20 to 100
        
        
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        else:
            print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
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
                         user_debug_gui=user_debug_gui, # Remove of RPM sliders from all single agent learning aviaries
                         )
        
        #### Set a limit on the maximum target speed ###############
        if  act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

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
            temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=np.concatenate([state[0:1],np.array([0]), np.array([0.5])]), # same as the current position on X, but should stay 0 on y and z = 0.5
                                                    target_rpy=np.array([0,0,0]), # keep orientation to base
                                                    target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
            rpm[k,:] = temp
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space.
        Simplified observation space with key state variables.
        
        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.
            
            Information of the self._getDroneStateVector:
                ndarray
                (25,)-shaped array of floats containing the state vector of the n-th drone. The state vector includes:

                3x Position (x, y, z) [0:3]                -> -np.inf bis np.inf
                4x Quaternion (qx, qy, qz, qw) [3:7]       -> nicht verwendet
                3x Roll, pitch, yaw (r, p, y) [7:10]       -> -np.inf bis np.inf
                3x Linear velocity (vx, vy, vz) [10:13]    -> -np.inf bis np.inf
                3x Angular velocity (wx, wy, wz) [13:16]     -> -np.inf bis np.inf
                4x Last clipped action [16:20]             -> 0 bis 2 (da 3 actions)
                5x Raycast readings (front, back, left, right, top) [20:25] -> 0 bis 9999
                
        """
       
                    
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([#lo,lo,lo, #Position, weil die Drohne nicht auf die Position zugreifen kann
                                     lo,lo,lo, #Roll, pitch, yaw
                                     lo,lo,lo, #Linear velocity
                                     lo,lo,lo, #Angular velocity
                                     #0,0,0,0,0, #previous raycast readings
                                     0,0,0,0,0, #actual raycast readings
                                     0,0,0,0] #Last clipped action = Action buffer
                                    )
        
        obs_upper_bound = np.array([#hi,hi,hi, #Position, weil die Drohne nicht auf die Position zugreifen kann
                                     hi,hi,hi, #Roll, pitch, yaw
                                     hi,hi,hi, #Linear velocity
                                     hi,hi,hi, #Angular velocity
                                     #9999,9999,9999,9999,9999, # previous raycast readings
                                     9999,9999,9999,9999,9999, # actual raycast readings
                                     2,2,2,2] #Last clipped action = Action buffer
                                   
                                    )
        
       
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float32
        )


    
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.
            
            Information of the self._getDroneStateVector:
                ndarray
                (25,)-shaped array of floats containing the state vector of the n-th drone. The state vector includes:

                3x Roll, pitch, yaw (r, p, y)                               [0:3]      -> -np.inf bis np.inf
                3x Linear velocity (vx, vy, vz)                             [3:6]      -> -np.inf bis np.inf
                3x Angular velocity (wx, wy, wz)                            [6:9]      -> -np.inf bis np.inf
                5x actual raycast readings (front, back, left, right, top)  [9:14]    -> 0 bis 9999
                4x Last clipped action                                      [14:18]    -> 0 bis 2 (da 3 actions)
        """

        
        
        obs = self._getDroneStateVector(0)
        # Select specific values from obs and concatenate them directly
        obs_18 = np.concatenate([
            #obs[0:3],    # Position x,y,z (Drohne kann in echt auch nicht auf die Position zugreifen)
            obs[7:10],   # Roll, pitch, yaw
            obs[10:13],  # Linear velocity
            obs[13:16],  # Angular velocity  
            #obs[16:21],  # previous raycast readings, ist für das RL vielleicht gar nicht so hilfreich
            obs[21:26],  # actual raycast readings
            obs[26:30]   # last clipped action
        ])
        return obs_18
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
    # ANCHOR - def computeReward
    def _computeReward(self): #copied from HoverAviary_TestFlo.py
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        reward = 0
        state = self._getDroneStateVector(0) #erste Drohne
        
        
        
        # #wenn vorheringer Raycastreading = Actual Raycastreading = 9999, dann abstand zu groß -> Vx > 0 (vorne fliegen ist gut, rückwärts fliegen ist schlecht)
        # if state[10] > 0:
        #     reward = 5
        # elif state[10] < 0:
        #     reward = -3 # von -5 auf -3 reduziert (9.2)
     
        # self.Ende_Crash = 0.2
        # self.Beginn_sweetspot = 0.5
        # self.Ende_sweetspot = 0.6
        
        # #im Stillstand und nicht im sweetspot (weiter als 0,8m von der Wand entfernt): leicht negativ: Bestrafung für Stillstand
        # if (state[10] < 0.01) and state[21] > self.Ende_sweetspot:
        #     reward = -0.5 # von -2,5 auf -0,5 reduziert
        # #im Stillstand und im sweetspot (zwischen 0,5m und 0,8m von der Wand entfernt): Belohnung für Stillstand im Sweetspot
        # elif (state[10] < 0.01) and state[21] > self.Beginn_sweetspot and state[21] < self.Ende_sweetspot:
        #     reward = 20 # von 50 auf 20 reduziert (9.2)
            
       
        #  # zu nah dran und vorwärts fliegen: Bestrafung; zu nah dran und zurückfliegen -> Belohnung (näher als 0,5 aber weiter weg als 0,20)
        # if state[21] < self.Beginn_sweetspot and state[21] > self.Ende_Crash and state[10] > 0:
        #     reward = -20
        # elif state[21] < self.Beginn_sweetspot and state[21] > self.Ende_Crash and state[10] < 0:
        #     reward = 20
        
        # # zu nah dran aka. gecrasht: maximale Bestrafung
        # if state[21] < self.Ende_Crash:
        #     reward = -300    # reward von -1000 auf -300 verringert, da die Drohne sonst nicht mehr lernt bzw. durch den Zusammenprall insgesamt negatives gesamtergebnis bekommt und dann ableitet, dass alles schlecht war und dann danach nur noch stehenbleibt

        '''Anpassung: Umstellung auf die Belohnung auf die Gewählte Action'''
        #wenn vorheringer Raycastreading = Actual Raycastreading = 9999, dann abstand zu groß -> Vx > 0 (vorne fliegen ist gut, rückwärts fliegen ist schlecht)
        #self.action[0][0] ist der Wert der Velocity in SOLL: x-Richtung
        if self.action[0][0] == 1: 
            reward = 5
        elif self.action[0][0] == -1:
            reward = -3 # von -5 auf -3 reduziert (9.2)
     
        self.Ende_Crash = 0.2
        self.Beginn_sweetspot = 0.4
        self.Ende_sweetspot = 0.7
        
        #im Stillstand und nicht im sweetspot (weiter als 0,8m von der Wand entfernt): leicht negativ: Bestrafung für Stillstand
        if self.action[0][0] == 0 and state[21] > self.Ende_sweetspot:
            reward = -0.5 # von -2,5 auf -0,5 reduziert
        #im Stillstand und im sweetspot (zwischen 0,5m und 0,8m von der Wand entfernt): Belohnung für Stillstand im Sweetspot
        elif self.action[0][0] == 0 and state[21] > self.Beginn_sweetspot and state[21] < self.Ende_sweetspot:
            reward = 20 # von 50 auf 20 reduziert (9.2)
            
       
         # zu nah dran und vorwärts fliegen: Bestrafung; zu nah dran und zurückfliegen -> Belohnung (näher als 0,5 aber weiter weg als 0,20)
        if self.action[0][0] == 1 and state[21] < self.Beginn_sweetspot and state[21] > self.Ende_Crash and state[10] > 0:
            reward = -20
        elif self.action[0][0] == -1 and state[21] < self.Beginn_sweetspot and state[21] > self.Ende_Crash and state[10] < 0:
            reward = 20
        
        # zu nah dran aka. gecrasht: maximale Bestrafung
        if state[21] <= self.Ende_Crash:
            reward = -300    # reward von -1000 auf -300 verringert, da die Drohne sonst nicht mehr lernt bzw. durch den Zusammenprall insgesamt negatives gesamtergebnis bekommt und dann ableitet, dass alles schlecht war und dann danach nur noch stehenbleibt
        
        
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
        if self.Beginn_sweetspot <= state[21] and state[21]<= self.Ende_sweetspot and np.all(np.abs(state[10:13]) < 0.001):
            self.still_time += (1/self.reward_and_action_change_freq)# Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
        else:
            self.still_time = 0.0 # Reset timer to 0 seconds

        #Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
        if self.Beginn_sweetspot <= state[21] and state[21]<= self.Ende_sweetspot and self.still_time >= 5:
            Grund_Terminated = "Drohne ist im sweet spot für 5 sekunden Stillstand und wird erfolgreich beendet"
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
            return True, Grund_Truncated
        
        # TBD wenn die Drone abstürzt, dann auch truncaten
        if state[2] < 0.1: #state[2] ist z_position der Drohne
            Grund_Truncated = "Crash"
            return True, Grund_Truncated

        #Wenn an einer Wand gecrashed wird, beenden!
        if (state[21] <= 0.2 or state[22] <= 0.2 or state[23] <= 0.2 or state[24] <= 0.2):
            Grund_Truncated = "Zu nah an der Wand"
            return True, Grund_Truncated
        
        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
 
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    #########################################################################################

    # def reset(self):
    #     self.state = np.zeros(14)
    #     self.current_step = 0
    #     p.resetSimulation(self.client)
    #     p.setGravity(0, 0, -9.8, physicsClientId=self.client)
    #     p.loadURDF("plane.urdf")
    #     self.drone = p.loadURDF("drone.urdf")
    #     return self.state
    
    
        
   
    
    