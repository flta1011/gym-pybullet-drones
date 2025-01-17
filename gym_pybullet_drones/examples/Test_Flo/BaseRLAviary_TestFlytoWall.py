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
                 ctrl_freq: int = 240,
                 gui=False,
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
        
        self.ACT_TYPE = act
        self.still_time = 0
        self.EPISODE_LEN_SEC = 20
        
        
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
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
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
                                                    target_pos=np.concatenate([state[0:2], np.array([0.5])]), # same as the current position
                                                    target_rpy=np.array([0,0,state[9]]), # keep current yaw
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
        obs_lower_bound = np.array([lo,lo,lo, #Position
                                     lo,lo,lo, #Roll, pitch, yaw
                                     lo,lo,lo, #Linear velocity
                                     lo,lo,lo, #Angular velocity
                                     0,0,0,0,0, #actual raycast readings
                                     0,0,0,0,0, #previous raycast readings
                                     0,0,0,0] #Last clipped action = Action buffer
                                    )
        
        obs_upper_bound = np.array([hi,hi,hi, #Position
                                     hi,hi,hi, #Roll, pitch, yaw
                                     hi,hi,hi, #Linear velocity
                                     hi,hi,hi, #Angular velocity
                                     9999,9999,9999,9999,9999, # actual raycast readings
                                     9999,9999,9999,9999,9999, # previous raycast readings
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
                5x previous raycast readings (front, back, left, right, top)[9:14]     -> 0 bis 9999
                5x actual raycast readings (front, back, left, right, top)  [14:19]    -> 0 bis 9999
                4x Last clipped action                                      [19:23]    -> 0 bis 2 (da 3 actions)
        """

        
        
        obs = self._getDroneStateVector(0)
        # Select specific values from obs and concatenate them directly
        obs_23 = np.concatenate([
            #obs[0:3],    # Position x,y,z (Drohne kann in echt auch nicht auf die Position zugreifen)
            obs[7:10],   # Roll, pitch, yaw
            obs[10:13],  # Linear velocity
            obs[13:16],  # Angular velocity  
            obs[16:21],  # previous raycast readings
            obs[21:26],  # actual raycast readings
            obs[26:30]   # last clipped action
        ])
        return obs_23
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

    def _computeReward(self): #copied from HoverAviary_TestFlo.py
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        reward = 0
        state = self._getDroneStateVector(0) #erste Drohne
        
        startOfLinearRewardMETER=2.5
        
        #wenn vorheringer Raycastreading = Actual Raycastreading = 9999, dann abstand zu groß -> Vx > 0 (vorne fliegen ist gut, rückwärts fliegen ist schlecht)
        if state[16] == 9999 and state[21] == 9999 and state[10] > 0:
            reward = 10
        elif state[16] == 9999 and state[21] == 9999 and state[10] < 0:
            reward = -10
        #sweet spot größere Belohnung
        
        #sweet spot und stillstand --> größte Belohnung
        
        # zu nah dran größere Straße
        
        # zu nah dran und zurückfliegen -> belohnung
        
        
            
        
        
        #print("Reward:", reward)
        #print("Abstand zur Wand:", state[20])
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

        #Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
        if 0.5 <= state[21] and state[21]<= 0.8 and np.all(np.abs(state[10:13]) < 0.01) and self.still_time > 5:
            return True
        
        return False
    
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
            return True
        
        # TBD wenn die Drone abstürzt, dann auch truncaten
        if state[2] < 0.1:
            return True

        #Wenn an einer Wand gecrashed wird, beenden!
        if (state[21] < 0.15 or state[22] < 0.15 or state[23] < 0.15 or state[24] < 0.15):
            return True
        
        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
       
        
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
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
    
    
        
   
    
    