import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.examples.Test_Flo.BaseAviary_TestFlo import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.examples.Test_Flo.DSLPIDControl_TestFlo import DSLPIDControl

class BaseRLAviary_TestFlo(BaseAviary):
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
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
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
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace_obersvationSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        self._action_to_movement_direction = {
            0: np.array([1, 0, 0, 0.99]), # Up
            1: np.array([-1, 0, 0, 0.99]), # Down
            2: np.array([0, 1, 0, 0.99]), # Yaw left
            3: np.array([0, -1, 0, 0.99]), # Yaw right
        }
        
        low_values = {
            'distance_front': 0,
            'distance_back': 0,
            'distance_left': 0,
            'distance_right': 0,
            'flow_sensor_x': 0,
            'flow_sensor_y': 0,
            'pressure_sensor': 0,
            'accelerometer_x': -np.inf,
            'accelerometer_y': -np.inf,
            'raycast_front': 0,
            'raycast_back': 0,
            'raycast_left': 0,
            'raycast_right': 0,
        }
        
        high_values = {
            'distance_front': np.inf,
            'distance_back': np.inf,
            'distance_left': np.inf,
            'distance_right': np.inf,
            'flow_sensor_x': np.inf,
            'flow_sensor_y': np.inf,
            'pressure_sensor': np.inf,
            'accelerometer_x': np.inf,
            'accelerometer_y': np.inf,
            'raycast_front': np.inf,
            'raycast_back': np.inf,
            'raycast_left': np.inf,
            'raycast_right': np.inf,
        }
        
        # Using .Dict instead of .Box because we have multiple values and can create a dictionary
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=low_values[key], high=high_values[key], shape=(1,), dtype=np.float32)
            for key in low_values
        })
        #
        return self._action_to_movement_direction, self.observation_space
        
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            
            if self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

        """
        
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_21 = np.zeros((self.NUM_DRONES,21))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_21[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:20], obs[20:25]]).reshape(21,)
            ret = np.array([obs_21[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

################################################################################

    def _computeReward(self): #copied from HoverAviary_TestFlo.py
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        
        
        if not hasattr(self, 'reward_grid'):
            self.reward_grid = np.zeros((int(2 * self.Test_Area_Size_x / 0.05), int(2 * self.Test_Area_Size_y / 0.05)))
        
        # wenn die Drohne auf 0,0 gespawned wird, dann ist der Index 0,0 und dann haben wir, wenn Sie nach links fliegt, negative x-Werte und können nicht auf den reward_grid zugreifen (da nur von 0 bis 2*Test_Area_Size_x bzw. 2xTest_Area_Size_y)
        if self.INIT_XYZS == None:
            state = self._getDroneStateVector(0) #erste Drohne
            x_idx = int((state[0] + self.Test_Area_Size_x) / 0.05)
            y_idx = int((state[1] + self.Test_Area_Size_y) / 0.05)
        
        # Wenn das Labyrinth von 0,0 bis SizeX,SizeY geht und die Drohe irgendwo in diesem Raum gespawned wird, können wir immer auf das reward_grid zugreifen da keine Negativen Werte rauskommen
        if self.INIT_XYZS!=None:
            state = self._getDroneStateVector(0) #erste Drohne
            x_idx = int((state[0]) / 0.05)
            y_idx = int((state[1]) / 0.05)
        
        if 0 <= x_idx < self.reward_grid.shape[0] and 0 <= y_idx < self.reward_grid.shape[1]:
            if self.reward_grid[x_idx, y_idx] == 0:
                print(f"Reward given for exploring new spot in discrete world: {x_idx}, {y_idx}")
            self.reward_grid[x_idx, y_idx] = 1
        
        # negative reward for crashing into walls
        state = self._getDroneStateVector(0)
        if (state[20] < 0.015 or state[21] < 0.015 or state[22] < 0.015 or state[23] < 0.015):
            neg_reward_wall_crash = -1000
        
        '''einbauen, dass der Reward erst kommt, wenn diese Bedingung länger als 3 Sekunden erfüllt ist --> die Drohne muss dann irgendwie geziehlt zurückfliegen oder Teilreward, dadurch, dass sie schon mal den upper belegt bekommen hat und dann endreward, wenn es für mind. 3 Sekunden belegt ist --> muss dann irgendwie in die Observation eingebaut werden'''    
        # positive reward for reaching the target (raycast_upper !=None and < 1.5)
        if self.raycast_upper != None and self.raycast_upper < 1.5:
            pos_reward_target = 2000
        
        reward_SUM = np.sum(self.reward_grid) + neg_reward_wall_crash + pos_reward_target
        return reward_SUM

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
         '''einbauen, dass der Reward erst kommt, wenn diese Bedingung länger als 3 Sekunden erfüllt ist --> die Drohne muss dann irgendwie geziehlt zurückfliegen oder Teilreward, dadurch, dass sie schon mal den upper belegt bekommen hat und dann endreward, wenn es für mind. 3 Sekunden belegt ist --> muss dann irgendwie in die Observation eingebaut werden'''  
        # #target errreicht
        # if self.raycast_upper != None and self.raycast_upper < 1.5:
        #     pos_reward_target = 2000
        
        
        return False
    
    ################################################################################
    
    def _computeTruncated(self): #coppied from HoverAviary_TestFlo.py
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        # Truncate when the drone is too tilted
        state = self._getDroneStateVector(0)
        if abs(state[7]) > .4 or abs(state[8]) > .4: 
            return True
        
        
        #Wenn an einer Wand gecrashed wird, beenden!
        if (state[20] < 0.015 or state[21] < 0.015 or state[22] < 0.015 or state[23] < 0.015):
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

    def reset(self):
        self.state = np.zeros(14)
        self.current_step = 0
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf")
        self.drone = p.loadURDF("drone.urdf")
        return self.state
    
    def _perform_raycast(self):
        # Perform raycasting in four directions
        ray_length = 10  # Define the length of the rays
        front_ray = p.rayTest([0, 0, 0], [ray_length, 0, 0])
        back_ray = p.rayTest([0, 0, 0], [-ray_length, 0, 0])
        left_ray = p.rayTest([0, 0, 0], [0, ray_length, 0])
        right_ray = p.rayTest([0, 0, 0], [0, -ray_length, 0])
        
        return {
            'front': front_ray[0][2],  # Distance to the first hit object
            'back': back_ray[0][2],
            'left': left_ray[0][2],
            'right': right_ray[0][2],
        }
        
    def step(self, action):
        # Apply action
        direction = self._action_to_direction[action]
        # Perform raycasting
        raycast_results = self._perform_raycast()
        
        # Update observation with raycast results
        observation = {
            'distance_front': 0,  # Replace with actual sensor data
            'distance_back': 0,   # Replace with actual sensor data
            'distance_left': 0,   # Replace with actual sensor data
            'distance_right': 0,  # Replace with actual sensor data
            'flow_sensor_x': 0,   # Replace with actual sensor data
            'flow_sensor_y': 0,   # Replace with actual sensor data
            'pressure_sensor': 0, # Replace with actual sensor data
            'accelerometer_x': 0, # Replace with actual sensor data
            'accelerometer_y': 0, # Replace with actual sensor data
            'raycast_front': raycast_results['front'],
            'raycast_back': raycast_results['back'],
            'raycast_left': raycast_results['left'],
            'raycast_right': raycast_results['right'],
        }
        
        reward = 0  # Define your reward function
        done = False  # Define your termination condition
        info = {}  # Additional information
        
        return observation, reward, done, info
    
    