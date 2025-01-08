import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.examples.Test_Flo.BaseAviary_TestFlo import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.examples.Test_Flo.DSLPIDControl_TestFlo import DSLPIDControl

from stable_baselines3.common.policies import ActorCriticPolicy

class BaseRLAviary_TestFlo(BaseAviary):
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
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL  #wurde rausgenommen
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
        self.Test_Area_Size_x = 10
        self.Test_Area_Size_y = 10
        #### Set a limit on the maximum target speed ###############
        if  act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################
    # TBD
    # def _addObstacles(self):
    #     """Add obstacles to the environment.

    #     Only if the observation is of type RGB, 4 landmarks are added.
    #     Overrides BaseAviary's method.

    #     """
    #     if self.OBS_TYPE == ObservationType.RGB:
    #         p.loadURDF("block.urdf",
    #                    [1, 0, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("cube_small.urdf",
    #                    [0, 1, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("duck_vhacd.urdf",
    #                    [-1, 0, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("teddy_vhacd.urdf",
    #                    [0, -1, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #     else:
    #         pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Discrete
        0: np.array([1, 0, 0, 0.99]), # Up
        1: np.array([-1, 0, 0, 0.99]), # Down
        2: np.array([0, 1, 0, 0.99]), # Yaw left
        3: np.array([0, -1, 0, 0.99]), # Yaw right

        """
        
        return spaces.Discrete(4)
        
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
        # Print the shape of the action array
        print("Shape of action:", action.shape)
        # die Action sind die 4 möglichen Bewegungen, die die Drohne machen kann
        # action_to_movement_direction = {
        #     0: np.array([1, 0, 0, 0.99]), # Up
        #     1: np.array([-1, 0, 0, 0.99]), # Down
        #     2: np.array([0, 1, 0, 0.99]), # Yaw left
        #     3: np.array([0, -1, 0, 0.99]), # Yaw right
        # }
        # Loop through the action array and extract the target TBD alex
        action = np.atleast_2d(action)
        for k in range(action.shape[0]):
            # Verify the index k
            if k < 0 or k >= action.shape[0]:
                raise IndexError(f"Index k={k} is out of bounds for axis 0 with size {action.shape[0]}")

            # Extract the target from the action array
            target = action[k, :]
            print(f"Target for index {k}:", target)
            
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
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., and ndarray of shape (NUM_DRONES, 20).

        """
    
        low_values = {
            0: -np.inf,  # x
            1: -np.inf,  # y
            2: 0,        # z
            3: -np.pi,   # Roll
            4: -np.pi,   # Pitch
            5: -np.pi,   # Yaw
            6: -np.inf,  # Vx
            7: -np.inf,  # Vy
            8: -np.inf,  # Vz
            9: -np.inf,  # angular_velocity_x
            10: -np.inf, # angular_velocity_y
            11: -np.inf, # angular_velocity_z
            12: 0,       # Last_Clipped_action_A0
            13: 0,       # Last_Clipped_action_A1
            14: 0,       # Last_Clipped_action_A2
            15: 0,       # Last_Clipped_action_A3
            # 16: 0,      # distance_front
            # 17: 0,      # distance_back
            # 18: 0,      # distance_left
            # 19: 0,      # distance_right
            # 20: 0,      # flow_sensor_x
            # 21: 0,      # flow_sensor_y
            # 22: 0,      # pressure_sensor
            # 23: -np.inf, # accelerometer_x
            # 24: -np.inf, # accelerometer_y
            16: 0,       # raycast_front
            17: 0,       # raycast_back
            18: 0,       # raycast_left
            19: 0,       # raycast_right
            20: 0        # raycast_top
        }

        # low_values = {
        #     'x': -np.inf,
        #     'y': -np.inf,
        #     'z': 0,
        #     'Roll': -np.pi,
        #     'Pitch': -np.pi,
        #     'Yaw': -np.pi,
        #     'Vx': -np.inf,
        #     'Vy': -np.inf,
        #     'Vz': -np.inf,
        #     'angular_velocity_x': -np.inf,
        #     'angular_velocity_y': -np.inf,
        #     'angular_velocity_z': -np.inf,
        #     'Last_Clipped_action_A0': 0,
        #     'Last_Clipped_action_A1': 0,
        #     'Last_Clipped_action_A2': 0,
        #     'Last_Clipped_action_A3': 0,
        #     # 'distance_front': 0,
        #     # 'distance_back': 0,
        #     # 'distance_left': 0,
        #     # 'distance_right': 0,
        #     # 'flow_sensor_x': 0,
        #     # 'flow_sensor_y': 0,
        #     # 'pressure_sensor': 0,
        #     # 'accelerometer_x': -np.inf,
        #     # 'accelerometer_y': -np.inf,
        #     'raycast_front': 0,
        #     'raycast_back': 0,
        #     'raycast_left': 0,
        #     'raycast_right': 0,
        #     'raycast_top': 0
        # }
        high_values = {
            0: np.inf,  # x
            1: np.inf,  # y
            2: np.inf,  # z
            3: np.pi,   # Roll
            4: np.pi,   # Pitch
            5: np.pi,   # Yaw
            6: np.inf,  # Vx
            7: np.inf,  # Vy
            8: np.inf,  # Vz
            9: np.inf,  # angular_velocity_x
            10: np.inf, # angular_velocity_y
            11: np.inf, # angular_velocity_z
            12: 3,      # Last_Clipped_action_A0
            13: 3,      # Last_Clipped_action_A1
            14: 3,      # Last_Clipped_action_A2
            15: 3,      # Last_Clipped_action_A3
            # 16: np.inf, # distance_front
            # 17: np.inf, # distance_back
            # 18: np.inf, # distance_left
            # 19: np.inf, # distance_right
            # 20: np.inf, # flow_sensor_x
            # 21: np.inf, # flow_sensor_y
            # 22: np.inf, # pressure_sensor
            # 23: np.inf, # accelerometer_x
            # 24: np.inf, # accelerometer_y
            16: np.inf, # raycast_front
            17: np.inf, # raycast_back
            18: np.inf, # raycast_left
            19: np.inf, # raycast_right
            20: np.inf  # raycast_top
        }

        #self.policy_class = ActorCriticPolicy
        # high_values = {
        #     'x': np.inf,
        #     'y': np.inf,
        #     'z': np.inf,
        #     'Roll': np.pi,
        #     'Pitch': np.pi,
        #     'Yaw': np.pi,
        #     'Vx': np.inf,
        #     'Vy': np.inf,
        #     'Vz': np.inf,
        #     'angular_velocity_x': np.inf,
        #     'angular_velocity_y': np.inf,
        #     'angular_velocity_z': np.inf,
        #     'Last_Clipped_action_A0': 3,
        #     'Last_Clipped_action_A1': 3,
        #     'Last_Clipped_action_A2': 3,
        #     'Last_Clipped_action_A3': 3,
        #     # 'distance_front': np.inf,
        #     # 'distance_back': np.inf,
        #     # 'distance_left': np.inf,
        #     # 'distance_right': np.inf,
        #     # 'flow_sensor_x': np.inf,
        #     # 'flow_sensor_y': np.inf,
        #     # 'pressure_sensor': np.inf,
        #     # 'accelerometer_x': np.inf,
        #     # 'accelerometer_y': np.inf,
        #     'raycast_front': np.inf,
        #     'raycast_back': np.inf,
        #     'raycast_left': np.inf,
        #     'raycast_right': np.inf,
        #     'raycast_top': np.inf
        # }

        # low_values = [
        #     -np.inf,  # x
        #     -np.inf,  # y
        #     0,        # z
        #     -np.pi,   # Roll
        #     -np.pi,   # Pitch
        #     -np.pi,   # Yaw
        #     -np.inf,  # Vx
        #     -np.inf,  # Vy
        #     -np.inf,  # Vz
        #     -np.inf,  # angular_velocity_x
        #     -np.inf,  # angular_velocity_y
        #     -np.inf,  # angular_velocity_z
        #     0,        # Last_Clipped_action_A0
        #     0,        # Last_Clipped_action_A1
        #     0,        # Last_Clipped_action_A2
        #     0,        # Last_Clipped_action_A3
        #     0,        # raycast_front
        #     0,        # raycast_back
        #     0,        # raycast_left
        #     0,        # raycast_right
        #     0         # raycast_top
        # ]

        # high_values = [
        #     np.inf,  # x
        #     np.inf,  # y
        #     np.inf,  # z
        #     np.pi,   # Roll
        #     np.pi,   # Pitch
        #     np.pi,   # Yaw
        #     np.inf,  # Vx
        #     np.inf,  # Vy
        #     np.inf,  # Vz
        #     np.inf,  # angular_velocity_x
        #     np.inf,  # angular_velocity_y
        #     np.inf,  # angular_velocity_z
        #     3,       # Last_Clipped_action_A0
        #     3,       # Last_Clipped_action_A1
        #     3,       # Last_Clipped_action_A2
        #     3,       # Last_Clipped_action_A3
        #     np.inf,  # raycast_front
        #     np.inf,  # raycast_back
        #     np.inf,  # raycast_left
        #     np.inf,  # raycast_right
        #     np.inf   # raycast_top
        # ]
        
        # Using .Dict instead of .Box because we have multiple values and can create a dictionary
        ### TBD TBD TBD Weil die Learn fkt nicht ausgeführt wird, weil mit strings nicht hochzählen
        # Reihenfolge auch weird. Sie Bilder WhatsAPP von Kameraman Alex
        # 09.12.2024 23:50 

        # Convert the low and high values to numpy arrays
        obs_lower_bound = np.array([low_values[key] for key in low_values], dtype=np.float32)
        obs_upper_bound = np.array([high_values[key] for key in high_values], dtype=np.float32)

        # self.observation_space = spaces.Dict({
        #     key: spaces.Box(low=low_values[key], high=high_values[key], shape=(1,), dtype=np.float32)
        #     for key in low_values
        # })
        
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        

        # Initializing the observation bounds
        # obs_lower_bound = np.array([low_values for _ in range(self.NUM_DRONES)])  # Use low_values for lower bound
        # obs_upper_bound = np.array([high_values for _ in range(self.NUM_DRONES)])  # Use high_values for upper bound
        # act_lo = -1
        # act_hi = +1
        # for i in range(self.ACTION_BUFFER_SIZE):
        #     if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
        #         obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
        #         obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
        # return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)



        
    
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

        """
        # TBD letzer Wert ist weg
        obs_21 = np.zeros(21)
        obs = self._getDroneStateVector(0)
        obs_21[:21] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:20], obs[20:25]])
     
        return obs_21
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
        
        # tbd FLORIAN Test_Area_Size_x und Test_Area_Size_y in der Observation mitgeben, um die Position der Drohne im Raum zu wissen
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
        
        #tbd FLORIAN:
        '''einbauen, dass der Reward erst kommt, wenn diese Bedingung länger als 3 Sekunden erfüllt ist --> die Drohne muss dann irgendwie geziehlt zurückfliegen oder Teilreward, dadurch, dass sie schon mal den upper belegt bekommen hat und dann endreward, wenn es für mind. 3 Sekunden belegt ist --> muss dann irgendwie in die Observation eingebaut werden'''    
        # positive reward for reaching the target (raycast_upper !=None and < 1.5)
        if self.ray_results[4] < 1.5:
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
        if self.ray_results[4] < 1.5:
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

    # def reset(self):
    #     self.state = np.zeros(14)
    #     self.current_step = 0
    #     p.resetSimulation(self.client)
    #     p.setGravity(0, 0, -9.8, physicsClientId=self.client)
    #     p.loadURDF("plane.urdf")
    #     self.drone = p.loadURDF("drone.urdf")
    #     return self.state
    
    
        
   
    
    