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
        self.still_time = 0
        self.EPISODE_LEN_SEC = 10
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
        0: np.array([1, 0, 0, 0.99]), # Fly Forward
        1: np.array([-1, 0, 0, 0.99]), # Fly Backward
        2: np.array([0, 1, 0, 0.99]), # Fly left
        3: np.array([0, -1, 0, 0.99]), # Fly right

        """
        
        return spaces.Discrete(3)
        
    ################################################################################

    def _preprocessAction(self, action):
        """Preprocesses the action from PPO to drone controls.
        Maps discrete actions to movement vectors.
        """
        # Convert action to movement vector
        # action_to_movement = {
        #     0: np.array([1, 0, 0, 0.99]),  # Forward
        #     1: np.array([-1, 0, 0, 0.99]), # Backward
        #     2: np.array([0, 0, 0, 0.99]),  # Stay
        # }
        
        movement = action
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        if self.ACT_TYPE == ActionType.VEL:
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                if np.linalg.norm(movement[0:3]) != 0:
                    v_unit_vector = movement[0:3] / np.linalg.norm(movement[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                
                rpm[i,:], _, _ = self.ctrl[i].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=state[0:3],
                    target_rpy=np.array([0,0,state[9]]),
                    target_vel=self.SPEED_LIMIT * movement[3] * v_unit_vector
                )
        
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space.
        Simplified observation space with key state variables.
        """
        # Core state variables
        obs_dim = 21  # Position (3), Rotation (3), Velocity (3), Angular vel (3), 
                    # Last action (4), Sensor readings (5)
        
        return spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32
        )


    
    
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

        neg_reward_wall_crash = 0
        pos_reward_target = 0
        
        # tbd FLORIAN Test_Area_Size_x und Test_Area_Size_y in der Observation mitgeben, um die Position der Drohne im Raum zu wissen
        if not hasattr(self, 'reward_grid'):
            self.reward_grid = np.zeros((int(2 * self.Test_Area_Size_x / 0.05), int(2 * self.Test_Area_Size_y / 0.05)))
        
        # wenn die Drohne auf 0,0 gespawned wird, dann ist der Index 0,0 und dann haben wir, wenn Sie nach links fliegt, negative x-Werte und können nicht auf den reward_grid zugreifen (da nur von 0 bis 2*Test_Area_Size_x bzw. 2xTest_Area_Size_y)
        if isinstance(self.INIT_XYZS,np.ndarray):
            if np.all(self.INIT_XYZS[:, :2] == 0):
                state = self._getDroneStateVector(0) #erste Drohne
                x_idx = int((state[0] + self.Test_Area_Size_x) / 0.05)
                y_idx = int((state[1] + self.Test_Area_Size_y) / 0.05)
        
        # Wenn das Labyrinth von 0,0 bis SizeX,SizeY geht und die Drohe irgendwo in diesem Raum gespawned wird, können wir immer auf das reward_grid zugreifen da keine Negativen Werte rauskommen
        if not isinstance(self.INIT_XYZS,np.ndarray):    
            #if self.INIT_XYZS is not None:
                state = self._getDroneStateVector(0) #erste Drohne
                x_idx = int((state[0]) / 0.05)
                y_idx = int((state[1]) / 0.05)
        #print(f"\n\nType of INIT_XYZS: {type(self.INIT_XYZS)}\n\n")
        
        # reward the agent when he is closer to the walls
        state = self._getDroneStateVector(0)
        if (state[20] < 0.5 or state[21] < 0.5 or state[22] < 0.5 or state[23] < 0.5):
            self.reward_grid[x_idx, y_idx] += 1

        
        # negative reward for crashing into walls
        state = self._getDroneStateVector(0)
        if (state[20] < 0.015 or state[21] < 0.015 or state[22] < 0.015 or state[23] < 0.015):
            neg_reward_wall_crash = -1000
        
        #tbd FLORIAN:
        '''einbauen, dass der Reward erst kommt, wenn diese Bedingung länger als 3 Sekunden erfüllt ist --> die Drohne muss dann irgendwie geziehlt zurückfliegen oder Teilreward, dadurch, dass sie schon mal den upper belegt bekommen hat und dann endreward, wenn es für mind. 3 Sekunden belegt ist --> muss dann irgendwie in die Observation eingebaut werden'''    
        # positive reward for reaching the target (raycast_upper !=None and < 1.5)
        # Calculate distance-based reward
        if state[20] == 9999:  # No wall detected in front
            pos_reward_target = 0
        elif state[10] < 0:  # Flying backwards
            pos_reward_target = -100
        elif state[20] < 0.5:  # Too close to wall
            pos_reward_target = -1500
        elif 0.5 <= state[20] <= 0.8:  # Sweet spot
            pos_reward_target = 1800
            # Additional reward for staying still in sweet spot
            if np.all(np.abs(state[10:13]) < 0.01):  # Check if velocity is close to zero
                if not hasattr(self, 'still_time'):
                    self.still_time = 0
                self.still_time += self.CTRL_TIMESTEP
                pos_reward_target += 200 * self.still_time
            else:
                if hasattr(self, 'still_time'):
                    self.still_time = 0
                    pos_reward_target += 200 * self.still_time
        elif 0.8 < state[20] <= 10:  # Linear reward zone
            pos_reward_target = 1500 * (1 - (state[20] - 0.8) / 9.2)
        else:  # Beyond detection range
            pos_reward_target = 0
        
        
        pos_reward_target
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
        state = self._getDroneStateVector(0)

        if self.still_time > 5 and np.all(np.abs(state[10:13]) < 0.01):
            return True

        # Wenn die Zeit abgelaufen ist, beenden!
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
        if (state[20] < 0.5 or state[21] < 0.015 or state[22] < 0.015 or state[23] < 0.015):
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
    
    
        
   
    
    