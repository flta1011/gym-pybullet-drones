import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
#from gym_pybullet_drones.examples.Test_Flo.BaseRLAviary_TestFlo import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.examples.Test_Flo.BaseRLAviary_TestFlo import BaseRLAviary

class DroneEnv(gym.Env):
    def __init__(self):
        
        self.max_steps = 360  # 3 minutes / 0.5s per step
        self.current_step = 0
        self.client = p.connect(p.GUI)  # Connect to PyBullet
        self.drone = p.loadURDF("drone.urdf")  # Load your drone URDF file

    def reset(self):
        self.state = np.zeros(14)
        self.current_step = 0
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf")
        self.drone = p.loadURDF("drone.urdf")
        return self.state
    
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
    
    def render(self, mode='human'):
        pass

# Create the environment
env = DummyVecEnv([lambda: DroneEnv()])

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("dqn_drone")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Load the agent
model = DQN.load("dqn_drone")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
