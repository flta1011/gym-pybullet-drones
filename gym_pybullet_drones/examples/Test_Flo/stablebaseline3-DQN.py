import gym
from gym import spaces
import numpy as np
import pybullet as p
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # Define action and observation space, 0: forward, 1:backward, 2:left, 3:right
        self.action_space = spaces.Discrete(4)
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will fly in if that action is taken. Coppied from the 20241109_1300_MS_pid_velocity-File
        """
        self._action_to_direction = {
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
            'accelerometer_z': -np.inf,
            'gyroscope_x': -np.inf,
            'gyroscope_y': -np.inf,
            'gyroscope_z': -np.inf,
            'other': -np.inf
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
            'accelerometer_z': np.inf,
            'gyroscope_x': np.inf,
            'gyroscope_y': np.inf,
            'gyroscope_z': np.inf,
            'other': np.inf
        }
        
        self.observation_space = spaces.Box(
            low=np.array([low_values[key] for key in low_values]),  # Lower bounds for each dimension
            high=np.array([high_values[key] for key in high_values]),  # Upper bounds for each dimension
            dtype=np.float32  # Data type of the elements
        )
        
        self.observation_dict = {
            'distance_front': 0,
            'distance_back': 1,
            'distance_left': 2,
            'distance_right': 3,
            'flow_sensor_x': 4,
            'flow_sensor_y': 5,
            'pressure_sensor': 6,
            'accelerometer_x': 7,
            'accelerometer_y': 8,
            'accelerometer_z': 9,
            'gyroscope_x': 10,
            'gyroscope_y': 11,
            'gyroscope_z': 12,
            'other': 13
        }
        
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
        # Apply action to the drone
        # For example, action could be thrust values for the motors
        # p.applyExternalForce(self.drone, linkIndex, forceObj, posObj, flags)
        
        # Simulate physics
        p.stepSimulation(self.client)
        
        # Get observation from PyBullet
        position, orientation = p.getBasePositionAndOrientation(self.drone)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.drone)
        
        # Update state with PyBullet data
        self.state = np.array([
            # ...existing code...
            position[0], position[1], position[2],  # Position x, y, z
            orientation[0], orientation[1], orientation[2], orientation[3],  # Orientation quaternion
            linear_velocity[0], linear_velocity[1], linear_velocity[2],  # Linear velocity x, y, z
            angular_velocity[0], angular_velocity[1], angular_velocity[2]  # Angular velocity x, y, z
        ])
        
        reward = 0
        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {}
        return self.state, reward, done, info
    
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
