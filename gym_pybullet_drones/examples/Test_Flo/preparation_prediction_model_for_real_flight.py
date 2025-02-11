from stable_baselines3 import PPO
import numpy as np

pathV1 = '/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/save-02.11.2025_16.29.19_V1_basic-Test_2D-Observation/final_model.zip'
pathV2 = '/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/save-02.11.2025_16.18.45_V2_Squared-Rewards-2D-Observation/final_model.zip'
pathV3 = '/home/florian/Documents/gym-pybullet-drones/save-02.11.2025_13.50.51_Static-Reward-MatrixV3_2D-Observation/final_model.zip'

# Load the model
modelV1 = PPO.load(pathV1)
modelV2 = PPO.load(pathV2)
modelV3 = PPO.load(pathV3)

# Create a sample observation (adjust values based on your observation space)
# If your observation space is like the one in your environment, it might include:
# - position (3 values)
# - velocity (3 values)
# - rotation (3 values)
# - raycast distances (6 values)
observation = np.array([0.3, 9999, 1]) # RayFront, Ray Back, Last Action


# Get prediction
actionV1, _states = modelV1.predict(observation, deterministic=True)
actionV2, _states = modelV2.predict(observation, deterministic=True)
actionV3, _states = modelV3.predict(observation, deterministic=True)

def Übersetzer(action):
    if action == 2:
        return "Stehenbleiben"
    elif action == 0:
        return "Nach vorne fliegen"
    elif action == 1:
        return "Nach hinten fliegen"

print(f"Predicted action V1: {actionV1} --> {Übersetzer(actionV1)}")
print(f"Predicted action V2: {actionV2} --> {Übersetzer(actionV2)}")
print(f"Predicted action V3: {actionV3} --> {Übersetzer(actionV3)}")
