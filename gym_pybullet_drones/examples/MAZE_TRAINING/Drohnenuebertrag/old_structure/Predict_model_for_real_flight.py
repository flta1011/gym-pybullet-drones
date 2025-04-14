#### actual version
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

# case "O8":  # 7 Kanäle für CNN-DQN
#     """
#     Returns the observation space for the CNN-DQN model.
#     The observation space is a Box with shape (7, grid_size, grid_size) containing:
#     - Channel 1: SLAM map (values in [0,255])
#     - Channel 2: X-Position (values in [-inf,inf])
#     - Channel 3: Y-Position (values in [-ing,inf])
#     - Channel 4: Raycast readings (values in [0,4])
#     - Channel 5: Interest Values (values in [0,32400])
#     - Channel 6: n last Clipped Actions (values in [0, 3])
#     """
#     last_actions_size = self.last_actions.shape[0]  # Number of last clipped actions

#     # Define the low and high bounds for the flattened observation
#     low = np.concatenate(
#         (
#             np.array([-np.inf], dtype=np.float32),  # X position
#             np.array([-np.inf], dtype=np.float32),  # Y position
#             np.zeros(4, dtype=np.float32),  # Raycast readings (values in [0, 4])
#             np.zeros(4, dtype=np.float32),  # Interest values
#             np.zeros(last_actions_size, dtype=np.float32),  # Last clipped actions
#         )
#     )

#     high = np.concatenate(
#         (
#             np.array([np.inf], dtype=np.float32),  # X position
#             np.array([np.inf], dtype=np.float32),  # Y position
#             np.full(4, 4, dtype=np.float32),  # Raycast readings (values in [0, 4])
#             np.full(4, 32400, dtype=np.float32),  # Interest values
#             np.full(last_actions_size, 6, dtype=np.float32),  # Last clipped actions
#         )
#     )

#     # Return the flattened observation space
#     return spaces.Box(low=low, high=high, dtype=np.float32)

# case "O9":  # 7 Kanäle für CNN-DQN
#     """
#     Returns the observation space for the CNN-DQN model.
#     The observation space is a Box with shape (7, grid_size, grid_size) containing:
#     - Channel 1: Normalized SLAM map (values in [0,1])
#     - Channel 2: Normalized x position (values in [0,1])
#     - Channel 3: Normalized y position (values in [0,1])
#     - Channel 6: last Clipped Action (values in [-1,1])
#     - Channel 7: second Last Clipped Action (values in [-1,1])
#     - Channel 8: third Last Clipped Action (values in [-1,1])
#     """
#     grid_size = int(self.slam.cropped_map_size_grid)

#     observationSpace = spaces.Dict(
#         {
#             "image": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 1), dtype=np.uint8),  # Grayscale image
#             "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
#             "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
#             "raycast": spaces.Box(low=0, high=4, shape=(4,), dtype=np.float32),
#             "interest_values": spaces.Box(low=0, high=32400, shape=(4,), dtype=np.uint8),
#             "last_clipped_actions": spaces.Box(low=0, high=6, shape=(self.last_actions.shape[0],), dtype=np.float32),
#         }
#     )

#     return observationSpace


# case "A2":  # Vier Richtungen

#     return spaces.Discrete(4)

# case "A3":  # Vier Richtungen, kontinuierlich
#     """For SAC, we use a continuous action space with:
#     - Linear velocity in x (-self.VelocityScale to self.VelocityScale)
#     - Linear velocity in y (-self.VelocityScale to self.VelocityScale)
#     """
#     return spaces.Box(
#         low=np.array([-self.VelocityScale, -self.VelocityScale]),
#         high=np.array([self.VelocityScale, self.VelocityScale]),
#         dtype=np.float32,
#     )


#### Old version
def get_PPO_Predcitions_1D_Observation(rayFront, model_path):
    """Get predictions from 3 different PPO models for 1D observations.

    Args:
        rayFront (float): Front ray distance measurement

    Returns:
        tuple: Predictions from models V1
    """
    # Load the models
    modelV1 = PPO.load(model_path)

    # Get predictions
    actionV1, _states = modelV1.predict(rayFront, deterministic=True)

    def Übersetzer(action):
        if action == 2:
            return ("x", 0)
        elif action == 0:
            return ("x", 1)
        elif action == 1:
            return ("x", -1)

    return actionV1


#### New version
import numpy as np
from stable_baselines3 import PPO

pathV1 = "/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/save-02.11.2025_16.29.19_V1_basic-Test_2D-Observation/final_model.zip"
pathV2 = "/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/save-02.11.2025_16.18.45_V2_Squared-Rewards-2D-Observation/final_model.zip"
pathV3 = "/home/florian/Documents/gym-pybullet-drones/save-02.11.2025_13.50.51_Static-Reward-MatrixV3_2D-Observation/final_model.zip"

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
observation = np.array([0.3, 9999, 1])  # RayFront, Ray Back, Last Action

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
