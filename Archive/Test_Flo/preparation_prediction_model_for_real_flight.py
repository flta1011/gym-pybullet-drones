from stable_baselines3 import PPO
import numpy as np


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
