from stable_baselines3 import PPO


class AIController:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def predict_action(self, observation_space):
        action, _ = self.model.predict(observation_space, deterministic=True)
        return action
