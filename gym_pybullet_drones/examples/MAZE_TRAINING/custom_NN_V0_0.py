# Importiere benötigte Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom Feature Extractor basierend auf unserer CNN-Architektur
class CustomNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=6):
        """
        observation_space: Erwartet einen Tensor der Form (channels, height, width), z. B. (5, 60, 60).
        features_dim: Dimension der extrahierten Features.
        """
        super(CustomNNFeatureExtractor, self).__init__(observation_space, features_dim)
        # Definiere deine CNN-Schichten (angepasst für 5 Eingangskanäle)

        self.nn = nn.Sequential(nn.Sequential(nn.Flatten(), nn.Linear(observation_space.shape[0], 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU()))
        # Bestimme die Größe der flachen Feature Map
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.nn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        # observations hat die Form [batch, channels, height, width]
        return self.linear(self.nn(observations))
