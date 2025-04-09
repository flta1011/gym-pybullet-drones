# Importiere benötigte Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom Feature Extractor basierend auf unserer CNN-Architektur
class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=6):
        """
        observation_space: Erwartet einen Tensor der Form (channels, height, width), z. B. (5, 60, 60).
        features_dim: Dimension der extrahierten Features.
        """
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        # Definiere deine CNN-Schichten (angepasst für 5 Eingangskanäle)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Bestimme die Größe der flachen Feature Map
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        # observations hat die Form [batch, channels, height, width]
        return self.linear(self.cnn(observations))
