import numpy as np
from gymnasium import spaces


def _observationSpace(self):
    """Returns the observation space.
    Simplified observation space with key state variables.

    10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

    Returns
    -------
    ndarray
        A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,21) depending on the observation type.

        Information of the self._getDroneStateVector:
            ndarray
            1x Raycast reading (forward) [21]          -> 0 bis 9999

    """

    lo = -np.inf
    hi = np.inf
    obs_lower_bound = np.array([-99, -99, -2 * np.pi, 0, 0, 0, 0, 0])  # x,y,yaw, Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

    obs_upper_bound = np.array([99, 99, 2 * np.pi, 9999, 9999, 9999, 9999, 9999])  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

    return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
