import numpy as np


def _computeObs(self):
    """Returns the current observation of the environment.
    10.2.25: deutlich vereinfachte Observation Space, damit es für den PPO einfacher ist, die Zuammenhänge zwischen den relevanten Observations und dem dafür erhaltenen Reward zu erkennen.

    28.2.25: vereinfachte Observation Space: in jede Richtung vorne, hinten, links, rechts, oben folgende Outputs möglich
    - 0: zu nahe an der Wand,
    - 1: Wand kommt näher,
    - 2: safe Distance,
    - 9999: Sensor oben frei

    Returns (28.2.25):
    -------
    ndarray
        A Box() of shape (NUM_DRONES,5) -> vorne, hinten, links, rechts, oben (1,9999)
        -> 0: zu nahe an der Wand,
        -> 1: Wand kommt näher,
        -> 2: safe Distance,
        -> 9999: Sensor oben frei
    """

    state = self._getDroneStateVector(0)

    # Select specific values from obs and concatenate them directly
    obs = [state[21], state[22], state[23], state[24]]  # Raycast reading forward, Raycast reading backward, Raycast reading left, Raycast reading right, Raycast reading up

    # NOTE - nachfolgend auf vereinfachte Observation Space umgestellt (28.2.25):
    # Modify observation based on distance thresholds
    modified_obs = []

    # NOTE - neue Tests mit X,Y, Yaw Position der Drohne (28.2.25) übergeben
    modified_obs.append(round(state[0], 3))  # x-Position
    modified_obs.append(round(state[1], 3))  # y-Position
    modified_obs.append(round(state[9], 3))  # Yaw-Position

    # abstände anhängen mit 3 Nachkommastellen
    for distance in obs:
        modified_obs.append(round(distance, 3))

    # raycast oben noch anhängen
    if state[25] < 1:
        modified_obs.append(1)
    else:
        modified_obs.append(9999)

    return np.array(modified_obs, dtype=np.float32)  # vorne (0,1,2), hinten (0,1,2), links (0,1,2), rechts (0,1,2), oben (1,9999)
