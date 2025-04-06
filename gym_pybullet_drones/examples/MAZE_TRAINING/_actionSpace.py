import numpy as np
from gymnasium import spaces


def _actionSpace(self):
    """Returns the action space of the environment.

    Returns
    -------
    spaces.Discrete
    DREHUNG MATHEMATISCH POSITIV (GEGEN DEN UHRZEIGER)

    1: np.array([[1, 0, 0, 0.99, 0]]), # Fly 90° (Forward)
    2: np.array([[-1, 0, 0, 0.99, 0]]), # Fly 180° (Backward)
    3: np.array([[0, 1, 0, 0.99, 0]]), # Fly 90° (Left)
    4: np.array([[0, -1, 0, 0.99, 0]]), # Fly 270° (Right)
    5: np.array([[0, 0, 0, 0.99, 1/4*np.pi]]), # 45° Left-Turn
    6: np.array([[0, 0, 0, 0.99, -1/4*np.pi]]), # 45° Right-Turn

    """

    #

    match self.ACTION_SPACE_VERSION:
        case "A1":  # Vier Richtugnen +- 1/72pi Drehung

            return spaces.Discrete(6)

        case "A2":  # Vier Richtungen

            return spaces.Discrete(4)

        case "A3":  # Vier Richtungen, kontinuierlich
            """For SAC, we use a continuous action space with:
            - Linear velocity in x (-self.VelocityScale to self.VelocityScale)
            - Linear velocity in y (-self.VelocityScale to self.VelocityScale)
            """
            return spaces.Box(
                low=np.array([-self.VelocityScale, -self.VelocityScale]),
                high=np.array([self.VelocityScale, self.VelocityScale]),
                dtype=np.float32,
            )
