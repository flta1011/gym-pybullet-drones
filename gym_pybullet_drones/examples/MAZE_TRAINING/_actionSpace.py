from gymnasium import spaces


def _actionSpace(self):

    match self.ACTION_SPACE_VERSION:
        case "DISKRET_PPO_MIT_DREHUNG":

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

            return spaces.Discrete(6)

        case "DISKRET_PPO_OHNE_DREHUNG":

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

            # Action Space ohne Drehung -> nur 5 Aktionen: Stehen, Vorwärts, Rückwärts, Links, Rechts

            return spaces.Discrete(4)
