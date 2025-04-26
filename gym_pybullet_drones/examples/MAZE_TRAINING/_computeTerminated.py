import time


def _computeTerminated(self):
    """Computes the current terminated value(s).

    Returns
    -------
    bool
        Whether the episode should be terminated.
    str
        Reason for termination.
    """
    match self.Terminated_Version:
        case "T1":
            """nur Terminated, wenn 80% erreicht sind"""

            Grund_Terminated = None

            if self.Ratio_Area >= 0.8:
                Grund_Terminated = "80 Prozent der Fläche wurde erkundet"
                self.Terminated_Truncated_Counter += 1
                return True, Grund_Terminated

            return False, Grund_Terminated

        case "T2":
            """
            Terminated, wenn die Zeit abgelaufen ist oder wenn eine der Abstandswerte geringer als X ist, oder wenn die Drohne abstürzt ist, oder zu tilted ist
            """

            Grund_Terminated = None

            if self.Ratio_Area >= 0.8:
                Grund_Terminated = "80 Prozent der Fläche wurde erkundet"
                self.Terminated_Truncated_Counter += 1
                return True, Grund_Terminated

            state = self._getDroneStateVector(0)

            if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
                Grund_Terminated = "Zu tilted"
                return True, Grund_Terminated

            if state[21] <= 0.1 or state[22] <= 0.1 or state[23] <= 0.1 or state[24] <= 0.1:
                Grund_Terminated = "Zu nah an der Wand"
                return True, Grund_Terminated

            return False, Grund_Terminated

        case _:
            raise ValueError(f"Unknown terminated version: {self.Terminated_Version}")
