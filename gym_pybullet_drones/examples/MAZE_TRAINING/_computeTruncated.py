def _computeTruncated(self):
    """Computes the current truncated value.

    Returns
    -------
    bool
        Whether the drone is too tilted or has crashed into a wall.

    """
    #NOTE - Auskommentiert, da wir ihn weiter fliegen lassen wollen wenn er an die Wand fliegt und somit lernt es nicht 
    # Truncate when the drone is too tilted
    # state = self._getDroneStateVector(0)
    # if abs(state[7]) > 0.4 or abs(state[8]) > 0.4:
    #     Grund_Truncated = "Zu tilted"
    #     return True, Grund_Truncated

    # # TBD wenn die Drone abstürzt, dann auch truncaten
    # if state[2] < 0.1:  # state[2] ist z_position der Drohne
    #     Grund_Truncated = "Crash, Abstand < 0.1 m"
    #     return True, Grund_Truncated

    # # Wenn an einer Wand gecrashed wird, beenden!
    # Abstand_truncated = self.Danger_Threshold_Wall - 0.05
    # if state[21] <= Abstand_truncated or state[22] <= Abstand_truncated or state[23] <= Abstand_truncated or state[24] <= Abstand_truncated:
    #     Grund_Truncated = f"Zu nah an der Wand (<{Abstand_truncated} m)"
    #     return True, Grund_Truncated

    # Wenn die Zeit abgelaufen ist, beenden!
    if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
        Grund_Truncated = "Zeit abgelaufen"
        # wenn Zeit abgeaufen ist, dann wird die Drohne bestraft
        self.RewardCounterActualTrainRun += -10
        return True, Grund_Truncated

    Grund_Truncated = None

    return False, Grund_Truncated
