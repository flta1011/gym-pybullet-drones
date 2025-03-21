import time


def _computeTerminated(self):
    """Computes the current terminated value(s).

    Unused as this subclass is not meant for reinforcement learning.

    Returns
    -------
    bool
        Dummy value.

    """
    state = self._getDroneStateVector(0)
    # starte einen Timer, wenn die Drohne im sweet spot ist
    if state[25] < 1:  # 0.15 = Radius Scheibe
        self.still_time += 1 / self.reward_and_action_change_freq  # Increment by simulation timestep (in seconds) # TBD: funktioniert das richtig?
    else:
        self.still_time = 0.0  # Reset timer to 0 seconds

    # Wenn die Drohne im sweet spot ist (bezogen auf Sensor vorne, Sensor und seit 5 sekunden still ist, beenden!
    if self.still_time >= 5:
        current_time = time.localtime()
        Grund_Terminated = f"Drohne ist 5 s lang unter dem Objekt gewesen. Zeitstempel (min:sek) {time.strftime('%M:%S', current_time)}"
        return True, Grund_Terminated

    Grund_Terminated = None

    return False, Grund_Terminated
