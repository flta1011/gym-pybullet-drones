def _computeTruncated(self):
    match self.Truncated_Version:
        case (
            "TR1"
        ):  # Stamdard Truncated, wenn die Drohne zu stark geneigt ist oder abstürzt
            """Computes the current truncated value.

            Returns
            -------
            bool
                Whether the drone is too tilted or has crashed into a wall.

            """

            # Wenn die Zeit abgelaufen ist, beenden!
            if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
                Grund_Truncated = "Zeit abgelaufen"
                # wenn Zeit abgeaufen ist, dann wird die Drohne bestraft
                # self.RewardCounterActualTrainRun += -10
                self.Terminated_Truncated_Counter += 1
                return True, Grund_Truncated

            Grund_Truncated = None

            return False, Grund_Truncated
