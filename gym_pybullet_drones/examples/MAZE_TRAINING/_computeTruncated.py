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
            # NOTE - Auskommentiert, da wir ihn weiter fliegen lassen wollen wenn er an die Wand fliegt und somit lernt es nicht
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
                # self.RewardCounterActualTrainRun += -10
                self.Terminated_Truncated_Counter += 1
                return True, Grund_Truncated

            Grund_Truncated = None

            return False, Grund_Truncated

        case "TR2":  # Bei Zeit oder Abstandswerten geringer als X wird truncated
            """truncated, wenn die Zeit abgelaufen ist und wenn einer der Sensoren geringer als einen Abstandswert X ist"""
            if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
                Grund_Truncated = "Zeit abgelaufen"
                # wenn Zeit abgeaufen ist, dann wird die Drohne bestraft
                # self.RewardCounterActualTrainRun += -10
                self.Terminated_Truncated_Counter += 1
                return True, Grund_Truncated

            state = self._getDroneStateVector(0)

            if (
                state[21] < self.Truncated_Wall_Distance
                or state[22] < self.Truncated_Wall_Distance
                or state[23] < self.Truncated_Wall_Distance
                or state[24] < self.Truncated_Wall_Distance
            ):
                Grund_Truncated = "Zu nah an der Wand"
                return True, Grund_Truncated

            Grund_Truncated = None
            return False, Grund_Truncated

        case _:
            raise ValueError(f"Unknown truncated version: {self.Truncated_Version}")
