def _computeTruncated(self):
    match self.Truncated_Version:
        case "TR1":  # Stamdard Truncated, wenn die Drohne zu stark geneigt ist oder abstürzt
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

            # Check if we've completed the entire sequence in ordered mode
            if self.UseOrderedMazeAndStartingPositionInsteadOfRandom:
                # If we're at the last maze, check if we're at the last position and have completed all runs
                if (
                    self.current_maze_index == len(self.List_MazesToUse) - 1
                    and self.current_start_position_index == len(self.List_Start_PositionsToUse) - 1
                    and self.current_position_run_count == self.NumberOfRunsOnEachStartingPosition - 1
                ):
                    if self.step_counter % 500 == 0:  # Log occasionally to avoid spam
                        print("Completed all ordered mazes and positions - truncating episode")
                    Grund_Truncated = "Completed all ordered mazes and positions"
                    return True, Grund_Truncated

            Grund_Truncated = None

            return False, Grund_Truncated
