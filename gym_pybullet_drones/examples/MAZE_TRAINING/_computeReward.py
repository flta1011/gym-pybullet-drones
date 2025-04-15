import csv
import heapq
import time

import matplotlib.pyplot as plt
import numpy as np


def _computeReward(
    self, Maze_Number, random_number_Start, random_number_Target
):  # Funktioniert und die Drohne lernt, nahe an die Wand, aber nicht an die Wand zu fliegen. Problem: die Drohne bleibt nicht sauber im Sweetspot stehen.
    """Computes the current reward value.
    Reward-Versionen:
    - R1: Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
    - R2: Ziel hat keinen Einfluss mehr, soll aufs erkunden belohnt werden
    - R3:
    - R4:


    Returns
    -------
    float
        The reward.

    """

    match self.REWARD_VERSION:
        case "R1":  # Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
            # Initialisierung der Reward-Map und a-Star-etc.
            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # print (Start_position, "Start-REward")
                # print (End_Position, "Target-reward")
                # Set the Startpoint of the Drone
                initial_position = [
                    Start_position[1] / 0.05,
                    Start_position[0] / 0.05,
                ]  # Startpunkt der Drohne
                self.reward_map[int(initial_position[0]), int(initial_position[1])] = 4  # Startpunkt

                # Set the Targetpoint of the Drone
                target_position = [
                    End_Position[1] / 0.05,
                    End_Position[0] / 0.05,
                ]  # Zielpunkt der Drohne
                self.reward_map[int(target_position[0]), int(target_position[1])] = 5  # Zielpunkt

                # Best way to fly via A* Algorithm
                self.best_way_map = np.zeros((60, 60), dtype=int)

                def heuristic(a, b):
                    return np.linalg.norm(np.array(a) - np.array(b))  # Euklidische Distanz

                def a_star_search(reward_map, start, goal):
                    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    close_set = set()
                    came_from = {}
                    gscore = {start: 0}
                    fscore = {start: heuristic(start, goal)}
                    oheap = []

                    heapq.heappush(oheap, (fscore[start], start))

                    while oheap:
                        current = heapq.heappop(oheap)[1]

                        if current == goal:
                            path = [goal]
                            while current in came_from:
                                current = came_from[current]
                                path.append(current)
                            path.reverse()
                            return path

                        close_set.add(current)
                        for i, j in neighbors:
                            neighbor = (current[0] + i, current[1] + j)

                            if not (0 <= neighbor[0] < reward_map.shape[0] and 0 <= neighbor[1] < reward_map.shape[1]):
                                continue  # Außerhalb des Grids

                            if reward_map[neighbor[0], neighbor[1]] == 6:
                                continue  # Hindernis überspringen

                            tentative_g_score = gscore[current] + reward_map[neighbor[0], neighbor[1]]  # Kosten berücksichtigen

                            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float("inf")):
                                continue

                            if tentative_g_score < gscore.get(neighbor, float("inf")) or all(n[1] != neighbor for n in oheap):
                                came_from[neighbor] = current
                                gscore[neighbor] = tentative_g_score
                                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                                heapq.heappush(oheap, (fscore[neighbor], neighbor))

                    return None  # Kein Pfad gefunden

                start = (int(initial_position[0]), int(initial_position[1]))
                goal = (int(target_position[0]), int(target_position[1]))
                path = a_star_search(self.reward_map, start, goal)

                # Initializing the best way map
                if path:
                    for position in path:
                        self.best_way_map[position[0], position[1]] = 1

                # Save the best way map to a CSV file
                with open("best_way_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(self.best_way_map)

                # Save the reward map to a CSV file
                with open("reward_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(self.reward_map)

            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne

            #### Rewards initialisieren ####
            self.reward_components["collision_penalty"] = 0
            self.reward_components["distance_reward"] = 0
            # self.reward_components["best_way_bonus"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["explore_bonus_visited_field"] = 0
            self.reward_components["Target_Hit_Reward"] = 0

            ###### 1.PUNISHMENT FOR COLLISION ######
            # if self.action_change_because_of_Collision_Danger == True:
            #     self.reward_components["collision_penalty"] = -10.0

            ###### 18.3: Ziel hat keinen Einfluss mehr, soll aufs erkunden belohnt werden
            # ###### 2.REWARD FOR DISTANCE TO TARGET (line of sight) ######
            # # Get current drone position and target position # STIMMT DER TARGET POSITION?

            # drone_pos = state[0:2]  # XY position from state vector
            # target_pos = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target][0:2] # XY position of the target
            # Target_Value_1 = target_pos[0]
            # Target_Value_2 = target_pos[1]
            # target_pos = [Target_Value_2, Target_Value_1] # X und Y Werte vertauschen, weil die Drohne die Werte vertauscht
            # # print(drone_pos, "Drone Position")s
            # # print(target_pos, "Target Position")
            # # Calculate distance to target
            # self.distance = np.linalg.norm(drone_pos - target_pos)

            # # print(self.distance, "Distance")
            # # print(drone_pos, "Drone Position")
            # # print(target_pos, "Target Position")

            # # Define max distance and max reward
            # MAX_DISTANCE = 3.0  # Maximum expected distance in meters
            # MAX_REWARD = 0.5    # Maximum reward for distance (excluding target hit bonus)

            # # Linear reward that scales from 0 (at MAX_DISTANCE) to MAX_REWARD (at distance=0)
            # distance_ratio = min(self.distance/MAX_DISTANCE, 1.0)
            # self.reward_components["distance_reward"] = MAX_REWARD * (1 - distance_ratio) ## 4.3.25: auf Linear umgestellt, damit auch in weiter entfernten Feldern noch ein Gradient erkannt werden kann

            # # Add huge reward if target is hit (within 0.05m) and top sensor shows no obstacle
            # if self.distance < 0.15 and state[25] < 1: # 0.15 = Radius Scheibe
            #     self.reward_components["Target_Hit_Reward"] += 1000.0
            #     print(f"Target hit. Zeitstempel (min:sek) {time.strftime('%M:%S', time.localtime())}")

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]

            ###### 3. REWARD FOR BEING ON THE BEST WAY ######
            # Get the current position of the drone

            # Check if the drone is on the best way
            # if self.best_way_map[current_position[0], current_position[1]] == 1:
            #     self.reward_components["best_way_bonus"] = 10

            ###### 4. REWARD FOR EXPLORING NEW AREAS ######
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            # Iterate through 5x5 grid centered on current position --> 3x3 grid
            for i in range(max(0, x - 2), min(60, x + 2)):
                for j in range(max(0, y - 2), min(60, y + 2)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += 1

            # Only give reward if any new cells were explored
            # if reward_given:
            #     self.reward_components["explore_bonus_new_field"] = 1
            # # Area visited once
            # elif self.reward_map[current_position[0], current_position[1]] == 1:
            #     self.reward_components["explore_bonus_visited_field"] = 0.1
            #     self.reward_map[current_position[0], current_position[1]] = 2
            # # Area visited twice
            # elif self.reward_map[current_position[0], current_position[1]] >=2:
            #     self.reward_components["explore_bonus_visited_field"] = -0.1# darf keine Bestrafung geben, wenn er noch mal auf ein bereits besuchtes Feld fliegt, aber auch keine Belohnung
            #     self.reward_map[current_position[0], current_position[1]] = 3

            if self.DASH_ACTIVE == True:
                # Save the best way map to a CSV file
                with open(
                    "gym_pybullet_drones/examples/MAZE_TRAINING/best_way_map.csv",
                    "w",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerows(self.best_way_map)

                # Save the reward map to a CSV file
                with open(
                    "gym_pybullet_drones/examples/MAZE_TRAINING/reward_map.csv",
                    "w",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerows(self.reward_map)

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty"]
                + self.reward_components["distance_reward"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["explore_bonus_visited_field"]
                + self.reward_components["Target_Hit_Reward"]
            )
            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward
        case "R2":  # Ziel hat keinen Einfluss mehr, soll aufs erkunden belohnt werden
            # Initialisierung der Reward-Map und a-Star-etc.
            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # print (Start_position, "Start-REward")
                # print (End_Position, "Target-reward")
                # Set the Startpoint of the Drone
                initial_position = [
                    Start_position[1] / 0.05,
                    Start_position[0] / 0.05,
                ]  # Startpunkt der Drohne
                self.reward_map[int(initial_position[0]), int(initial_position[1])] = 4  # Startpunkt

                # Set the Targetpoint of the Drone
                target_position = [
                    End_Position[1] / 0.05,
                    End_Position[0] / 0.05,
                ]  # Zielpunkt der Drohne
                self.reward_map[int(target_position[0]), int(target_position[1])] = 5  # Zielpunkt

                # Best way to fly via A* Algorithm
                self.best_way_map = np.zeros((60, 60), dtype=int)

                def heuristic(a, b):
                    return np.linalg.norm(np.array(a) - np.array(b))  # Euklidische Distanz

                def a_star_search(reward_map, start, goal):
                    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    close_set = set()
                    came_from = {}
                    gscore = {start: 0}
                    fscore = {start: heuristic(start, goal)}
                    oheap = []

                    heapq.heappush(oheap, (fscore[start], start))

                    while oheap:
                        current = heapq.heappop(oheap)[1]

                        if current == goal:
                            path = [goal]
                            while current in came_from:
                                current = came_from[current]
                                path.append(current)
                            path.reverse()
                            return path

                        close_set.add(current)
                        for i, j in neighbors:
                            neighbor = (current[0] + i, current[1] + j)

                            if not (0 <= neighbor[0] < reward_map.shape[0] and 0 <= neighbor[1] < reward_map.shape[1]):
                                continue  # Außerhalb des Grids

                            if reward_map[neighbor[0], neighbor[1]] == 6:
                                continue  # Hindernis überspringen

                            tentative_g_score = gscore[current] + reward_map[neighbor[0], neighbor[1]]  # Kosten berücksichtigen

                            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float("inf")):
                                continue

                            if tentative_g_score < gscore.get(neighbor, float("inf")) or all(n[1] != neighbor for n in oheap):
                                came_from[neighbor] = current
                                gscore[neighbor] = tentative_g_score
                                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                                heapq.heappush(oheap, (fscore[neighbor], neighbor))

                    return None  # Kein Pfad gefunden

                start = (int(initial_position[0]), int(initial_position[1]))
                goal = (int(target_position[0]), int(target_position[1]))
                path = a_star_search(self.reward_map, start, goal)

                # Initializing the best way map
                if path:
                    for position in path:
                        self.best_way_map[position[0], position[1]] = 1

                if self.DASH_ACTIVE == True:
                    # Save the best way map to a CSV file
                    with open("best_way_map.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(self.best_way_map)

                    # Save the reward map to a CSV file
                    with open("reward_map.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(self.reward_map)

            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne

            #### Rewards initialisieren ####
            self.reward_components["collision_penalty"] = 0
            self.reward_components["distance_reward"] = 0
            # self.reward_components["best_way_bonus"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["explore_bonus_visited_field"] = 0
            self.reward_components["Target_Hit_Reward"] = 0

            ###### 1.PUNISHMENT FOR COLLISION ######
            # if self.action_change_because_of_Collision_Danger == True:
            #     self.reward_components["collision_penalty"] = -1.0

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]

            ###### 3. REWARD FOR BEING ON THE BEST WAY ######
            # Get the current position of the drone

            # Check if the drone is on the best way
            # if self.best_way_map[current_position[0], current_position[1]] == 1:
            #     self.reward_components["best_way_bonus"] = 10

            ###### 4. REWARD FOR EXPLORING NEW AREAS ######
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            # Iterate through 5x5 grid centered on current position --> 3x3 grid
            for i in range(max(0, x - 2), min(60, x + 2)):
                for j in range(max(0, y - 2), min(60, y + 2)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += 1

            # Only give reward if any new cells were explored
            # if reward_given:
            #     self.reward_components["explore_bonus_new_field"] = 1
            # # Area visited once
            # elif self.reward_map[current_position[0], current_position[1]] == 1:
            #     self.reward_components["explore_bonus_visited_field"] = 0.1
            #     self.reward_map[current_position[0], current_position[1]] = 2
            # # Area visited twice
            # elif self.reward_map[current_position[0], current_position[1]] >=2:
            #     self.reward_components["explore_bonus_visited_field"] = -0.1# darf keine Bestrafung geben, wenn er noch mal auf ein bereits besuchtes Feld fliegt, aber auch keine Belohnung
            #     self.reward_map[current_position[0], current_position[1]] = 3

            # Save the best way map to a CSV file
            with open(
                "gym_pybullet_drones/examples/MAZE_TRAINING/best_way_map.csv",
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerows(self.best_way_map)

            # Save the reward map to a CSV file
            with open(
                "gym_pybullet_drones/examples/MAZE_TRAINING/reward_map.csv",
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerows(self.reward_map)

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty"]
                + self.reward_components["distance_reward"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["explore_bonus_visited_field"]
                + self.reward_components["Target_Hit_Reward"]
            )
            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case "R3":  # Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
            # Initialisierung der Reward-Map und a-Star-etc.
            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # Save the reward map to a CSV file
                with open("reward_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(self.reward_map)

            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne

            #### Rewards initialisieren ####
            self.reward_components["collision_penalty"] = 0
            self.reward_components["distance_reward"] = 0
            # self.reward_components["best_way_bonus"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["explore_bonus_visited_field"] = 0
            self.reward_components["Target_Hit_Reward"] = 0

            ###### 4. REWARD FOR EXPLORING NEW AREAS ######

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            # Iterate through 5x5 grid centered on current position --> 3x3 grid
            for i in range(max(0, x - 2), min(60, x + 2)):
                for j in range(max(0, y - 2), min(60, y + 2)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += 5

            # Save the reward map to a CSV file
            with open(
                "gym_pybullet_drones/examples/MAZE_TRAINING/reward_map.csv",
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerows(self.reward_map)

            ###### 1.PUNISHMENT FOR COLLISION ######
            # NOTE - Collision Penalty über HeatMap muss noch gemacht werden

            # 7.783662131519274 Wert für Mitte eines Objekts
            # 7.0619562663615865 Wert für eine zeile/spalte um die mitte
            # 2.1139027463965774 Wert für zwei zeilen/spalte um die mitte
            # 0.8171304187933945 Wert für drei zeilen/spalte um die mitte
            # 0.3265204823040592 Wert für vier zeilen/spalte um die mitte
            # 0.11973387347488995 Wert für fünf zeilen/spalte um die mitte
            # 0.03434013766226782 Wert für sechs zeilen/spalte um die mitte
            # 0.0049732348218131705 Wert für sieben zeilen/spalte um die mitte

            Value_on_Heatmap = self.potential_map[x, y]

            self.reward_components["collision_penalty"] = -Value_on_Heatmap * self.Multiplier_Collision_Penalty

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty"]
                + self.reward_components["distance_reward"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["explore_bonus_visited_field"]
                + self.reward_components["Target_Hit_Reward"]
            )
            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case "R4":  # Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
            # Initialisierung der Reward-Map und a-Star-etc.
            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                Explore_Matrix_Size_Init_Punishment = self.Explore_Matrix_Size * self.Explore_Matrix_Size * self.Reward_for_new_field
                self.reward_components["Start"] = -Explore_Matrix_Size_Init_Punishment

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                # Erstellen der Heatmap für die Belohnung des Abstandes zur Wand und die DistanzMap
                self._compute_potential_fields()

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # # Save the reward map to a CSV file
                # with open("reward_map.csv", "w", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerows(self.reward_map)

                self.Area_counter = 0

            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne

            #### Rewards initialisieren ####
            self.reward_components["collision_penalty"] = 0
            self.reward_components["Step_counter"] = 0
            # self.reward_components["best_way_bonus"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["Prozentual_Bonus"] = 0

            ###### 4. REWARD FOR EXPLORING NEW AREAS ######

            if self.step_counter % 1 == 0:
                self.reward_components["Step_counter"] = self.Punishment_for_Step

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            Value1_for_Matrix = int(self.Explore_Matrix_Size / 2 - 0.5)
            Value2_for_Matrix = int(self.Explore_Matrix_Size / 2 + 0.5)

            # Iterate through 5x5 grid centered on current position
            for i in range(max(0, x - Value1_for_Matrix), min(60, x + Value2_for_Matrix)):
                for j in range(max(0, y - Value1_for_Matrix), min(60, y + Value2_for_Matrix)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += self.Reward_for_new_field
                        self.Area_counter += 1

            ###### 1.PUNISHMENT FOR COLLISION #####

            Value_on_Heatmap = self.potential_map[x, y]

            self.reward_components["collision_penalty"] = -Value_on_Heatmap * self.Multiplier_Collision_Penalty

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty"]
                + self.reward_components["Step_counter"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["Prozentual_Bonus"]
                + self.reward_components["Start"]
            )

            self.reward_components["Start"] = 0

            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case "R5":  # R4 mit dem Zusatz, dass diese Variante für TR2 optimiert ist, und für den Abstand der Wand nur eine Bestrafun bekommt, wenn danach auch truncated wird

            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                Explore_Matrix_Size_Init_Punishment = self.Explore_Matrix_Size * self.Explore_Matrix_Size * self.Reward_for_new_field
                self.reward_components["Start"] = -Explore_Matrix_Size_Init_Punishment

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                # Erstellen der Heatmap für die Belohnung des Abstandes zur Wand und die DistanzMap
                self._compute_potential_fields()

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # # Save the reward map to a CSV file
                # with open("reward_map.csv", "w", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerows(self.reward_map)

                self.Area_counter = 0

            #### Rewards initialisieren ####
            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne
            self.reward_components["collision_penalty_terminated"] = 0
            self.reward_components["Step_counter"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["Prozentual_Bonus"] = 0

            ###### REWARD FOR EXPLORING NEW AREAS ######
            if self.step_counter % 1 == 0:
                self.reward_components["Step_counter"] = self.Punishment_for_Step

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            Value1_for_Matrix = int(self.Explore_Matrix_Size / 2 - 0.5)
            Value2_for_Matrix = int(self.Explore_Matrix_Size / 2 + 0.5)

            # Iterate through 5x5 grid centered on current position
            for i in range(max(0, x - Value1_for_Matrix), min(60, x + Value2_for_Matrix)):
                for j in range(max(0, y - Value1_for_Matrix), min(60, y + Value2_for_Matrix)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += self.Reward_for_new_field
                        self.Area_counter += 1

            ###### PUNISHMENT FOR COLLISION, wenn Truncated sein sollte, gibt es eine große Bestrafung #####

            if state[21] < self.Terminated_Wall_Distance or state[22] < self.Terminated_Wall_Distance or state[23] < self.Terminated_Wall_Distance or state[24] < self.Terminated_Wall_Distance:
                self.reward_components["collision_penalty_terminated"] = self.collision_penalty_terminated

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty_terminated"]
                + self.reward_components["Step_counter"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["Prozentual_Bonus"]
                + self.reward_components["Start"]
            )

            self.reward_components["Start"] = 0

            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case "R6":  # R5 mit dem Zusatz, dass wenn die Drohne nicht zu nah an der Wand ist, gibt es einen definierten Bonus (Anstatt nur Peitsche jetzt Zuckerbrot und Peitsche)

            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                Explore_Matrix_Size_Init_Punishment = self.Explore_Matrix_Size * self.Explore_Matrix_Size * self.Reward_for_new_field
                self.reward_components["Start"] = -Explore_Matrix_Size_Init_Punishment

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                # Erstellen der Heatmap für die Belohnung des Abstandes zur Wand und die DistanzMap
                self._compute_potential_fields()

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # # Save the reward map to a CSV file
                # with open("reward_map.csv", "w", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerows(self.reward_map)

                self.Area_counter = 0

            #### Rewards initialisieren ####
            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne
            self.reward_components["collision_penalty_terminated"] = 0
            self.reward_components["Step_counter"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["Prozentual_Bonus"] = 0
            self.reward_components["no_collision_reward"] = 0
            ###### REWARD FOR EXPLORING NEW AREAS ######
            if self.step_counter % 1 == 0:
                self.reward_components["Step_counter"] = self.Punishment_for_Step

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            Value1_for_Matrix = int(self.Explore_Matrix_Size / 2 - 0.5)
            Value2_for_Matrix = int(self.Explore_Matrix_Size / 2 + 0.5)

            # Iterate through 5x5 grid centered on current position
            for i in range(max(0, x - Value1_for_Matrix), min(60, x + Value2_for_Matrix)):
                for j in range(max(0, y - Value1_for_Matrix), min(60, y + Value2_for_Matrix)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += self.Reward_for_new_field
                        self.Area_counter += 1

            ###### PUNISHMENT FOR COLLISION, wenn Truncated sein sollte, gibt es eine große Bestrafung #####

            if state[21] < self.Terminated_Wall_Distance or state[22] < self.Terminated_Wall_Distance or state[23] < self.Terminated_Wall_Distance or state[24] < self.Terminated_Wall_Distance:
                self.reward_components["collision_penalty_terminated"] = self.collision_penalty_terminated
            else:
                self.reward_components["no_collision_reward"] = self.no_collision_reward  # ein Bonuspunkt, wenn nicht zu nah an der Wand

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty_terminated"]
                + self.reward_components["Step_counter"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["Prozentual_Bonus"]
                + self.reward_components["Start"]
                + self.reward_components["no_collision_reward"]
            )

            self.reward_components["Start"] = 0

            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case "R7":

            # Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
            # Initialisierung der Reward-Map und a-Star-etc.
            if self.step_counter == 0:
                # NOTE - Reward Map

                # 0 = Unbesucht,
                # 1 = Einmal besucht,
                # 2 = Zweimal besucht,
                # 3 = Dreimal besucht,
                # 4 = Startpunkt,
                # 5 = Zielpunkt,
                # 6 = Wand

                Explore_Matrix_Size_Init_Punishment = self.Explore_Matrix_Size * self.Explore_Matrix_Size * self.Reward_for_new_field
                self.reward_components["Start"] = -Explore_Matrix_Size_Init_Punishment

                # Initializing Reward Map
                self.reward_map = np.zeros((60, 60), dtype=int)
                # Loading the Walls of the CSV Maze into the reward map as ones
                reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_{Maze_Number}.csv"
                # reward_map_file_path = f"gym_pybullet_drones/examples/maze_urdf_test/self_made_maps/maps/map_1.csv"

                with open(reward_map_file_path, "r") as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        for j, value in enumerate(row):
                            if value == "1":
                                self.reward_map[j, i] = 6  # Wand
                # Mirror the reward map vertically
                # self.reward_map = np.flipud(self.reward_map)
                # Rotate the reward map 90° mathematically negative
                # self.reward_map = np.rot90(self.reward_map, k=4)

                # Erstellen der Heatmap für die Belohnung des Abstandes zur Wand und die DistanzMap
                self._compute_punishment_map()

                # # Visualize the negative reward map
                # plt.figure(figsize=(10, 10))
                # plt.imshow(self.linear_punishment_map, cmap="hot", interpolation="nearest")
                # # plt.colorbar(label="Negative Reward Value")
                # plt.title("Linear Punishment Map Visualization")
                # plt.xlabel("X-axis")
                # plt.ylabel("Y-axis")
                # # plt.savefig("linear_punishment_map_visualization.png")  # Save the visualization as a PNG file
                # plt.show()

                if Maze_Number == 0:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number+1}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number+1}"][0][random_number_Target]
                else:
                    Start_position = self.INIT_XYZS[f"map{Maze_Number}"][0][random_number_Start]
                    End_Position = self.TARGET_POSITION[f"map{Maze_Number}"][0][random_number_Target]

                # Save the reward map to a CSV file
                with open("reward_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(self.reward_map)

                self.Area_counter = 0

            reward = 0
            state = self._getDroneStateVector(0)  # erste Drohne

            #### Rewards initialisieren ####
            self.reward_components["collision_penalty"] = 0
            self.reward_components["Step_counter"] = 0
            # self.reward_components["best_way_bonus"] = 0
            self.reward_components["explore_bonus_new_field"] = 0
            self.reward_components["Prozentual_Bonus"] = 0

            ###### 4. REWARD FOR EXPLORING NEW AREAS ######

            if self.step_counter % 1 == 0:
                self.reward_components["Step_counter"] = self.Punishment_for_Step

            # Get current position
            current_position = [int(state[0] / 0.05), int(state[1] / 0.05)]
            # Vereinfachung 18.3: 5x5 grid um die Drohne herum
            x, y = current_position[0], current_position[1]

            Value1_for_Matrix = int(self.Explore_Matrix_Size / 2 - 0.5)
            Value2_for_Matrix = int(self.Explore_Matrix_Size / 2 + 0.5)

            # Iterate through 5x5 grid centered on current position
            for i in range(max(0, x - Value1_for_Matrix), min(60, x + Value2_for_Matrix)):
                for j in range(max(0, y - Value1_for_Matrix), min(60, y + Value2_for_Matrix)):
                    if self.reward_map[i, j] == 0:
                        self.reward_map[i, j] = 1
                        self.reward_components["explore_bonus_new_field"] += self.Reward_for_new_field
                        self.Area_counter += 1

            ###### 1.PUNISHMENT FOR COLLISION #####

            Value_on_Heatmap = self.linear_punishment_map[x, y]

            self.reward_components["collision_penalty"] = -Value_on_Heatmap * self.Multiplier_Collision_Penalty

            # COMPUTE TOTAL REWARD
            reward = (
                self.reward_components["collision_penalty"]
                + self.reward_components["Step_counter"]
                + self.reward_components["explore_bonus_new_field"]
                + self.reward_components["Prozentual_Bonus"]
                + self.reward_components["Start"]
            )

            self.reward_components["Start"] = 0

            self.last_total_reward = reward  # Save the last total reward for the dashboard

            return reward

        case _:
            raise ValueError(f"Unknown reward version: {self.REWARD_VERSION}")
