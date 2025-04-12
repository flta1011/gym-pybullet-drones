import numpy as np


def _preprocessAction(self, action):
    """Preprocesses the action from PPO to drone controls.
    Maps discrete actions to movement vectors.

    12.1.25:FT: gecheckt, ist gleich mit der Standard BaseRLAviary
    """
    # Convert action to movement vector
    # action_to_movement = {
    #     0: np.array([[1, 0, 0, 0.99, 0]]),  # Forward
    #     1: np.array([[-1, 0, 0, 0.99, 0]]), # Backward
    #     2: np.array([[0, 0, 0, 0.99, 0]]),  # Stay
    # }

    rpm = np.zeros((self.NUM_DRONES, 4))
    for k in range(self.NUM_DRONES):
        #### Get the current state of the drone  ###################
        state = self._getDroneStateVector(k)
        target_v = action[k, :4]
        #### Normalize the first 3 components of the target velocity
        if np.linalg.norm(target_v[0:3]) != 0:
            v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
        else:
            v_unit_vector = np.zeros(3)

        match self.ACTION_SPACE_VERSION:
            case "A2" | "A3":
                # Keine Drehung zulassen, Yaw bleibt konstant
                Calculate_new_yaw = self.INIT_RPYS[0, 2]  # Initiale Yaw-Ausrichtung
                cur_ang_vel = np.array([0, 0, 0])  # Keine Drehgeschwindigkeit
            case "A1":
                # NOTE - neu hinzuegefügt, dass die Drohne sich auch drehen kann
                # Drehung zulassen
                current_yaw = state[9]
                change_value_yaw = action[k, 4]
                Calculate_new_yaw = current_yaw + change_value_yaw
                cur_ang_vel = state[13:16]  # Aktuelle Drehgeschwindigkeit

        #### Compute Control #######################################
        temp, _, _ = self.ctrl[k].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=cur_ang_vel,  # Dynamisch basierend auf Action Space
            target_pos=np.array([state[0], state[1], 0.5]),  # Zielposition
            target_rpy=np.array([0, 0, Calculate_new_yaw]),  # Zielausrichtung
            target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector,  # Zielgeschwindigkeit
        )
        rpm[k, :] = temp
    return rpm
