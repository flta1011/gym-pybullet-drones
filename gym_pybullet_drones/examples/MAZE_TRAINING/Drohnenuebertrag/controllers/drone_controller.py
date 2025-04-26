import logging
import time

import cflib
import cflib.crazyflie
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal
from stable_baselines3 import DQN, PPO, SAC

from .obs_manager import OBSManager


class DroneController(QObject):
    # Define signals for thread-safe updates
    position_updated = pyqtSignal(list)
    measurement_updated = pyqtSignal(dict)

    def __init__(self, uri, observation_type, action_type, model_type, model_path):
        super().__init__()
        self.emergency_stop_active = False
        # Increased SAFE_DISTANCE for earlier detection
        self.SAFE_DISTANCE = 0.1
        # Increased PUSHBACK_DISTANCE for stronger reaction
        self.PUSHBACK_VEL = 0.4
        # AI Prediction frequency
        self.ai_prediction_counter = 0
        self.hover_frequency = 100  # Hz
        self.hover_interval = int(1000 / self.hover_frequency)  # seconds
        self.ai_frequency = 2  # Hz
        self.ai_prediction_rate = self.hover_frequency / self.ai_frequency  # Only predict every 25th cycle

        self.uri = uri
        self.observation_type = observation_type
        self.action_type = action_type
        self.model_type = model_type
        self.model_path = model_path
        self.latest_position = None
        self.latest_measurement = None
        self.SPEED_FACTOR = 0.1
        self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}  # SECTION Höhe ändern
        self.hoverTimer = None
        self.number_last_actions = 100  # SECTION Nummer ändern für last actions
        self.last_actions = np.zeros(self.number_last_actions)
        self.obs_manager = OBSManager(observation_type=observation_type)

        # Safety system improvements
        self.measurement_history = []
        self.history_size = 5  # Keep track of the last 5 measurements for filtering
        self.safety_override_active = False
        self.safety_override_timer = 0
        self.safety_timeout = 20  # Safety override remains active for 20 cycles (about 1 second)
        self.consecutive_danger_readings = 0
        self.min_consecutive_readings = 2  # Require multiple consecutive readings before triggering safety

        self.ai_control_active = False

        # GUI Callbacks
        self.position_callback = None
        self.measurement_callback = None
        self.start_fly_callback = None
        self.emergency_stop_callback = None
        self.update_slam_map_callback = None
        self.ai_action_callback = None
        self.last_action_was_pushback = False

        # Connect signals to slots
        self.position_updated.connect(self._on_position_updated)
        self.measurement_updated.connect(self._on_measurement_updated)

        # Load the appropriate model type based on MODEL_Version
        match model_type:
            case "M1":
                self.model = PPO.load(model_path)
                print(f"Loaded PPO model from {model_path}")
            case "M2":
                self.model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
            case "M3":
                self.model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
            case "M4":
                self.model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
            case "M5":
                self.model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
            case "M6":
                self.model = SAC.load(model_path)
                print(f"Loaded SAC model from {model_path}")
            case _:
                print(f"[ERROR]: Unknown model version in TEST-(PREDICTION)-MODE: {model_type}")

    def set_position_callback(self, callback):
        self.position_callback = callback

    def set_measurement_callback(self, callback):
        self.measurement_callback = callback

    def set_update_slam_map_callback(self, callback):
        """Set the callback for SLAM map updates."""
        self.obs_manager.set_slam_update_callback(callback)

    def set_ai_action_callback(self, callback):
        """Set the callback for AI action updates."""
        self.ai_action_callback = callback

    def connect(self):
        try:
            cflib.crtp.init_drivers()
            self.cf = Crazyflie(ro_cache=None, rw_cache="cache")

            # Connect callbacks from the Crazyflie API
            self.cf.connected.add_callback(self.connected)
            self.cf.disconnected.add_callback(self.disconnected)

            # Connect to the Crazyflie
            if hasattr(self, "cf") and self.cf.is_connected():
                self.cf.close_link()
                time.sleep(1)
            self.cf.open_link(self.uri)

            # Arm the Crazyflie
            self.cf.platform.send_arming_request(True)
        except Exception as e:
            print(f"Connection error: {str(e)}")

    def connected(self, URI):
        print("We are now connected to {}".format(URI))

        # The definition of the logconfig can be made before connecting
        lpos = LogConfig(name="Position", period_in_ms=100)
        lpos.add_variable("stateEstimate.x")
        lpos.add_variable("stateEstimate.y")
        lpos.add_variable("stateEstimate.z")

        try:
            self.cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.start()
        except KeyError as e:
            print("Could not start log configuration," "{} not found in TOC".format(str(e)))
        except AttributeError:
            print("Could not add Position log config, bad configuration.")

        lmeas = LogConfig(name="Meas", period_in_ms=100)
        lmeas.add_variable("range.front")
        lmeas.add_variable("range.back")
        lmeas.add_variable("range.up")
        lmeas.add_variable("range.left")
        lmeas.add_variable("range.right")
        lmeas.add_variable("range.zrange")
        lmeas.add_variable("stabilizer.roll")
        lmeas.add_variable("stabilizer.pitch")
        lmeas.add_variable("stabilizer.yaw")

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except KeyError as e:
            logging.error(f"LogConfig error: {str(e)}. Ensure the variable exists in the TOC.")
        except AttributeError as e:
            logging.error(f"AttributeError in LogConfig: {str(e)}. Check the configuration.")

    def disconnected(self, URI):
        print(f"Disconnected from {URI}")
        # if hasattr(self, "hoverTimer") and self.hoverTimer.isActive():
        #     self.hoverTimer.stop()

    def pos_data(self, timestamp, data, logconf):
        position = [data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"]]
        self.latest_position = position
        self.position_updated.emit(position)

    def _on_position_updated(self, position):
        if self.position_callback:
            self.position_callback(position)

    def get_position(self):
        return self.latest_position

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            "roll": data["stabilizer.roll"],
            "pitch": data["stabilizer.pitch"],
            "yaw": data["stabilizer.yaw"] * np.pi / 180.0,
            "front": data["range.front"] / 1000.0,
            "back": data["range.back"] / 1000.0,
            "up": data["range.up"] / 1000.0,
            "down": data["range.zrange"] / 1000.0,
            "left": data["range.left"] / 1000.0,
            "right": data["range.right"] / 1000.0,
        }

        # Add distance validation for sensors
        for sensor in ["front", "back", "left", "right", "up", "down"]:
            # Check if the current sensor reading exceeds 4 meters
            if measurement[sensor] > 4.0:
                # Replace with a fixed value to match simulation traning behavior
                measurement[sensor] = int(4.0)
                # print(f"Sensor {sensor} reading capped at 4.0m")

        # Update measurement history for safety filtering
        self.measurement_history.append(measurement.copy())
        if len(self.measurement_history) > self.history_size:
            self.measurement_history.pop(0)

        self.latest_measurement = measurement
        self.measurement_updated.emit(measurement)
        # print(self.ai_control_active)

    def _on_measurement_updated(self, measurement):
        if self.measurement_callback:
            self.measurement_callback(measurement)

        self.trigger_obs_update()

    def get_measurements(self):
        return self.latest_measurement

    def trigger_obs_update(self):
        # Create a copy of the measurements with correctly renamed keys for obs_manager
        if self.latest_measurement and self.latest_position:
            adjusted_measurements = {
                "roll": self.latest_measurement["roll"],
                "pitch": self.latest_measurement["pitch"],
                "yaw": self.latest_measurement["yaw"],
                "front": self.latest_measurement["front"],
                "back": self.latest_measurement["back"],
                "up": self.latest_measurement["up"],
                "down": self.latest_measurement["down"],
                "left": self.latest_measurement["left"],
                "right": self.latest_measurement["right"],
            }
            self.obs_manager.update(position=self.latest_position, measurements=adjusted_measurements, last_actions=self.last_actions)

    def start_fly(self):
        self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}  # SECTION Höhe ändern

        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(self.hover_interval)
        self.hoverTimer.start()

    def emergency_stop(self):
        if self.hoverTimer is not None:
            self.hoverTimer.stop()
            self.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            self.cf.commander.send_stop_setpoint()
            self.emergency_stop_active = True
            print("Emergency stop activated")
        else:
            print("No hover timer to stop")
            self.emergency_stop_active = True

    def set_ai_control_state(self, state):
        """
        Update the AI control state.
        :param state: The new state of AI control (True or False).
        """
        self.ai_control_active = state
        print(f"DroneController: AI control is now {'active' if state else 'inactive'}.")

    def sendHoverCommand(self):
        if self.emergency_stop_active:
            return

        # Add AI control logic directly in the control loop
        if self.ai_control_active and hasattr(self, "obs_manager"):
            print(f"[AI Control] AI control is active. Current hover: {self.hover}")
            # Only predict at 2Hz (every 25 cycles at 50Hz)
            self.ai_prediction_counter += 1
            if self.ai_prediction_counter >= self.ai_prediction_rate:
                self.ai_prediction_counter = 0
                # Get current observation from obs_manager
                observation_space = self.obs_manager.get_observation()
                print(f"Observation space: {observation_space}")

                if observation_space is not None:
                    # Update hover values based on AI prediction
                    new_x_vel, new_y_vel = self.predict_action(observation_space)
                    self.hover["x"] = new_x_vel * self.SPEED_FACTOR
                    self.hover["y"] = new_y_vel * self.SPEED_FACTOR

            self.check_safety()
        self.cf.commander.send_hover_setpoint(self.hover["x"], self.hover["y"], self.hover["yaw"], self.hover["height"])

    def check_safety(self):
        """
        Check the distance sensors and trigger safety mechanisms if a wall is too close.
        """
        if self.latest_measurement is None:
            return  # No measurements available yet

        # Add the latest measurement to the history
        self.measurement_history.append(self.latest_measurement)
        if len(self.measurement_history) > self.history_size:
            self.measurement_history.pop(0)

        # Calculate the average of the last few measurements to filter out noise
        avg_measurement = {key: np.mean([m[key] for m in self.measurement_history]) for key in self.latest_measurement.keys()}

        # Get distance sensor readings
        front = avg_measurement.get("front", float("inf"))
        back = avg_measurement.get("back", float("inf"))
        left = avg_measurement.get("left", float("inf"))
        right = avg_measurement.get("right", float("inf"))

        # Check if any sensor detects a wall too close
        if front < self.SAFE_DISTANCE:
            self.consecutive_danger_readings += 1
            if self.consecutive_danger_readings >= self.min_consecutive_readings:
                self.trigger_safety("front", back, left, right)
        elif back < self.SAFE_DISTANCE:
            self.consecutive_danger_readings += 1
            if self.consecutive_danger_readings >= self.min_consecutive_readings:
                self.trigger_safety("back", front, left, right)
        elif left < self.SAFE_DISTANCE:
            self.consecutive_danger_readings += 1
            if self.consecutive_danger_readings >= self.min_consecutive_readings:
                self.trigger_safety("left", right, front, back)
        elif right < self.SAFE_DISTANCE:
            self.consecutive_danger_readings += 1
            if self.consecutive_danger_readings >= self.min_consecutive_readings:
                self.trigger_safety("right", left, front, back)
        else:
            self.consecutive_danger_readings = 0  # Reset counter if no danger is detected

    def trigger_safety(self, direction, opposite, adjacent1, adjacent2):
        """
        Trigger the safety mechanism to stop the drone and push it back to a safe distance.
        :param direction: The direction of the detected wall ("front", "back", "left", "right").
        :param opposite: The distance in the opposite direction.
        :param adjacent1: The distance to the first adjacent side.
        :param adjacent2: The distance to the second adjacent side.
        """
        print(f"[SAFETY] Wall detected too close in the {direction} direction!")

        # Stop all movement
        self.hover["x"] = 0.0
        self.hover["y"] = 0.0
        self.hover["yaw"] = 0.0
        print("[SAFETY] Drone movement stopped.")

        # Determine pushback direction
        if direction == "front" and opposite > self.PUSHBACK_VEL:
            self.hover["x"] = -self.PUSHBACK_VEL * 0.5  # Push backward
            if self.model_type == "M6":
                self.update_last_actions(np.array([-self.PUSHBACK_VEL, 0]))  # No lateral movement for SAC
                self.last_action_was_pushback = True
            elif self.model_type == "M1" or self.model_type == "M2" or self.model_type == "M3" or self.model_type == "M4" or self.model_type == "M5":
                self.update_last_actions(1)  # Update last action to backward
                self.last_action_was_pushback = True
        elif direction == "back" and opposite > self.PUSHBACK_VEL:
            self.hover["x"] = self.PUSHBACK_VEL * 0.5  # Push forward
            if self.model_type == "M6":
                self.update_last_actions(np.array([self.PUSHBACK_VEL, 0]))
                self.last_action_was_pushback = True
            elif self.model_type == "M1" or self.model_type == "M2" or self.model_type == "M3" or self.model_type == "M4" or self.model_type == "M5":
                self.update_last_actions(0)
                self.last_action_was_pushback = True
        elif direction == "left" and opposite > self.PUSHBACK_VEL:
            self.hover["y"] = -self.PUSHBACK_VEL * 0.5  # Push right
            if self.model_type == "M6":
                self.update_last_actions(np.array([0, -self.PUSHBACK_VEL]))
                self.last_action_was_pushback = True
            elif self.model_type == "M1" or self.model_type == "M2" or self.model_type == "M3" or self.model_type == "M4" or self.model_type == "M5":
                self.update_last_actions(3)
                self.last_action_was_pushback = True
        elif direction == "right" and opposite > self.PUSHBACK_VEL:
            self.hover["y"] = self.PUSHBACK_VEL * 0.5  # Push left
            if self.model_type == "M6":
                self.update_last_actions(np.array([0, self.PUSHBACK_VEL]))
                self.last_action_was_pushback = True
            elif self.model_type == "M1" or self.model_type == "M2" or self.model_type == "M3" or self.model_type == "M4" or self.model_type == "M5":
                self.update_last_actions(2)
                self.last_action_was_pushback = True
        else:
            self.last_action_was_pushback = False

        # Ensure no collision during pushback
        if adjacent1 < self.SAFE_DISTANCE or adjacent2 < self.SAFE_DISTANCE:
            print("[SAFETY] Pushback canceled due to nearby walls.")
            self.hover["x"] = 0.0
            self.hover["y"] = 0.0

        print(f"[SAFETY] Updated hover for pushback: {self.hover}")

        self.pushback_hover = self.hover.copy()

    def updateHover(self, k=None, v=None, observation_space=None):
        """
        Updates the hover dictionary based on keyboard input.
        This method handles manual control inputs, which always take priority over AI control.
        :param k: The key to update (e.g., "x", "y", "height")
        :param v: The value to update (e.g., 1 for forward, -1 for backward)
        """
        # Always process keyboard commands, regardless of AI control state
        # This ensures manual override is always possible for safety reasons
        if k in self.hover:
            self.hover[k] += v * self.SPEED_FACTOR
            # If AI control is active, notify that manual override is happening
            if self.ai_control_active:
                print(f"[Manual Override] Manual input overriding AI control: {self.hover}")
            else:
                print(f"[Manual Control] Updated hover via keyboard: {self.hover}")

            # Reset AI prediction counter to avoid immediate AI override of manual input
            self.ai_prediction_counter = 0
        else:
            print(f"[Manual Control] Invalid hover key: {k}")

    def predict_action(self, observation_space):
        action, _ = self.model.predict(observation_space, deterministic=True)
        print(f"Predicted action: {action}")
        # Call the callback here, where we know action is available
        if hasattr(self, "ai_action_callback") and self.ai_action_callback is not None:
            self.ai_action_callback(action)

        if self.action_type == "A3":
            new_x_vel = action[0]
            new_y_vel = action[1]
        else:
            # 1: np.array([[1, 0, 0, 0.99, 0]]), # Fly 90° (Forward)
            # 2: np.array([[-1, 0, 0, 0.99, 0]]), # Fly 180° (Backward)
            # 3: np.array([[0, 1, 0, 0.99, 0]]), # Fly 90° (Left)
            # 4: np.array([[0, -1, 0, 0.99, 0]]), # Fly 270° (Right)

            if action == 0:
                new_x_vel = 1
                new_y_vel = 0
            elif action == 1:
                new_x_vel = -1
                new_y_vel = 0
            elif action == 2:
                new_x_vel = 0
                new_y_vel = 1
            elif action == 3:
                new_x_vel = 0
                new_y_vel = -1
            else:
                new_x_vel = 0
                new_y_vel = 0

        self.update_last_actions(action)
        return new_x_vel, new_y_vel

    def update_last_actions(self, action):

        match self.model_type:
            case "M1" | "M2" | "M3" | "M4" | "M5":
                # For DQN models, store the last action in a circular buffer
                self.last_actions = np.roll(self.last_actions, 1)
                # Handle different action formats (scalar or array)
                self.last_actions[0] = action
            case "M6":
                # For SAC models, store the last action in a circular buffer
                # Store the last action in the first position
                # and roll the rest of the array
                if self.last_action_was_pushback != True:
                    self.last_actions = np.roll(self.last_actions, 2)
                    # Handle different action formats (scalar or array)
                    self.last_actions[0] = action[0]
                    self.last_actions[1] = action[1]
