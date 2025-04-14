import logging
import time

import cflib
import cflib.crazyflie
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt6 import QtCore


class DroneController:
    def __init__(self, uri):
        self.uri = uri
        self.latest_position = None
        self.latest_measurement = None
        self.SPEED_FACTOR = 0.5
        self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}

    def connect(self):
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

    def get_position(self):
        return self.latest_position

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            "roll": data["stabilizer.roll"],
            "pitch": data["stabilizer.pitch"],
            "yaw": data["stabilizer.yaw"],
            "front": data["range.front"],
            "back": data["range.back"],
            "up": data["range.up"],
            "down": data["range.zrange"],
            "left": data["range.left"],
            "right": data["range.right"],
        }
        self.latest_measurement = measurement

    def get_measurements(self):
        return self.latest_measurement

    def start_fly(self):
        self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}

        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(500)  # 14.04.2025; 16:11; first version was 100ms now updated to 500ms to predict with 2Hz
        self.hoverTimer.start()

    def sendHoverCommand(self, emergency_stop_active, ai_control_active):
        if emergency_stop_active:
            self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}
            self.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, 0.0)
            return

        if ai_control_active:
            print("Implement AI Fly Controller !!!!")

        self.cf.commander.send_hover_setpoint(self.hover["x"], self.hover["y"], self.hover["yaw"], self.hover["height"])

    def updateHover(self, k, v):
        """
        Updates the hover dictionary based on the key and value.
        :param k: The key to update (e.g., "x", "y", "height").
        :param v: The value to update (e.g., 1 for forward, -1 for backward).

        self.hover = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "height": 0.5}
        """
        if k in self.hover:
            self.hover[k] += v * self.SPEED_FACTOR
            print(f"Updated hover: {self.hover}")
        else:
            print(f"Invalid hover key: {k}")
