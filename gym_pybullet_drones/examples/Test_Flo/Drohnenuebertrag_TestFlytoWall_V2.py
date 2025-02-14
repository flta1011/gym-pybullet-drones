"""
 When the application is started the Crazyflie will hover at 0.5m.
 The Crazyflie then hovers without movement until the Enter key is pressed.
 If the Enter key is pressed the Crazyflie will start moving after the prediction
 of the best ai model of "TestFlytoWall".
    - It will only move
        - forward (towards the wall)
        - backwards (away from the wall)
        - stand still
The movement by ai model can be stoped by pressing the Enter key again.

To do an automated landing / emergency landing, press the 'Space' key.

For the example to run the following hardware is needed:
 * Crazyflie 2.0
 * Crazyradio PA
 * Flow deck
 * Multiranger deck
"""

import logging
import math
import sys
import time

import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QLabel, QPushButton

from stable_baselines3 import PPO
from preparation_prediction_model_for_real_flight import get_PPO_Predcitions_1D_Observation
pathV1 = 'gym_pybullet_drones/Saved-Models_FlyToWall/save-02.11.2025_16.51.09_V1_1D-Observation/best_model.zip'
modelV1 = PPO.load(pathV1)

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Set the sensor threshold (in mm)
SENSOR_TH = 4000
# Set the speed factor for moving and rotating
SPEED_FACTOR = 0.5


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, URI):
        QtWidgets.QMainWindow.__init__(self)

        self.emergency_stop_active = False
        self.latest_position = None
        self.latest_measurement = None
        self.ai_control_active = False

        self.resize(700, 500)
        self.setWindowTitle("Fly to Wall")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Create the labels
        self.labels = {
            "MultiRanger values:": QLabel("MultiRanger values:"),
            "Front:": QLabel("Front: ___.__ mm"),
            "Back:": QLabel("Back: ___.__ mm"),
            "Left:": QLabel("Left: ___.__ mm"),
            "Right:": QLabel("Right: ___.__ mm"),
            "Up:": QLabel("Up: ___.__ mm"),
            "Down:": QLabel("Down: ___.__ mm"),
            "Yaw:": QLabel("Yaw: ___.__"),
            "Pitch:": QLabel("Pitch: ___.__"),
            "Roll:": QLabel("Roll: ___.__"),
            "Position:": QLabel("Position:"),
            "StateEstimate X:": QLabel("StateEstimate X: ___.__ mm"),
            "StateEstimate Y:": QLabel("StateEstimate Y: ___.__ mm"),
            "StateEstimate Z:": QLabel("StateEstimate Z: ___.__ mm"),
            "AI Control Action:": QLabel("AI Control Action: ___")
        }

        # Create the buttons
        self.connect_button = QPushButton("Connect")
        self.start_button = QPushButton("Start")
        self.emergency_stop_button = QPushButton("Emergency stop")
        self.toggle_ai_control = QtWidgets.QCheckBox("AI Control")
        #self.reset_emergency_stop_button = QPushButton("Reset Emergency Stop")

        # Create the layout
        layout = QtWidgets.QVBoxLayout()

        # Create the left layout for the labels
        left_layout = QtWidgets.QVBoxLayout()
        for label in self.labels.values():
            left_layout.addWidget(label)

        # Create the right layout for the 3D mapping placeholder
        right_layout = QtWidgets.QVBoxLayout()
        self.mapping_placeholder = QtWidgets.QWidget()
        #self.mapping_placeholder.setFixedSize(400, 400)  # Placeholder size
        self.mapping_placeholder.setStyleSheet("background-color: gray;")
        right_layout.addWidget(self.mapping_placeholder)

        # Create the bottom layout for the buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.connect_button)
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.emergency_stop_button)
        bottom_layout.addWidget(self.toggle_ai_control)
        #bottom_layout.addWidget(self.reset_emergency_stop_button)
        bottom_layout.addStretch()

        # Create the main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        layout.addLayout(main_layout)
        layout.addLayout(bottom_layout)

        # Create a central widget and set the layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect the buttons to their functions
        self.connect_button.clicked.connect(self.connect)
        self.start_button.clicked.connect(self.start_fly)
        self.emergency_stop_button.clicked.connect(self.emergency_stop)
        self.toggle_ai_control.stateChanged.connect(self.toggle_ai_control_changed)
        #self.reset_emergency_stop_button.clicked.connect(self.reset_emergency_stop)

    def toggle_ai_control_changed(self, state):
        self.ai_control_active = (state == 2) # (state == QtCore.Qt.CheckState.Checked)
        # print("DEBUG: state =", state)
        # print("DEBUG: self.ai_control_active =", self.ai_control_active)

    def connect(self):
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        # Connect callbacks from the Crazyflie API
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)

        # Connect to the Crazyflie
        if hasattr(self, 'cf') and self.cf.is_connected():
            self.cf.close_link()
            time.sleep(1)
        self.cf.open_link(URI)
        
        # Arm the Crazyflie
        self.cf.platform.send_arming_request(True)

    def start_fly(self):
        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.5}

        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(100)
        self.hoverTimer.start()

    def sendHoverCommand(self):
        if not self.emergency_stop_active:
            if self.ai_control_active:
                self.AIFlycommands()

            self.cf.commander.send_hover_setpoint(
                self.hover['x'], self.hover['y'], self.hover['yaw'],
                self.hover['height'])

    def updateHover(self, k, v):
        if (k != 'height'):
            self.hover[k] = v * SPEED_FACTOR
        else:
            self.hover[k] += v

    def disconnected(self, URI):
        print('Disconnected')

    def connected(self, URI):
        print('We are now connected to {}'.format(URI))

        # The definition of the logconfig can be made before connecting
        lpos = LogConfig(name='Position', period_in_ms=100)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')

        try:
            self.cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Position log config, bad configuration.')

        lmeas = LogConfig(name='Meas', period_in_ms=100)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.up')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        lmeas.add_variable('range.zrange')
        lmeas.add_variable('stabilizer.roll')
        lmeas.add_variable('stabilizer.pitch')
        lmeas.add_variable('stabilizer.yaw')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Measurement log config, bad configuration.')

    def pos_data(self, timestamp, data, logconf):
        position = [
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z']
        ]

        self.latest_position = position

        self.labels["StateEstimate X:"].setText(f"StateEstimate X: {position[0]:.2f} mm")
        self.labels["StateEstimate Y:"].setText(f"StateEstimate Y: {position[1]:.2f} mm")
        self.labels["StateEstimate Z:"].setText(f"StateEstimate Z: {position[2]:.2f} mm")

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            'roll': data['stabilizer.roll'],
            'pitch': data['stabilizer.pitch'],
            'yaw': data['stabilizer.yaw'],
            'front': data['range.front'],
            'back': data['range.back'],
            'up': data['range.up'],
            'down': data['range.zrange'],
            'left': data['range.left'],
            'right': data['range.right']
        }

        self.latest_measurement = measurement

        self.labels["Front:"].setText(f"Front: {measurement['front']:.2f} mm")
        self.labels["Back:"].setText(f"Back: {measurement['back']:.2f} mm")
        self.labels["Left:"].setText(f"Left: {measurement['left']:.2f} mm")
        self.labels["Right:"].setText(f"Right: {measurement['right']:.2f} mm")
        self.labels["Up:"].setText(f"Up: {measurement['up']:.2f} mm")
        self.labels["Down:"].setText(f"Down: {measurement['down']:.2f} mm")
        self.labels["Yaw:"].setText(f"Yaw: {measurement['yaw']:.2f}")
        self.labels["Pitch:"].setText(f"Pitch: {measurement['pitch']:.2f}")
        self.labels["Roll:"].setText(f"Roll: {measurement['roll']:.2f}")

    def closeEvent(self, event):
        if (self.cf is not None):
            self.cf.close_link()
        event.accept()
        sys.exit(0)

    def keyPressEvent(self, event):
        if (not event.isAutoRepeat()):
            if (event.key() == QtCore.Qt.Key.Key_Space):
                self.emergency_stop()
            if not self.ai_control_active:
                if (event.key() == QtCore.Qt.Key.Key_Left):
                    self.updateHover('y', 1)
                if (event.key() == QtCore.Qt.Key.Key_Right):
                    self.updateHover('y', -1)
                if (event.key() == QtCore.Qt.Key.Key_Up):
                    self.updateHover('x', 1)
                if (event.key() == QtCore.Qt.Key.Key_Down):
                    self.updateHover('x', -1)
                if (event.key() == QtCore.Qt.Key.Key_A):
                    self.updateHover('yaw', -70)
                if (event.key() == QtCore.Qt.Key.Key_D):
                    self.updateHover('yaw', 70)
                if (event.key() == QtCore.Qt.Key.Key_Z):
                    self.updateHover('yaw', -200)
                if (event.key() == QtCore.Qt.Key.Key_X):
                    self.updateHover('yaw', 200)
                if (event.key() == QtCore.Qt.Key.Key_W):
                    self.updateHover('height', 0.1)
                if (event.key() == QtCore.Qt.Key.Key_S):
                    self.updateHover('height', -0.1)

    def keyReleaseEvent(self, event):
        if (not event.isAutoRepeat()):
            if not self.ai_control_active:
                if (event.key() == QtCore.Qt.Key.Key_Left):
                    self.updateHover('y', 0)
                if (event.key() == QtCore.Qt.Key.Key_Right):
                    self.updateHover('y', 0)
                if (event.key() == QtCore.Qt.Key.Key_Up):
                    self.updateHover('x', 0)
                if (event.key() == QtCore.Qt.Key.Key_Down):
                    self.updateHover('x', 0)
                if (event.key() == QtCore.Qt.Key.Key_A):
                    self.updateHover('yaw', 0)
                if (event.key() == QtCore.Qt.Key.Key_D):
                    self.updateHover('yaw', 0)
                if (event.key() == QtCore.Qt.Key.Key_W):
                    self.updateHover('height', 0)
                if (event.key() == QtCore.Qt.Key.Key_S):
                    self.updateHover('height', 0)
                if (event.key() == QtCore.Qt.Key.Key_Z):
                    self.updateHover('yaw', 0)
                if (event.key() == QtCore.Qt.Key.Key_X):
                    self.updateHover('yaw', 0)

    def emergency_stop(self):
        self.emergency_stop_active = True
        self.cf.commander.send_stop_setpoint()
        print("Emergency stop")

    # def reset_emergency_stop(self):
    #     self.emergency_stop_active = False
    #     self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.0}
    #     self.hoverTimer.stop()
        
    def AIFlycommands(self):
        if self.ai_control_active:
            if self.latest_measurement is not None:
                observation = [self.latest_measurement['front']] # 1D observation
                action, _states = modelV1.predict(observation, deterministic=True)
                self.labels["AI Control Action:"].setText(f"AI Control Action: {action}")

            # Übersetzer
            if action == 0:
                self.updateHover('x', 1)
            elif action == 1:
                self.updateHover('x', -1)
            elif action == 2:
                self.updateHover('x', 0)
            else:
                print("Error: Action not found")

if __name__ == '__main__':
    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow(URI)
    win.show()
    appQt.exec()
