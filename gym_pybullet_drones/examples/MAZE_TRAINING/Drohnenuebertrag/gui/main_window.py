import sys

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QLabel
from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, drone_controller):
        super().__init__()
        self.drone_controller = drone_controller

        # Window setup
        self.resize(800, 600)
        self.setWindowTitle("Fly to Wall")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Install event filter to capture key events regardless of focus
        self.installEventFilter(self)

        # Create the labels
        self.labels = {
            "MultiRanger values:": QtWidgets.QLabel("MultiRanger values"),
            "Front:": QtWidgets.QLabel("Front: ___.__ m"),
            "Back:": QtWidgets.QLabel("Back: ___.__ m"),
            "Left:": QtWidgets.QLabel("Left: ___.__ m"),
            "Right:": QtWidgets.QLabel("Right: ___.__ m"),
            "Up:": QtWidgets.QLabel("Up: ___.__ m"),
            "Down:": QtWidgets.QLabel("Down: ___.__ m"),
            "Yaw:": QtWidgets.QLabel("Yaw: ___.__"),
            "Pitch:": QtWidgets.QLabel("Pitch: ___.__"),
            "Roll:": QtWidgets.QLabel("Roll: ___.__"),
            "Position:": QtWidgets.QLabel("Position"),
            "StateEstimate X:": QtWidgets.QLabel("StateEstimate X: ___.__ m"),
            "StateEstimate Y:": QtWidgets.QLabel("StateEstimate Y: ___.__ m"),
            "StateEstimate Z:": QtWidgets.QLabel("StateEstimate Z: ___.__ m"),
            "AI:": QtWidgets.QLabel("AI"),
            "AI Control Action:": QtWidgets.QLabel("AI Control Action: ___"),
        }

        # Set the style for the Captions
        self.labels["MultiRanger values:"].setStyleSheet("font-weight: bold; text-decoration: underline;")
        self.labels["Position:"].setStyleSheet("font-weight: bold; text-decoration: underline;")
        self.labels["AI:"].setStyleSheet("font-weight: bold; text-decoration: underline;")

        # Create the buttons
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.start_button = QtWidgets.QPushButton("Start")
        self.emergency_stop_button = QtWidgets.QPushButton("Emergency stop")
        self.switch_view_button = QtWidgets.QPushButton("Switch View")  # Button to switch views
        self.toggle_ai_control_button = QtWidgets.QCheckBox("AI Control")
        self.toggle_ai_control_button.setTristate(False)

        # Create the layout
        layout = QtWidgets.QVBoxLayout()

        # Create the left layout for the labels
        left_layout = QtWidgets.QVBoxLayout()
        for label in self.labels.values():
            left_layout.addWidget(label)

        # Create the right layout with a QStackedWidget
        right_layout = QtWidgets.QVBoxLayout()
        self.stacked_widget = QtWidgets.QStackedWidget()

        # 3D Mapping Placeholder
        self.mapping_placeholder = QtWidgets.QWidget()
        self.mapping_placeholder.setStyleSheet("background-color: gray;")
        self.stacked_widget.addWidget(self.mapping_placeholder)

        # SLAM Map Placeholder
        self.slam_map_placeholder = QtWidgets.QWidget()
        self.slam_map_placeholder.setStyleSheet("background-color: lightblue;")
        self.stacked_widget.addWidget(self.slam_map_placeholder)

        right_layout.addWidget(self.stacked_widget)

        # Create the bottom layout for the buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.connect_button)
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.emergency_stop_button)
        bottom_layout.addWidget(self.switch_view_button)
        bottom_layout.addWidget(self.toggle_ai_control_button)
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

        self.ai_control_active = False

        # Connect the buttons to their functions
        self.connect_button.clicked.connect(self.connect)
        self.start_button.clicked.connect(self.on_start_fly)
        self.emergency_stop_button.clicked.connect(self.on_emergency_stop)
        self.switch_view_button.clicked.connect(self.switch_view)
        self.toggle_ai_control_button.stateChanged.connect(self.on_ai_control_checkbox_changed)

        # Create the 3D mapping
        self.scene_canvas = scene.SceneCanvas(keys=None)
        self.scene_canvas.unfreeze()
        self.scene_canvas.view = self.scene_canvas.central_widget.add_view()
        self.scene_canvas.view.bgcolor = "#ffffff"
        self.scene_canvas.view.camera = TurntableCamera(fov=10.0, distance=30.0, up="+z", center=(0.0, 0.0, 0.0))

        self.scene_canvas.unfreeze()

        # Add the SceneCanvas to the mapping placeholder
        layout_right_for_canvas = QtWidgets.QVBoxLayout(self.mapping_placeholder)
        layout_right_for_canvas.setContentsMargins(0, 0, 0, 0)
        layout_right_for_canvas.addWidget(self.scene_canvas.native)

        # Set the SceneCanvas to scale dynamically
        self.scene_canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.PLOT_CF = True
        self.last_pos = [0, 0, 0]
        self.pos_markers = visuals.Markers()
        self.meas_markers = visuals.Markers()
        self.position_data = np.array([0, 0, 0], ndmin=2)
        self.measurement_data = np.array([0, 0, 0], ndmin=2)
        self.lines = []

        self.scene_canvas.view.add(self.pos_markers)
        self.scene_canvas.view.add(self.meas_markers)
        for _ in range(6):
            line = visuals.Line()
            self.lines.append(line)
            self.scene_canvas.view.add(line)

        # Add XYZ axes to the SceneCanvas
        scene.visuals.XYZAxis(parent=self.scene_canvas.view.scene)

        self.scene_canvas.freeze()

        # Register callbacks
        self.drone_controller.set_position_callback(self.update_position_labels)
        self.drone_controller.set_measurement_callback(self.update_measurement_labels)
        self.drone_controller.set_update_slam_map_callback(self.update_slam_map)
        self.drone_controller.set_ai_action_callback(self.update_ai_action_label)
        self.start_fly_callback = self.drone_controller.start_fly
        self.emergency_stop_callback = self.drone_controller.emergency_stop
        self.drone_controller.set_update_slam_map_callback(self.update_slam_map)

        # Define key mappings for key press events
        # Define key mappings for key press events
        self.key_press_mappings = {
            QtCore.Qt.Key.Key_A: lambda: self.drone_controller.updateHover("y", 1),  # Move left
            QtCore.Qt.Key.Key_D: lambda: self.drone_controller.updateHover("y", -1),  # Move right
            QtCore.Qt.Key.Key_W: lambda: self.drone_controller.updateHover("x", 1),  # Move forward
            QtCore.Qt.Key.Key_S: lambda: self.drone_controller.updateHover("x", -1),  # Move backward
            QtCore.Qt.Key.Key_Left: lambda: self.drone_controller.updateHover("yaw", -70),  # Rotate counterclockwise
            QtCore.Qt.Key.Key_Right: lambda: self.drone_controller.updateHover("yaw", 70),  # Rotate clockwise
            QtCore.Qt.Key.Key_Z: lambda: self.drone_controller.updateHover("yaw", -200),  # Fast rotate counterclockwise
            QtCore.Qt.Key.Key_X: lambda: self.drone_controller.updateHover("yaw", 200),  # Fast rotate clockwise
            QtCore.Qt.Key.Key_Up: lambda: self.drone_controller.updateHover("height", 0.1),  # Ascend
            QtCore.Qt.Key.Key_Down: lambda: self.drone_controller.updateHover("height", -0.1),  # Descend
            QtCore.Qt.Key.Key_Space: self.on_emergency_stop,  # Emergency stop
            QtCore.Qt.Key.Key_K: self.on_ai_control_checkbox_changed,  # Toggle AI control
        }

        # Define key mappings for key release events
        self.key_release_mappings = {
            QtCore.Qt.Key.Key_A: lambda: self.drone_controller.updateHover("y", 0),  # Stop left/right movement
            QtCore.Qt.Key.Key_D: lambda: self.drone_controller.updateHover("y", 0),  # Stop left/right movement
            QtCore.Qt.Key.Key_W: lambda: self.drone_controller.updateHover("x", 0),  # Stop forward/backward movement
            QtCore.Qt.Key.Key_S: lambda: self.drone_controller.updateHover("x", 0),  # Stop forward/backward movement
            QtCore.Qt.Key.Key_Left: lambda: self.drone_controller.updateHover("yaw", 0),  # Stop rotation
            QtCore.Qt.Key.Key_Right: lambda: self.drone_controller.updateHover("yaw", 0),  # Stop rotation
            QtCore.Qt.Key.Key_Z: lambda: self.drone_controller.updateHover("yaw", 0),  # Stop fast rotation
            QtCore.Qt.Key.Key_X: lambda: self.drone_controller.updateHover("yaw", 0),  # Stop fast rotation
            QtCore.Qt.Key.Key_Up: lambda: self.drone_controller.updateHover("height", 0),  # Stop ascending/descending
            QtCore.Qt.Key.Key_Down: lambda: self.drone_controller.updateHover("height", 0),  # Stop ascending/descending
        }

    def update_position_labels(self, position):
        self.labels["StateEstimate X:"].setText(f"StateEstimate X: {position[0]:.2f} m")
        self.labels["StateEstimate Y:"].setText(f"StateEstimate Y: {position[1]:.2f} m")
        self.labels["StateEstimate Z:"].setText(f"StateEstimate Z: {position[2]:.2f} m")

    def update_measurement_labels(self, measurement):
        self.labels["Front:"].setText(f"Front: {measurement['front']:.2f} m")
        self.labels["Back:"].setText(f"Back: {measurement['back']:.2f} m")
        self.labels["Left:"].setText(f"Left: {measurement['left']:.2f} m")
        self.labels["Right:"].setText(f"Right: {measurement['right']:.2f} m")
        self.labels["Up:"].setText(f"Up: {measurement['up']:.2f} m")
        self.labels["Down:"].setText(f"Down: {measurement['down']:.2f} m")
        self.labels["Yaw:"].setText(f"Yaw: {measurement['yaw']:.2f}")
        self.labels["Pitch:"].setText(f"Pitch: {measurement['pitch']:.2f}")
        self.labels["Roll:"].setText(f"Roll: {measurement['roll']:.2f}")

    def update_ai_action_label(self, action):
        """
        Update the AI action label with the provided action value
        :param action: The AI's most recent action
        """
        if action is not None:
            # Assuming action is a numeric value
            self.labels["AI Control Action:"].setText(f"AI Control Action: {action}")

    def update_slam_map(self, slam_map):
        if slam_map is not None:
            try:
                # Get the squeezed 2D version of the map
                map_2d = slam_map.squeeze()
                h, w = map_2d.shape[:2]

                # Create a colored map for better visualization
                colored_map = np.zeros((h, w, 3), dtype=np.uint8)

                # SLAM map values:
                # 0 = unknown (gray)
                # 50 = wall (black)
                # 125 = visited (light blue)
                # 200 = free space (white)
                # 255 = current position (red)

                # Unknown areas (value 0) - gray
                colored_map[map_2d == 0] = [128, 128, 128]

                # Free space (value 200) - white
                colored_map[map_2d == 200] = [255, 255, 255]

                # Walls (value 50) - black
                colored_map[map_2d == 50] = [0, 0, 0]

                # Visited areas (value 125) - light blue
                colored_map[map_2d == 125] = [173, 216, 230]

                # Current position (value 255) - red
                colored_map[map_2d == 255] = [255, 0, 0]

                # Convert to QImage and display
                qimg = QImage(colored_map.data, w, h, w * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                if not hasattr(self, "slam_map_label"):
                    self.slam_map_label = QLabel(self.slam_map_placeholder)
                    self.slam_map_label.setGeometry(0, 0, self.slam_map_placeholder.width(), self.slam_map_placeholder.height())
                    self.slam_map_label.setScaledContents(True)
                    self.slam_map_placeholder.setLayout(QtWidgets.QVBoxLayout())
                    self.slam_map_placeholder.layout().addWidget(self.slam_map_label)

                self.slam_map_label.setPixmap(pixmap)

            except Exception as e:
                print(f"Error in update_slam_map: {e}")
                # Fall back to a gray placeholder if there's an error
                placeholder = np.ones((64, 64, 3), dtype=np.uint8) * 128
                qimg = QImage(placeholder.data, 64, 64, 64 * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                if not hasattr(self, "slam_map_label"):
                    self.slam_map_label = QLabel(self.slam_map_placeholder)
                    self.slam_map_label.setGeometry(0, 0, self.slam_map_placeholder.width(), self.slam_map_placeholder.height())
                    self.slam_map_label.setScaledContents(True)
                    self.slam_map_placeholder.setLayout(QtWidgets.QVBoxLayout())
                    self.slam_map_placeholder.layout().addWidget(self.slam_map_label)

                self.slam_map_label.setPixmap(pixmap)

    def switch_view(self):
        # Switch between the 3D mapping view and the SLAM map view
        current_index = self.stacked_widget.currentIndex()
        new_index = (current_index + 1) % self.stacked_widget.count()
        self.stacked_widget.setCurrentIndex(new_index)

        # Make sure main window retains focus after view switch
        self.activateWindow()
        self.setFocus()

    def connect(self):
        if self.drone_controller:
            self.drone_controller.connect()
        else:
            print("Error: DroneController is not initialized.")

    def on_start_fly(self):
        if self.start_fly_callback:
            self.start_fly_callback()
            print("Start fly callback executed.")
        else:
            print("Error: Start fly callback is not set.")

    def on_emergency_stop(self):
        if self.emergency_stop_callback:
            self.emergency_stop_callback()

    def on_ai_control_checkbox_changed(self, state):
        """
        Handle the state change of the AI control checkbox and update the DroneController.
        :param state: The new state of the checkbox (0 for unchecked, 2 for checked).
        """
        if state == 2:  # Fully checked
            self.ai_control_active = True
            print(f"AI control state updated to: {self.ai_control_active}")
        elif state == 0:  # Fully unchecked
            self.ai_control_active = False
            print(f"AI control state updated to: {self.ai_control_active}")
        else:
            print(f"AI control state is in an intermediate state: {state}")

    def keyPressEvent(self, event):
        """
        Handle key press events using the key mappings.
        """
        if not event.isAutoRepeat() and event.key() in self.key_press_mappings:
            self.key_press_mappings[event.key()]()

    def keyReleaseEvent(self, event):
        """
        Handle key release events using the key release mappings.
        """
        if not event.isAutoRepeat() and event.key() in self.key_release_mappings:
            self.key_release_mappings[event.key()]()

    def closeEvent(self, event):
        if self.drone_controller:
            self.drone_controller.cf.close_link()
        event.accept()
        sys.exit(0)

    def eventFilter(self, obj, event):
        """
        Global event filter to capture key events regardless of focus
        """
        if event.type() == QtCore.QEvent.Type.KeyPress:
            if not event.isAutoRepeat() and event.key() in self.key_press_mappings:
                self.key_press_mappings[event.key()]()
                return True
        elif event.type() == QtCore.QEvent.Type.KeyRelease:
            if not event.isAutoRepeat() and event.key() in self.key_release_mappings:
                self.key_release_mappings[event.key()]()
                return True

        # Pass other events on
        return super().eventFilter(obj, event)
