import numpy as np
from PyQt6 import QtCore, QtWidgets
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

        # Create the labels
        self.labels = {
            "MultiRanger values:": QtWidgets.QLabel("MultiRanger values"),
            "Front:": QtWidgets.QLabel("Front: ___.__ mm"),
            "Back:": QtWidgets.QLabel("Back: ___.__ mm"),
            "Left:": QtWidgets.QLabel("Left: ___.__ mm"),
            "Right:": QtWidgets.QLabel("Right: ___.__ mm"),
            "Up:": QtWidgets.QLabel("Up: ___.__ mm"),
            "Down:": QtWidgets.QLabel("Down: ___.__ mm"),
            "Yaw:": QtWidgets.QLabel("Yaw: ___.__"),
            "Pitch:": QtWidgets.QLabel("Pitch: ___.__"),
            "Roll:": QtWidgets.QLabel("Roll: ___.__"),
            "Position:": QtWidgets.QLabel("Position"),
            "StateEstimate X:": QtWidgets.QLabel("StateEstimate X: ___.__ mm"),
            "StateEstimate Y:": QtWidgets.QLabel("StateEstimate Y: ___.__ mm"),
            "StateEstimate Z:": QtWidgets.QLabel("StateEstimate Z: ___.__ mm"),
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
        self.start_button.clicked.connect(self.start_fly)
        self.emergency_stop_button.clicked.connect(self.emergency_stop)
        self.switch_view_button.clicked.connect(self.switch_view)
        self.toggle_ai_control_button.stateChanged.connect(self.drone_controller.toggle_ai_control)

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

    def switch_view(self):
        # Switch between the 3D mapping view and the SLAM map view
        current_index = self.stacked_widget.currentIndex()
        new_index = (current_index + 1) % self.stacked_widget.count()
        self.stacked_widget.setCurrentIndex(new_index)

    def connect(self):
        self.drone_controller.connect()

    def start_fly(self):
        self.drone_controller.start_fly()

    def emergency_stop(self):
        self.drone_controller.emergency_stop()

    def toggle_ai_control(self, state):
        # Toggle AI control
        self.drone_controller.toggle_ai_control(state)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == QtCore.Qt.Key.Key_Space:
                self.emergency_stop()
            elif event.key() == QtCore.Qt.Key.Key_K:
                self.toggle_ai_control()

            if event.key() == QtCore.Qt.Key.Key_Left:
                self.drone_controller.updateHover("y", 1)
            if event.key() == QtCore.Qt.Key.Key_Right:
                self.drone_controller.updateHover("y", -1)
            if event.key() == QtCore.Qt.Key.Key_Up:
                self.drone_controller.updateHover("x", 1)
            if event.key() == QtCore.Qt.Key.Key_Down:
                self.drone_controller.updateHover("x", -1)
            if event.key() == QtCore.Qt.Key.Key_A:
                self.drone_controller.updateHover("yaw", -70)
            if event.key() == QtCore.Qt.Key.Key_D:
                self.drone_controller.updateHover("yaw", 70)
            if event.key() == QtCore.Qt.Key.Key_Z:
                self.drone_controller.updateHover("yaw", -200)
            if event.key() == QtCore.Qt.Key.Key_X:
                self.drone_controller.updateHover("yaw", 200)
            if event.key() == QtCore.Qt.Key.Key_W:
                self.drone_controller.updateHover("height", 0.1)
            if event.key() == QtCore.Qt.Key.Key_S:
                self.drone_controller.updateHover("height", -0.1)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == QtCore.Qt.Key.Key_Left:
                self.drone_controller.updateHover("y", 0)
            if event.key() == QtCore.Qt.Key.Key_Right:
                self.drone_controller.updateHover("y", 0)
            if event.key() == QtCore.Qt.Key.Key_Up:
                self.drone_controller.updateHover("x", 0)
            if event.key() == QtCore.Qt.Key.Key_Down:
                self.drone_controller.updateHover("x", 0)
            if event.key() == QtCore.Qt.Key.Key_A:
                self.drone_controller.updateHover("yaw", 0)
            if event.key() == QtCore.Qt.Key.Key_D:
                self.drone_controller.updateHover("yaw", 0)
            if event.key() == QtCore.Qt.Key.Key_W:
                self.drone_controller.updateHover("height", 0)
            if event.key() == QtCore.Qt.Key.Key_S:
                self.drone_controller.updateHover("height", 0)
            if event.key() == QtCore.Qt.Key.Key_Z:
                self.drone_controller.updateHover("yaw", 0)
            if event.key() == QtCore.Qt.Key.Key_X:
                self.drone_controller.updateHover("yaw", 0)

    def closeEvent(self, event):
        if self.cf is not None:
            self.cf.close_link()
        event.accept()
        sys.exit(0)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    drone_controller = None  # Replace with actual drone controller instance
    slam_manager = None  # Replace with actual SLAM manager instance
    ai_controller = None  # Replace with actual AI controller instance

    window = MainWindow(drone_controller, slam_manager, ai_controller)
    window.show()
    sys.exit(app.exec())
