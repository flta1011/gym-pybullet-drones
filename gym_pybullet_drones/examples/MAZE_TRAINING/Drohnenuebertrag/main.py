import sys

from controllers.drone_controller import DroneController
from gui.main_window import MainWindow
from PyQt6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    observation_type = "O8"
    action_type = "A4"
    model_type = "M1"
    model_path = "model_path"

    # Initialize drone controller
    drone_controller = DroneController(uri="drone_uri", observation_type=observation_type, action_type=action_type, model_type=model_type, model_path=model_path)

    # Initialize GUI
    main_window = MainWindow(drone_controller)
    main_window.show()

    sys.exit(app.exec())
