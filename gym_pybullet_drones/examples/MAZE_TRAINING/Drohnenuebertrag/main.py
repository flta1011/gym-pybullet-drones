import sys

from cflib.utils import uri_helper
from controllers.drone_controller import DroneController
from gui.main_window import MainWindow
from PyQt6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    observation_type = "O8"
    action_type = "A2"
    model_type = "M3"
    model_path = "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/results/save-04.12.2025_21.26.56/final_model.zip"

    URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

    # Initialize drone controller
    drone_controller = DroneController(uri=URI, observation_type=observation_type, action_type=action_type, model_type=model_type, model_path=model_path)

    # Initialize GUI
    main_window = MainWindow(drone_controller)
    main_window.show()

    sys.exit(app.exec())
