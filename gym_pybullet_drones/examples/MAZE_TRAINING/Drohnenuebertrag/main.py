import sys

from cflib.utils import uri_helper
from controllers.drone_controller import DroneController
from gui.main_window import MainWindow
from PyQt6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # observation_type = "O8"
    # action_type = "A2"
    # model_type = "M3"
    # model_path = "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/results/save-04.12.2025_21.26.56/final_model.zip"

    observation_type = "O9"
    action_type = "A2"
    model_type = "M5"
    model_path = "/home/moritz_s/Desktop/M5_R6_O9_A2_TR1_T1_20250411-180656_Flo_Model_One/save-04.11.2025_18.06.56/final_model.zip"

    # SECTION Depending on which drone you set 60 or 80 for the channel
    URI = uri_helper.uri_from_env(default="radio://0/60/2M/E7E7E7E7E7")

    # Initialize drone controller
    drone_controller = DroneController(uri=URI, observation_type=observation_type, action_type=action_type, model_type=model_type, model_path=model_path)

    # Initialize GUI
    main_window = MainWindow(drone_controller)
    main_window.show()

    sys.exit(app.exec())
