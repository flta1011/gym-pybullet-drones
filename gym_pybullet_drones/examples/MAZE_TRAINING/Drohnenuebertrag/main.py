import sys

from controllers.ai_controller import AIController
from controllers.drone_controller import DroneController
from controllers.slam_manager import SLAMManager
from gui.main_window import MainWindow
from PyQt6 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Initialize components
    drone_controller = DroneController(uri="drone_uri")
    slam_manager = SLAMManager()
    ai_controller = AIController(model_path="models/model_v1.zip")

    # Initialize GUI
    main_window = MainWindow(drone_controller, slam_manager, ai_controller)
    main_window.show()

    sys.exit(app.exec())
