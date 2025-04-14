import sys

from controllers.ai_controller import AIController
from controllers.drone_controller import DroneController
from gui.main_window import MainWindow
from PyQt6 import QtWidgets

from gym_pybullet_drones.examples.MAZE_TRAINING.Drohnenuebertrag.controllers.obs_manager import (
    OBSManager,
)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Initialize components
    drone_controller = DroneController(uri="drone_uri")
    obs_manager = OBSManager()
    ai_controller = AIController(model_path="models/model_v1.zip")

    # Initialize GUI
    main_window = MainWindow(drone_controller, obs_manager, ai_controller)
    main_window.show()

    sys.exit(app.exec())
