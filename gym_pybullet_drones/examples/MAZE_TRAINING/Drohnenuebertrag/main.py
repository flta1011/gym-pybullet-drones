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

    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/moritz_s/Desktop/M5_R6_O9_A2_TR1_T1_20250411-180656_Flo_Model_One/save-04.11.2025_18.06.56/final_model.zip"

    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/M6_R6_O8_A3_TR1_T1_20250415-211929_schwere_Mazes_Flo_Model_Two/save-04.15.2025_21.19.29/final_model.zip"

    ## Fliegt von rechts Ecke nach links nicht soo schlecht
    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/Alex_modelle/1/M6_R6_O8_A3_TR1_T1_20250409-210009/save-04.09.2025_21.00.09/results/save-04.09.2025_21.00.09/_250000_steps.zip"

    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/Alex_modelle/3/M6_R6_O8_A3_TR1_T1_20250410-195535/save-04.10.2025_19.55.35/results/save-04.10.2025_19.55.35/_750000_steps.zip"

    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/20250426_SAC_mehr_Abstand_FLO/M6_R6_O8_A3_TR1_T1_20250425-233402_neues_Maz_29/save-04.25.2025_23.34.02/final_model.zip"

    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/SAC_025_Abstand/_700000_steps.zip"

    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/SAC_025_Abstand_800000/_800000_steps.zip"

    #############################
    # New Models 27.04.2025 (Flo)
    #############################

    # # 40 last clipped actions
    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/moritz_s/Desktop/20250427_Models/M5_R6_O9_A2_TR1_T1_20250427-010154_DQN-Neue_Mazes_0-3_Abstand/_1700000_steps.zip"

    # 100 last clipped actions
    # observation_type = "O10"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/20250427_Models/M6_R6_O10_A3_TR1_T1_20250427-003400_SAC-O10_0.35-abstand-neue-Mazes-ohne-SLAM/_2200000_steps.zip"

    ###ALEX###

    # observation_type = "O10"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/alex/Desktop/FundEModelle/M6_R6_O10_A3_TR1_T1_20250427-003400_SAC-O10_0.35-abstand-neue-Mazes-ohne-SLAM/_2200000_steps.zip"

    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/alex/Desktop/FundEModelle/M5_R6_O9_A2_TR1_T1_20250427-010154_DQN-Neue_Mazes_0-3_Abstand/_1700000_steps.zip"

    #############
    # Test Series
    #############

    # 1
    observation_type = "O8"
    action_type = "A3"
    model_type = "M6"
    model_path = "/home/moritz_s/Desktop/Test_Series/01/M6_R6_O8_A3_TR1_T1_20250415_211929_schwere_Mazes_SAC_alt_2Hz_100LA/save-04.15.2025_21.19.29/final_model.zip"

    # #2
    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/moritz_s/Desktop/Test_Series/02/M5_R6_O9_A2_TR1_T1_20250411_180656_DQN_alt_01_Abstand_2hz_20LA/save-04.11.2025_18.06.56/final_model.zip"

    # #3
    # observation_type = "O8"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/Test_Series/03/M6_R6_O8_A3_TR1_T1_20250426_131746_SAC_025A_20LA_2Hz_Abstand_modell_nachmittags_Vr_Rechner_/_800000_steps.zip"

    # #4
    # observation_type = "O10"
    # action_type = "A3"
    # model_type = "M6"
    # model_path = "/home/moritz_s/Desktop/Test_Series/04/M6_R6_O10_A3_TR1_T1_20250427_003400_SAC_O10_035_Abstand_5Hz_100LA_neue_Mazes_ohne_SLAM_neu_ueber_nacht/_2200000_steps.zip"

    # #5
    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/moritz_s/Desktop/Test_Series/05/M5_R6_O9_A2_TR1_T1_20250427_010154_DQN_Neue_Mazes_03_Abstand_5Hz_40LA_VR_Rechner_Trainier_ueber_nacht/_1700000_steps.zip"

    # #6
    # observation_type = "O9"
    # action_type = "A2"
    # model_type = "M5"
    # model_path = "/home/moritz_s/Desktop/Test_Series/06/M5_R6_O9_A2_TR1_T1_20250427_110353_DQN_VR_Rechner_6LA_5Hz_03_Abstand_neustes/_1200000_steps.zip"

    # SECTION Depending on which drone you set 60 or 80 for the channel
    URI = uri_helper.uri_from_env(default="radio://0/60/2M/E7E7E7E7E7")

    # Initialize drone controller
    drone_controller = DroneController(uri=URI, observation_type=observation_type, action_type=action_type, model_type=model_type, model_path=model_path)

    # Initialize GUI
    main_window = MainWindow(drone_controller)
    main_window.show()

    sys.exit(app.exec())
