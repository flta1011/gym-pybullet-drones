import argparse
import csv
import os
import shutil
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecTransposeImage

from gym_pybullet_drones.examples.MAZE_TRAINING.BaseAviary_MAZE_TRAINING import (
    BaseRLAviary_MAZE_TRAINING,
)
from gym_pybullet_drones.examples.MAZE_TRAINING.custom_CNN_V0_0 import (
    CustomCNNFeatureExtractor,
)
from gym_pybullet_drones.examples.MAZE_TRAINING.custom_NN_V0_0 import (
    CustomNNFeatureExtractor,
)
from gym_pybullet_drones.examples.MAZE_TRAINING.Logger_MAZE_TRAINING_BUGGY import Logger
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)
from gym_pybullet_drones.utils.utils import str2bool, sync

#####################################################################################################
###########################INITIAL-SETTINGS DES REPOSITORYS##########################################
#####################################################################################################

####### TRAINING-MODE #######
Training_Mode = "Training"  # "Training" oder "Test"
GUI_Mode = "Train"  # "Train" oder "Test" oder "NoGUI"

####### GUI-SETTINGS (u.a. Debug-GUI) #######
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_ADVANCED_STATUS_PLOT = False
DEFAULT_DASH_ACTIVE = False

####### Hyperparameter-Set speichern #######
DEFAULT_SAVE_HYPERPARAMETER_SET = False  # auf false belassen, wird unten auf true gesetzt, sofern gespeichert werden soll (bei Bedarf unten nachlesen)
### Hier den Beschreibungstext für das Hyperparameter-Set einfügen ####
HyperparameterSetDescription = "FT:#Neue Hyperparameter-Sets, weil mit dem Abstand von 0,15 auch bei gutem Flugverhalten sehr viel negativer Reward gesammt wurde --> Mazes sind zu eng--> auf 0,1 verringert + belohnung für Kollision auf 0,75 reduziert"

###### Verwendung von gespeicherten Hyperparameter-Sets #######
DEFAULT_USE_SAVED_HYPERPARAMETER_SET = True
HyperparameterSetIDtoLoad = "SET_20250410-011939"
# SET_20250409-233159 = am 9.4.25 besprochenen Hyperparameter-Set, mit dem Alex mit DQN erste gute Ergebnisse erzielt hat
# SET_20250410-011939 = Belohnung für nicht Kollision auf 0,75 reduziert (vorher 1)


####### Verwendung von Pretrained-Modellen #######
DEFAULT_USE_PRETRAINED_MODEL = False
# DEFAULT_PRETRAINED_MODEL_PATH = '/home/florian/Documents/gym-pybullet-drones/results/durchgelaufen-DQN/final_model.zip'
# DEFAULT_PRETRAINED_MODEL_PATH = "/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/results/save-03.07.2025_02.23.46/best_model.zip"
DEFAULT_PRETRAINED_MODEL_PATH = (
    "/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/Auswertung_der_Modelle_Archieve/M6_R6_O5_A3_TR1_T1_20250407-013856/save-04.07.2025_01.38.56/best_model.zip"
)
DEFAULT_PRETRAINED_MODEL_PATH = (
    "/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/Auswertung_der_Modelle_Archieve/M5_R6_O7_A2_TR1_T1_20250410-093525/save-04.10.2025_09.35.25/best_model.zip"
)

###############################################################################
######################## TRAINING-SETTINGS (Hyperparameter)#################
##############################################################################

Ziel_Training_TIME_In_Simulation = 150 * 60 * 60  # 5 Stunden
DEFAULT_EVAL_FREQ = 5 * 1e4
DEFAULT_EVAL_EPISODES = 1
# DEFAULT_TRAIN_TIMESTEPS = 8 * 1e5  # nach 100000 Steps sollten schon mehrbahre Erkenntnisse da sein
DEFAULT_TARGET_REWARD = 99999
DEFAULT_NUMBER_LAST_ACTIONS = 20
DEFAULT_PYB_FREQ = 100
DEFAULT_CTRL_FREQ = 50
DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ = 2  # mit 5hz fliegt die Drohne noch zu oft an die Wand, ohne das das Pushback aktiv werden kann (mit Drehung aktiv) -> 10 HZ
DEFAULT_EPISODE_LEN_SEC = 5 * 60  # 15 * 60
DEFAULT_PUSHBACK_ACTIVE = False
DEFAULT_EVAL_FREQ = 5 * 1e4
DEFAULT_EVAL_EPISODES = 1
NumberOfInterationsTillNextCheckpoint = 250000  # Anzahl Steps, bis ein Modell als .zip gespeichert wird
DEFAULT_TRAIN_TIMESTEPS = Ziel_Training_TIME_In_Simulation * DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ  # nach 100000 Steps sollten schon mehrbahre Erkenntnisse da sein
DEFAULF_NUMBER_LAST_ACTIONS = 80
DEFAULT_VelocityScale = 0.5

#########REWARD-SETTINGS##########
# Bei wie viel Prozent der Fläche einen Print ausgeben
DEFAULT_Procent_Step = 0.01  # nur im Debug-Mode aktiv?
DEFAULT_REWARD_FOR_NEW_FIELD = 4
DEFAULT_Punishment_for_Step = -0.5
DEFAULT_Multiplier_Collision_Penalty = 2
DEFAULT_no_collision_reward = 0.75  # nur bei R6 aktiv! Ist das Zuckerbrot für den Abstand zur Wand
DEFAULT_Punishment_for_Walls = 8  # R7 - Negative Reward Map Settings
DEFAULT_Influence_of_Walls = 4  # R7 - Negative Reward Map Settings


##########EXPLORATION/MAZE-SETTINGS######
DEFAULT_explore_Matrix_Size = 5  # 5 bedeutet eine 5x5 Matrix um die Drohne herum (in diesem Bereich kann die Drohne die Felder einsammeln und diese als erkundet markieren)
DEFAULT_List_MazesToUse = (22, 23, 24, 25)  # Mazes 0-26 stehen zur Verfügung
DEFAULT_List_Start_PositionsToUse = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # Startpositionen 0-9  stehen zur Verfügung (10 Startpositionen)
DEFAULT_MaxRoundsOnOneMaze = 6  # nach wie vielen Schritten wird ein neues maze gewählt # NOTE - Verändert nichts
DEFAULT_MaxRoundsSameStartingPositions = 2
DEFAULT_collision_penalty_terminated = -100  # mit -10 Trainiert SAC gut, bleibt aber noch ca. 50 mal an der Wand hängen--
DEFAULT_Terminated_Wall_Distance = 0.1  # DAS IST AUCH DER WERT; ABER DER DIE WANDBESTRAFUNG IM R5/R6 gegeben wird. --> worst case betrachtung; wenn Drohe im 45 Grad winkel auf die Wand schaut muss dieser mit cos(45) verrechnet werden --> Distanz: 0,25 -> Worstcase-Distanz = 0,18 ; 0,3 -> 0,21; 0,35 --> 0,25;; WAR ORIGINAL ALEX BEI =0,15!!


###############################################################################################
########################MODEL-&Umbgebungs-SETTINGS##############################################
###############################################################################################

#####################################MODEL_VERSION###########################
MODEL_VERSION = "M5"

"""MODEL_Versionen: 
- M1:   PPO
- M2:   DQN_CNNPolicy_StandardFeatureExtractor
- M3:   DQN_MLPPolicy
- M4:   DQN_CNNPolicy_CustomFeatureExtractor # NO SUPPORT 
- M5:   DQN_NN_MultiInputPolicy mit fullyConnectLayer
- M6:   SAC
"""


#####################################REWARD_VERSION###########################
REWARD_VERSION = "R6"

"""REWARD_VERSIONen: siehe BaseAviary_MAZE_TRAINING.py für Details
- R1:   Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
- R2:   Zusätzlich Bestrafung für zu nah an der Wand
- R3:   Collision zieht je nach Wert auf Heatmap diesen von der Reward ab (7 etwa Wand, 2 nahe Wand, 0.)
- R4:   Collision zieht je nach Wert auf Heatmap diesen von der Reward ab (7 etwa Wand, 2 nahe Wand, 0.) und Abzug für jeden Step
- R5:   R4 mit dem Anpassung der Bestrafung für die Wand: ist optimiert für T2 optimiert.  Zu nah an der Wand wird nur einmal Bestraft, nämlich dann, wenn Terminated wird (wegen zu nah an der Wand) (Wert = DEFAULT_collision_penalty_terminated)
- R6:   R5 mit dem Zusatz, dass wenn die Drohne nicht zu nah an der Wand ist, gibt es einen definierten Bonus (Anstatt nur Peitsche jetzt Zuckerbrot und Peitsche)
- R7:   Statt Heatmap nun Bestrafungsmap (lineare Bestrafung - Abstand zur Wand), Truncated bei Wandberührung, Abzug für jeden Step
"""


#####################################OBSERVATION_TYPE###########################
OBSERVATION_TYPE = "O7"

"""ObservationType:
        - O1: X, Y, Yaw, Raycast readings (nur PPO)                             # no longer supported (Februar 2025) 
        - O2: 5 Kanäliges Bild CNN                                      # no longer supported (April 2025)
        - O3: 5 Kanäliges Bild CNN mit zweimal last Clipped Actions     # no longer supported (April 2025)
- O4: 1 Kanal: Normalisierte SLAM Map (Occupancy Map)
- O5: XY Position, Yaw (sin,cos), Raycast readings, last clipped actions 
- O6: cropped Slam-image, XY Position, Yaw (sin,cos), last actions (n Stück)
- O7: cropped Slam-image, XY Position, Yaw (sin,cos), last actions (n Stück), raycasts
- O8: X-Pos,Y-Pos, 4-raycast_readings, 4-interest_values, n-last_clipped_actions
- O9: cropped Slam-image, x-Pos, y-Pos, 4-racast_readings, 4-interest_values, n-last_clipped_actions
 !!!Bei neuer Oberservation Type mit SLAM dies in den IF-Bedingungen (BaseAviary_MAZE_TRAINING.py) erweitern!!!
"""

#####################################ACTION_TYPE###########################
ACTION_TYPE = "A2"

"""ActionType:'
- A1: Vier Richtungen und zwei Drehungen, diskret
- A2: Vier Richtungen, diskret
- A3: Vier Richtungen, kontinuierlich # für SAC
"""

#####################################TRUNCATED_TYPE###########################
TRUNCATED_TYPE = "TR1"

""" Truncated_type:
- TR1: Zeit abgelaufen
"""


#####################################TERMINATED_TYPE###########################
TERMINATED_TYPE = "T1"

""" Terminated_type:
- T1: 80% der Fläche erkundet
- T2: 80% der Fläche erkundet oder Crash (Abstandswert geringer als X)
"""


##############################################################################
################# Backgrund-Maze-Preparation ###############################
##############################################################################

DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_OBS = ObservationType("kin")  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType("vel")  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_DRONE_MODEL = DroneModel("cf2x")


if Training_Mode == "Training":
    Default_Train = True
    Default_Test = False
elif Training_Mode == "Test":
    Default_Train = False
    Default_Test = True

if GUI_Mode == "Train":
    DEFAULT_GUI_TRAIN = True
    DEFAULT_GUI_TEST = False
elif GUI_Mode == "Test":
    DEFAULT_GUI_TRAIN = False
    DEFAULT_GUI_TEST = True
elif GUI_Mode == "NoGUI":
    DEFAULT_GUI_TRAIN = False
    DEFAULT_GUI_TEST = False

# file_path = "gym_pybullet_drones/examples/MAZE_TRAINING/Maze_init_target.yaml"
file_path = os.path.join(os.path.dirname(__file__), "Maze_init_target.yaml")


def loadyaml_(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


Target_Start_Values = loadyaml_(file_path)

# Initialisieren Sie das Dictionary, um die Werte zu speichern
INIT_XYZS = {}
DEFAULT_TARGET_POSITION = {}
DEFAULT_ALTITUDE = 0.5
# Iterieren Sie über die Maps und speichern Sie die Werte im Dictionary
for map_name, map_values in Target_Start_Values.items():
    INIT_XYZS[map_name] = (map_values["initial_xyzs"],)
    DEFAULT_TARGET_POSITION[map_name] = (map_values["target_position"],)
    # INIT_RPYS = map_values['initial_rpys']


# INIT_RPYS = {}
if ACTION_TYPE == "A1":
    INIT_RPYS = np.array(
        [
            [0, 0, np.random.uniform(0, 2 * np.pi)],
        ]
    )
else:
    INIT_RPYS = np.array(
        [
            [
                0,
                0,
                np.pi / 2,
            ],  # nicht 0 Grad bei z, da dann die Achsen wie das Haupt-Koordinatensystem liegt. Vorwärts wäre dann nach rechts im Bild, da die x-Achse von oben gesehen nach Rechts zeigt
        ]
    )

############################ Hyperparameter-Set speichern ###########################
HYPERPARAMETER_SET_PATH = os.path.join(os.path.dirname(__file__), "hyperparameter_sets.csv")

if DEFAULT_SAVE_HYPERPARAMETER_SET:
    # Define the headers for the CSV
    headers = [
        "set_id",
        "description",
        "Ziel_Training_TIME_In_Simulation",
        "DEFAULT_EVAL_FREQ",
        "DEFAULT_EVAL_EPISODES",
        "DEFAULT_TRAIN_TIMESTEPS",
        "DEFAULT_TARGET_REWARD",
        "DEFAULT_NUMBER_LAST_ACTIONS",
        "DEFAULT_PYB_FREQ",
        "DEFAULT_CTRL_FREQ",
        "DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ",
        "DEFAULT_EPISODE_LEN_SEC",
        "DEFAULT_PUSHBACK_ACTIVE",
        "NumberOfInterationsTillNextCheckpoint",
        "DEFAULF_NUMBER_LAST_ACTIONS",
        "DEFAULT_VelocityScale",
        "DEFAULT_Procent_Step",
        "DEFAULT_REWARD_FOR_NEW_FIELD",
        "DEFAULT_Punishment_for_Step",
        "DEFAULT_Multiplier_Collision_Penalty",
        "DEFAULT_Punishment_for_Walls",
        "DEFAULT_Influence_of_Walls",
        "DEFAULT_no_collision_reward",
        "DEFAULT_explore_Matrix_Size",
        "DEFAULT_List_MazesToUse",
        "DEFAULT_List_Start_PositionsToUse",
        "DEFAULT_MaxRoundsOnOneMaze",
        "DEFAULT_MaxRoundsSameStartingPositions",
        "DEFAULT_collision_penalty_terminated",
        "DEFAULT_Terminated_Wall_Distance",
        "date_created",
        "comment",
    ]

    # Check if the hyperparameter CSV file exists in current folder

    if not os.path.exists(HYPERPARAMETER_SET_PATH):
        # Create the file if it doesn't exist
        with open(HYPERPARAMETER_SET_PATH, "w") as f:
            pass

        # Create the CSV file with headers
        with open(HYPERPARAMETER_SET_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        print(f"Created hyperparameter CSV at {HYPERPARAMETER_SET_PATH}")
    else:
        print(f"Hyperparameter CSV already exists at {HYPERPARAMETER_SET_PATH}")

    # Get current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create a unique set ID
    set_id = f"SET_{timestamp}"

    # Get current parameters
    current_params = {
        "set_id": set_id,
        "description": HyperparameterSetDescription,
        "Ziel_Training_TIME_In_Simulation": Ziel_Training_TIME_In_Simulation,
        "DEFAULT_EVAL_FREQ": DEFAULT_EVAL_FREQ,
        "DEFAULT_EVAL_EPISODES": DEFAULT_EVAL_EPISODES,
        "DEFAULT_TRAIN_TIMESTEPS": DEFAULT_TRAIN_TIMESTEPS,
        "DEFAULT_TARGET_REWARD": DEFAULT_TARGET_REWARD,
        "DEFAULT_NUMBER_LAST_ACTIONS": DEFAULT_NUMBER_LAST_ACTIONS,
        "DEFAULT_PYB_FREQ": DEFAULT_PYB_FREQ,
        "DEFAULT_CTRL_FREQ": DEFAULT_CTRL_FREQ,
        "DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ": DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,
        "DEFAULT_EPISODE_LEN_SEC": DEFAULT_EPISODE_LEN_SEC,
        "DEFAULT_PUSHBACK_ACTIVE": DEFAULT_PUSHBACK_ACTIVE,
        "NumberOfInterationsTillNextCheckpoint": NumberOfInterationsTillNextCheckpoint,
        "DEFAULF_NUMBER_LAST_ACTIONS": DEFAULF_NUMBER_LAST_ACTIONS,
        "DEFAULT_VelocityScale": DEFAULT_VelocityScale,
        "DEFAULT_Procent_Step": DEFAULT_Procent_Step,
        "DEFAULT_REWARD_FOR_NEW_FIELD": DEFAULT_REWARD_FOR_NEW_FIELD,
        "DEFAULT_Punishment_for_Step": DEFAULT_Punishment_for_Step,
        "DEFAULT_Multiplier_Collision_Penalty": DEFAULT_Multiplier_Collision_Penalty,
        "DEFAULT_Punishment_for_Walls": DEFAULT_Punishment_for_Walls,
        "DEFAULT_Influence_of_Walls": DEFAULT_Influence_of_Walls,
        "DEFAULT_no_collision_reward": DEFAULT_no_collision_reward,
        "DEFAULT_explore_Matrix_Size": DEFAULT_explore_Matrix_Size,
        "DEFAULT_List_MazesToUse": str(DEFAULT_List_MazesToUse),  # Convert tuple to string
        "DEFAULT_List_Start_PositionsToUse": str(DEFAULT_List_Start_PositionsToUse),  # Convert tuple to string
        "DEFAULT_MaxRoundsOnOneMaze": DEFAULT_MaxRoundsOnOneMaze,
        "DEFAULT_MaxRoundsSameStartingPositions": DEFAULT_MaxRoundsSameStartingPositions,
        "DEFAULT_collision_penalty_terminated": DEFAULT_collision_penalty_terminated,
        "DEFAULT_Terminated_Wall_Distance": DEFAULT_Terminated_Wall_Distance,
        "date_created": timestamp,
        "comment": "feel free to add any comments here manually",
    }

    # Read the existing CSV to get the headers and check if all parameters are present
    try:
        with open(HYPERPARAMETER_SET_PATH, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

            # Check if all current parameters exist in headers
            missing_params = []
            for param in current_params.keys():
                if param not in headers:
                    missing_params.append(param)

            if missing_params:
                raise ValueError(f"The following parameters are missing in the CSV headers: {missing_params}\nPlease add these parameters to the CSV file before saving.")

    except Exception as e:
        print(f"Error reading CSV file or checking parameters: {str(e)}")
        raise

    # Append the new parameter set to the CSV
    with open(HYPERPARAMETER_SET_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Create a row with values in the same order as headers
        row = [current_params.get(header, "") for header in headers]
        writer.writerow(row)

    print(f"\nSaved current parameters as set {set_id} with description: {HyperparameterSetDescription}\n")


#######################################HYPERPARAMETER SET LOAD####################
def load_hyperparameter_set(csv_file_path, set_id):
    """
    Load a specific hyperparameter set from the CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing hyperparameter sets
        set_id (str): ID of the hyperparameter set to load (e.g., 'SET_20250409-233159')

    Returns:
        dict: Dictionary containing the hyperparameters
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path} while trying to load hyperparameter set {set_id}")
        return {}

    try:
        with open(csv_file_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            # Find the row with the matching set_id
            for row in reader:
                if row["set_id"] == set_id:
                    # Convert string values to appropriate types
                    params = {}
                    for key, value in row.items():
                        # Skip non-parameter columns
                        if key in ["set_id", "description", "date_created", "comment"]:
                            continue

                        # Convert string values to appropriate types
                        try:
                            if value == "True":
                                params[key] = True
                            elif value == "False":
                                params[key] = False
                            elif value.startswith("(") and value.endswith(")"):
                                # Convert tuple string to tuple of integers
                                num_str = value.strip("()").strip()
                                if num_str:
                                    nums = [int(x.strip()) for x in num_str.split(",")]
                                    params[key] = tuple(nums)
                                else:
                                    params[key] = tuple()
                            else:
                                # Try converting to float first
                                try:
                                    params[key] = float(value)
                                    # If it's a whole number, convert to int
                                    if params[key].is_integer():
                                        params[key] = int(params[key])
                                except ValueError:
                                    params[key] = value
                        except Exception as e:
                            print(f"Warning: Could not convert {key}={value}, using as string. Error: {str(e)}")
                            params[key] = value

                    print(f"Successfully loaded hyperparameter set {set_id}")
                    return params

            print(f"Error: Hyperparameter set {set_id} not found in CSV file")
            return {}

    except Exception as e:
        print(f"Error loading hyperparameter set: {str(e)}")
        return {}


if DEFAULT_USE_SAVED_HYPERPARAMETER_SET:
    hyperparameter_set = load_hyperparameter_set(HYPERPARAMETER_SET_PATH, HyperparameterSetIDtoLoad)
    for key, value in hyperparameter_set.items():
        if key in globals():
            globals()[key] = value


######################################CSV ERSTELLEN######################################


# Der Dateipfad zur CSV-Datei
timestamp = time.strftime("%Y%m%d-%H%M%S")
# check if folder gym_pybullet_drones/Auswertungen_der_Modelle/ exists and if not create it
if not os.path.exists("gym_pybullet_drones/Auswertungen_der_Modelle/"):
    os.makedirs("gym_pybullet_drones/Auswertungen_der_Modelle/")
if Training_Mode == "Training":
    Auswertungs_CSV_Datei = (
        f"gym_pybullet_drones/Auswertungen_der_Modelle/TRAINING_{MODEL_VERSION}_{REWARD_VERSION}_{OBSERVATION_TYPE}_{ACTION_TYPE}_{TRUNCATED_TYPE}_{TERMINATED_TYPE}_{timestamp}.csv"
    )
elif Training_Mode == "Test":
    Auswertungs_CSV_Datei = f"gym_pybullet_drones/Auswertungen_der_Modelle/TEST_{MODEL_VERSION}_{REWARD_VERSION}_{OBSERVATION_TYPE}_{ACTION_TYPE}_{TRUNCATED_TYPE}_{TERMINATED_TYPE}_{timestamp}.csv"
# Funktion, um eine CSV zu erstellen (beim ersten Aufruf) oder zu erweitern

# Prüfen, ob die Datei existiert
datei_existiert = os.path.exists(Auswertungs_CSV_Datei)

header_params = [
    "DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ",
    "DEFAULT_EPISODE_LEN_SEC",
    "DEFAULT_REWARD_FOR_NEW_FIELD",
    "DEFAULT_Punishment_for_Step",
    "DEFAULT_EVAL_FREQ",
    "DEFAULT_EVAL_EPISODES",
    "DEFAULT_TRAIN_TIMESTEPS",
    "DEFAULT_TARGET_REWARD",
    "DEFAULT_NUMBER_LAST_ACTIONS",
    "DEFAULT_PYB_FREQ",
    "DEFAULT_CTRL_FREQ",
    "DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ",
    "DEFAULT_EPISODE_LEN_SEC",
    "DEFAULT_Multiplier_Collision_Penalty",
    "DEFAULT_VelocityScale",
    "DEFAULT_explore_Matrix_Size",
    "DEFAULT_collision_penalty_terminated",
    "DEFAULT_Terminated_Wall_Distance",
    "DEFAULT_no_collision_reward",
    "DEFAULT_Punishment_for_Walls",
    "DEFAULT_Influence_of_Walls",
    "DEFAULT_USE_PRETRAINED_MODEL",
    "DEFAULT_PRETRAINED_MODEL_PATH",
    "DEFAULT_List_MazesToUse",
    "DEFAULT_List_Start_PositionsToUse",
    "DEFAULT_MaxRoundsOnOneMaze",
    "DEFAULT_MaxRoundsSameStartingPositions",
]

# Header für die dynamischen Daten (Trainingsergebnisse)
header_training = [
    "Runde",
    "Terminated",
    "Truncated",
    "Map-Abgedeckt",
    "Wand berührungen",
    "Summe Reward",
    "Flugzeit der Runde",
    "Maze_number",
    "Start-Position",
    "Uhrzeit Welt",
]

# Beispielwerte für die Parameter (statisch)
parameter_daten = [
    DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,  # DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ
    DEFAULT_EPISODE_LEN_SEC,  # DEFAULT_EPISODE_LEN_SEC
    DEFAULT_REWARD_FOR_NEW_FIELD,  # Bonus_new_Field
    DEFAULT_Punishment_for_Step,  # Punish_Step_Counter
    DEFAULT_EVAL_FREQ,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_TRAIN_TIMESTEPS,
    DEFAULT_TARGET_REWARD,
    DEFAULT_NUMBER_LAST_ACTIONS,
    DEFAULT_PYB_FREQ,
    DEFAULT_CTRL_FREQ,
    DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,
    DEFAULT_EPISODE_LEN_SEC,
    DEFAULT_Multiplier_Collision_Penalty,
    DEFAULT_VelocityScale,
    DEFAULT_explore_Matrix_Size,
    DEFAULT_collision_penalty_terminated,
    DEFAULT_Terminated_Wall_Distance,
    DEFAULT_no_collision_reward,
    DEFAULT_USE_PRETRAINED_MODEL,
    DEFAULT_PRETRAINED_MODEL_PATH,
    DEFAULT_NUMBER_LAST_ACTIONS,
    DEFAULT_Punishment_for_Walls,
    DEFAULT_Influence_of_Walls,
    DEFAULT_List_MazesToUse,
    DEFAULT_List_Start_PositionsToUse,
    DEFAULT_MaxRoundsOnOneMaze,
    DEFAULT_MaxRoundsSameStartingPositions,
]

# Öffnen oder Erstellen der CSV-Datei
with open(Auswertungs_CSV_Datei, mode="a", newline="") as file:
    writer = csv.writer(file)

if not datei_existiert:
    with open(Auswertungs_CSV_Datei, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Wenn die Datei nicht existiert oder wir neue Werte beim ersten Aufruf schreiben, dann Header hinzufügen
        writer.writerow(header_params)  # Die statischen Parameter (Header)
        writer.writerow(parameter_daten)  # Die zugehörigen Parameter-Werte
        writer.writerow([])  # Leere Zeile zur Trennung der Tabellen
        writer.writerow(header_training)  # Header für die dynamischen Trainingsdaten


#####################################################################################
##############################RUN-FUNCTION###########################################
#####################################################################################


def run(
    multiagent=DEFAULT_MA,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui_Train=DEFAULT_GUI_TRAIN,
    gui_Test=DEFAULT_GUI_TEST,
    plot=True,
    colab=DEFAULT_COLAB,
    record_video=DEFAULT_RECORD_VIDEO,
    local=True,
    pyb_freq=DEFAULT_PYB_FREQ,
    ctrl_freq=DEFAULT_CTRL_FREQ,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    reward_and_action_change_freq=DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,
    drone_model=DEFAULT_DRONE_MODEL,
    advanced_status_plot=DEFAULT_ADVANCED_STATUS_PLOT,
    target_position=DEFAULT_TARGET_POSITION,
    EPISODE_LEN_SEC=DEFAULT_EPISODE_LEN_SEC,
    dash_active=DEFAULT_DASH_ACTIVE,
    MODEL_Version=MODEL_VERSION,
    reward_version=REWARD_VERSION,
    ObservationType=OBSERVATION_TYPE,
    Truncated_Type=TRUNCATED_TYPE,
    Terminated_Type=TERMINATED_TYPE,
    Action_Type=ACTION_TYPE,
    Pushback_active=DEFAULT_PUSHBACK_ACTIVE,
    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
    DEFAULT_VelocityScale=DEFAULT_VelocityScale,
    Procent_Step=DEFAULT_Procent_Step,
    TRAIN=Default_Train,
    TEST=Default_Test,
    number_last_actions=DEFAULT_NUMBER_LAST_ACTIONS,
    Reward_for_new_field=DEFAULT_REWARD_FOR_NEW_FIELD,
    Punishment_for_Step=DEFAULT_Punishment_for_Step,
    Auswertungs_CSV_Datei=Auswertungs_CSV_Datei,
    Explore_Matrix_Size=DEFAULT_explore_Matrix_Size,
    List_MazesToUse=DEFAULT_List_MazesToUse,
    List_Start_PositionsToUse=DEFAULT_List_Start_PositionsToUse,
    MaxRoundsOnOneMaze=DEFAULT_MaxRoundsOnOneMaze,
    MaxRoundsSameStartingPositions=DEFAULT_MaxRoundsSameStartingPositions,
    collision_penalty_terminated=DEFAULT_collision_penalty_terminated,
    Terminated_Wall_Distance=DEFAULT_Terminated_Wall_Distance,
    no_collision_reward=DEFAULT_no_collision_reward,
    punishment_for_walls=DEFAULT_Punishment_for_Walls,
    influence_of_walls=DEFAULT_Influence_of_Walls,
):
    if TRAIN:
        filename = os.path.join(output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename + "/")

            # ANCHOR - learn_MAZE_TRAINING ENVS

            train_env = make_vec_env(
                BaseRLAviary_MAZE_TRAINING,
                env_kwargs=dict(
                    drone_model=drone_model,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    physics=Physics.PYB,
                    gui=gui_Train,
                    user_debug_gui=user_debug_gui,
                    pyb_freq=pyb_freq,
                    ctrl_freq=ctrl_freq,  # Ansatz: von 60 auf 10 reduzieren, damit die gewählte Action länger wirkt
                    reward_and_action_change_freq=reward_and_action_change_freq,  # Ansatz: neu hinzugefügt, da die Step-Funktion vorher mit der ctrl_freq aufgerufen wurde, Problem war dann, dass bei hoher Frequenz die Raycasts keine Änderung hatten, dafür die Drohne aber sauber geflogen ist (60). Wenn der Wert niedriger war, hat es mit den Geschwindigkeiten und Actions besser gepasst, dafür ist die Drohne nicht sauber geflogen, weil die Ctrl-Frequenz für das erreichen der gewählten Action zu niedrig war (10/20).
                    act=ActionType.VEL,
                    target_position=target_position,
                    dash_active=dash_active,
                    EPISODE_LEN_SEC=EPISODE_LEN_SEC,
                    REWARD_VERSION=reward_version,
                    ACTION_TYPE=Action_Type,
                    OBSERVATION_TYPE=ObservationType,
                    Truncated_Type=Truncated_Type,
                    Terminated_Type=Terminated_Type,
                    Pushback_active=Pushback_active,
                    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                    VelocityScale=DEFAULT_VelocityScale,
                    Procent_Step=Procent_Step,
                    number_last_actions=number_last_actions,
                    Punishment_for_Step=Punishment_for_Step,
                    Reward_for_new_field=Reward_for_new_field,
                    csv_file_path=Auswertungs_CSV_Datei,  # Pfad zur CSV-Datei
                    Explore_Matrix_Size=Explore_Matrix_Size,
                    List_MazesToUse=List_MazesToUse,
                    List_Start_PositionsToUse=List_Start_PositionsToUse,
                    MaxRoundsOnOneMaze=MaxRoundsOnOneMaze,
                    MaxRoundsSameStartingPositions=MaxRoundsSameStartingPositions,
                    collision_penalty_terminated=collision_penalty_terminated,
                    Terminated_Wall_Distance=Terminated_Wall_Distance,
                    no_collision_reward=no_collision_reward,
                    punishment_for_walls=punishment_for_walls,
                    influence_of_walls=influence_of_walls,
                ),
                n_envs=1,
                seed=0,
            )

            eval_env = make_vec_env(
                BaseRLAviary_MAZE_TRAINING,
                env_kwargs=dict(
                    drone_model=drone_model,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    gui=gui_Test,
                    physics=Physics.PYB,
                    user_debug_gui=user_debug_gui,
                    pyb_freq=pyb_freq,
                    ctrl_freq=ctrl_freq,
                    reward_and_action_change_freq=reward_and_action_change_freq,
                    act=ActionType.VEL,
                    target_position=target_position,
                    dash_active=dash_active,
                    EPISODE_LEN_SEC=EPISODE_LEN_SEC,
                    REWARD_VERSION=reward_version,
                    ACTION_TYPE=Action_Type,
                    OBSERVATION_TYPE=ObservationType,
                    Truncated_Type=Truncated_Type,
                    Terminated_Type=Terminated_Type,
                    Pushback_active=Pushback_active,
                    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                    VelocityScale=DEFAULT_VelocityScale,
                    Procent_Step=Procent_Step,
                    number_last_actions=number_last_actions,
                    Punishment_for_Step=Punishment_for_Step,
                    Reward_for_new_field=Reward_for_new_field,
                    csv_file_path=Auswertungs_CSV_Datei,  # Pfad zur CSV-Datei
                    Explore_Matrix_Size=Explore_Matrix_Size,
                    List_MazesToUse=List_MazesToUse,
                    List_Start_PositionsToUse=List_Start_PositionsToUse,
                    MaxRoundsOnOneMaze=MaxRoundsOnOneMaze,
                    MaxRoundsSameStartingPositions=MaxRoundsSameStartingPositions,
                    collision_penalty_terminated=collision_penalty_terminated,
                    Terminated_Wall_Distance=Terminated_Wall_Distance,
                    no_collision_reward=no_collision_reward,
                    punishment_for_walls=punishment_for_walls,
                    influence_of_walls=influence_of_walls,
                ),
                n_envs=1,
                seed=0,
            )

        #### Check the environment's spaces ########################
        print("[INFO] Action space:", train_env.action_space)
        print("[INFO] Observation space:", train_env.observation_space)
        # NOTE - FIX THE ACTION SPACE
        #### Load existing model or create new one ###################
        match MODEL_Version:
            case "M1":  # M1: PPO
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = PPO.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = PPO(
                        "MlpPolicy",
                        train_env,
                        verbose=1,
                        # Learning-Rate 0,0002 zu gering -> auf 0.0004 erhöht -> auf 0.0005 erhöht --> auf 0.0004 reduziert, da die Policy zu stark angepasst wurde, obwohl es schon 5s am Ziel war..
                    )

            case "M2":  # M2: DQN_CNNPolicy_StandardFeatureExtractor
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = DQN(
                        "CnnPolicy",
                        train_env,
                        # learning_rate=0.0004, #nicht verwendet --> erst mal standard fürs Training
                        device="cuda:0",
                        verbose=1,
                        # buffer_size=500000,
                    )

            case "M3":  # M3: DQN_MLPPolicy
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = DQN(
                        "MlpPolicy",
                        train_env,
                        device="cuda:0",
                        # learning_rate=0.0004,
                        # policy_kwargs=dict(net_arch=[128, 64, 32]),
                        # learning_rate=0.004,
                        verbose=1,
                        seed=42,
                        # buffer_size=5000,
                        # gamma=0.8,
                    )

            case "M4":  # M4: DQN_CNNPolicy_CustomFeatureExtractor
                # ANCHOR - CNN-DQN
                # Setze die policy_kwargs, um deinen Custom Feature Extractor zu nutzen:
                policy_kwargs = dict(
                    features_extractor_class=CustomCNNFeatureExtractor(),
                    features_extractor_kwargs=dict(features_dim=4),
                )
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = DQN(
                        "CnnPolicy",
                        train_env,
                        policy_kwargs=policy_kwargs,
                        device="cuda:0",
                        # learning_rate=0.0004,
                        # learning_rate=0.001,
                        verbose=1,
                        seed=42,
                        # buffer_size=5000,
                    )  # Reduced from 1,000,000 to 10,000 nochmal reduziert auf 5000 da zu wenig speicher
            # NOTE - OHNE ZUFALLSWERTE AM ANFANG

            case "M5":  # M5: DQN_NN_MulitInputPolicy
                # ANCHOR - NN-DQN-MI
                # Setze die policy_kwargs, um deinen Custom Feature Extractor zu nutzen:
                # policy_kwargs = dict(features_extractor_class=CustomNNFeatureExtractor, features_extractor_kwargs=dict(features_dim=4))
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = DQN(
                        "MultiInputPolicy",
                        train_env,
                        device="cuda:0",
                        # policy_kwargs=dict(net_arch=[128, 64, 32]),
                        # learning_rate=0.004,
                        verbose=1,
                        # batch_size=32,
                        seed=42,
                        buffer_size=500000,
                        # gamma=0.8,
                    )  # Reduced from 1,000,000 to 10,000 nochmal reduziert auf 5000 da zu wenig speicher
            case "M6":  # M6: SAC
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(
                        f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}"
                    )
                    model = SAC.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new {MODEL_VERSION} with {REWARD_VERSION}, {OBSERVATION_TYPE}, {TRUNCATED_TYPE}, {TERMINATED_TYPE}, {ACTION_TYPE}")
                    model = SAC(
                        "MlpPolicy",
                        train_env,
                        verbose=1,
                        device="cuda:0",
                        seed=42,
                    )
            case _:
                raise ValueError(f"Invalid model version: {MODEL_Version}")

        # ###### Fix Warnuning on using M5:
        # /home/florian/anaconda3/envs/drones/lib/python3.10/site-packages/stable_baselines3/common/callbacks.py:414: UserWarning: Training and eval env are not of the same type<stable_baselines3.common.vec_env.vec_transpose.VecTransposeImage object at 0x7cbda05a2a70> != <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7cbda05a3640> warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}") ###########

        wrapped_train_env = model.get_env()
        # print(f"Type of wrapped training environment: {type(wrapped_train_env)}")
        # Apply the same wrapper to eval_env
        if isinstance(wrapped_train_env, VecTransposeImage):
            eval_env = VecTransposeImage(eval_env)

        ## Schreiben der CSV für die Auswertung unserer Ergebnisse

        #### Target cumulative rewards (problem-dependent) ##########
        target_reward = DEFAULT_TARGET_REWARD
        print(target_reward)
        # The StopTrainingOnRewardThreshold callback is used to stop the training once a certain reward threshold is reached.
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
        # The EvalCallback is used to evaluate the agent periodically during training.
        # eval_env: The environment used for evaluation.
        # callback_on_new_best: Callback to trigger when a new best model is found.
        # verbose=1: Info messages will be printed during evaluation.
        # best_model_save_path: Path to save the best model.
        # log_path: Path to save evaluation logs.
        # eval_freq: Frequency of evaluations (every 1000 steps in this case).
        # deterministic=True: Use deterministic actions during evaluation.
        # render=False: Do not render the environment during evaluation.
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            verbose=1,
            best_model_save_path=filename + "/",
            log_path=filename + "/",
            eval_freq=DEFAULT_EVAL_FREQ,  # alle 10000 Schritte wird die Evaluation durchgeführt (mit Frequenz reward_and_action_change_freq)
            deterministic=True,
            render=False,  # nicht auf True setzbar, da dem RL-Environment keine render_mode="human"übergeben werden kann
            n_eval_episodes=1,
        )  # neu eingefügt, dass es schneller durch ist mit der Visu
        # The model.learn function is used to train the model.
        # total_timesteps: The total number of timesteps to train for. It is set to 1e7 (10 million) if local is True, otherwise 1e2 (100) for shorter training in GitHub Actions pytest.
        # callback: The callback to use during training, in this case, eval_callback.
        # log_interval: The number of timesteps between logging events.
        # In your code, the model will train for a specified number of timesteps, using the eval_callback for periodic evaluation, and log information every 100 timesteps.

        # # Definiere den Speicherpfad und die Häufigkeit der Checkpoints (z.B. alle 10.000 Schritte)
        checkpoint_callback = CheckpointCallback(
            save_freq=NumberOfInterationsTillNextCheckpoint,  # Speichert alle 10.000 Schritte
            save_path=filename + "/",  # Speicherpfad für die Modelle
            name_prefix=filename + "/",  # Präfix für die Modell-Dateien
            save_replay_buffer=True,  # Speichert auch den Replay Buffer
            save_vecnormalize=True,  # Falls VecNormalize genutzt wird, wird es mitgespeichert
        )

        # # Kombiniere beide Callbacks mit CallbackList
        callback_list = CallbackList([checkpoint_callback, eval_callback])

        start_time = time.time()  # Startzeit erfassen
        model.learn(
            total_timesteps=DEFAULT_TRAIN_TIMESTEPS,
            callback=callback_list,
            log_interval=1000,
            progress_bar=True,
        )  # shorter training in GitHub Actions pytest
        end_time = time.time()  # Endzeit erfassen
        elapsed_time = end_time - start_time  # Dauer berechnen
        elapsed_time_hours = elapsed_time / 3600  # Zeit in Stunden umrechnen
        print(f"Training abgeschlossen. Dauer: {elapsed_time_hours:.2f} Stunden")

        datei_existiert = os.path.exists(Auswertungs_CSV_Datei)

        # Öffne die CSV-Datei zum Anhängen oder Erstellen
        with open(Auswertungs_CSV_Datei, mode="a", newline="") as file:
            writer = csv.writer(file)

            if datei_existiert:
                # Schreibe die Trainingsdaten in die zweite Tabelle
                writer.writerow([])
                writer.writerow(["Trainingszeit (Stunden)"])
                writer.writerow([f"{elapsed_time_hours:.2f}"])

        #### Save the model ########################################

        # Überprüfen, ob das Verzeichnis existiert
        if os.path.exists(filename):
            # Extrahiere den letzten Ordnernamen
            last_folder = os.path.basename(filename)

            # Erstelle den vollständigen Pfad zum Ordner, dessen Inhalt du löschen möchtest
            target_folder = os.path.join("results", last_folder)

            # Überprüfen, ob das Zielverzeichnis existiert
            if os.path.exists(target_folder):
                # Alle Dateien und Ordner im Verzeichnis durchgehen
                for item in os.listdir(target_folder):
                    item_path = os.path.join(target_folder, item)  # Vollständiger Pfad zu den Dateien/Ordnern
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # Wenn es ein Verzeichnis ist, entferne es rekursiv
                        else:
                            os.remove(item_path)  # Wenn es eine Datei ist, entferne sie
                    except Exception as e:
                        print(f"Fehler beim Löschen von {item_path}: {e}")
            else:
                print(f"Verzeichnis {target_folder} existiert nicht.")
        else:
            print(f"Verzeichnis {filename} existiert nicht.")

        model.save(filename + "/final_model.zip")
        print(filename)

        #### Print training progression ############################
        with np.load(filename + "/evaluations.npz") as data:
            for j in range(data["timesteps"].shape[0]):
                print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))

    ############################################################

    # if local:
    #     input("Press Enter to continue...")

    if TEST:
        """MODEL_Versionen:
        - M1:   PPO
        - M2:   DQN_CNNPolicy_StandardFeatureExtractor
        - M3:   DQN_MLPPolicy
        - M4:   DQN_CNNPolicy_CustomFeatureExtractor
        - M5:   DQN_NN_MultiInputPolicy mit fullyConnectLayer
        - M6:   SAC
        """

        # Load the appropriate model type based on MODEL_Version
        match MODEL_Version:
            case "M1":
                model = PPO.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded PPO model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case "M2":
                model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded DQN model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case "M3":
                model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded DQN model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case "M4":
                model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded DQN model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case "M5":
                model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded DQN model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case "M6":
                model = SAC.load(DEFAULT_PRETRAINED_MODEL_PATH)
                print(f"Loaded SAC model from {DEFAULT_PRETRAINED_MODEL_PATH}")
            case _:
                print(f"[ERROR]: Unknown model version in TEST-(PREDICTION)-MODE: {MODEL_Version}")

        #### Show (and record a video of) the model's performance ##

        eval_env = make_vec_env(
            BaseRLAviary_MAZE_TRAINING,
            env_kwargs=dict(
                drone_model=drone_model,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                physics=Physics.PYB,
                gui=gui_Train,
                user_debug_gui=user_debug_gui,
                pyb_freq=pyb_freq,
                ctrl_freq=ctrl_freq,  # Ansatz: von 60 auf 10 reduzieren, damit die gewählte Action länger wirkt
                reward_and_action_change_freq=reward_and_action_change_freq,  # Ansatz: neu hinzugefügt, da die Step-Funktion vorher mit der ctrl_freq aufgerufen wurde, Problem war dann, dass bei hoher Frequenz die Raycasts keine Änderung hatten, dafür die Drohne aber sauber geflogen ist (60). Wenn der Wert niedriger war, hat es mit den Geschwindigkeiten und Actions besser gepasst, dafür ist die Drohne nicht sauber geflogen, weil die Ctrl-Frequenz für das erreichen der gewählten Action zu niedrig war (10/20).
                act=ActionType.VEL,
                target_position=target_position,
                dash_active=dash_active,
                EPISODE_LEN_SEC=EPISODE_LEN_SEC,
                REWARD_VERSION=reward_version,
                ACTION_TYPE=Action_Type,
                OBSERVATION_TYPE=ObservationType,
                Truncated_Type=Truncated_Type,
                Terminated_Type=Terminated_Type,
                Pushback_active=Pushback_active,
                DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                VelocityScale=DEFAULT_VelocityScale,
                Procent_Step=Procent_Step,
                number_last_actions=number_last_actions,
                Punishment_for_Step=Punishment_for_Step,
                Reward_for_new_field=Reward_for_new_field,
                csv_file_path=Auswertungs_CSV_Datei,  # Pfad zur CSV-Datei
                Explore_Matrix_Size=Explore_Matrix_Size,
                List_MazesToUse=List_MazesToUse,
                List_Start_PositionsToUse=List_Start_PositionsToUse,
                MaxRoundsOnOneMaze=MaxRoundsOnOneMaze,
                MaxRoundsSameStartingPositions=MaxRoundsSameStartingPositions,
                collision_penalty_terminated=collision_penalty_terminated,
                Terminated_Wall_Distance=Terminated_Wall_Distance,
                no_collision_reward=no_collision_reward,
            ),
            n_envs=1,
            seed=0,
        )

        wrapped_train_env = model.get_env()
        # print(f"Type of wrapped training environment: {type(wrapped_train_env)}")
        # Apply the same wrapper to eval_env
        if isinstance(wrapped_train_env, VecTransposeImage):
            eval_env = VecTransposeImage(eval_env)

        # logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=DEFAULT_AGENTS if multiagent else 1, output_folder=output_folder, colab=colab)
        # The evaluate_policy function is used to evaluate the performance of the trained model.
        # model: The trained model to be evaluated.
        # test_env_nogui: The environment used for evaluation without GUI.
        # n_eval_episodes=10: The number of episodes to run for evaluation.
        # In your code, the function will evaluate the model over 10 episodes and return the mean and standard deviation of the rewards.
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
        # The reset function is used to reset the environment to its initial state.
        # seed=42: The seed for the random number generator to ensure reproducibility.
        # options={}: Additional options for resetting the environment.
        # In your code, obs will contain the initial observation, and info will contain additional information provided by the environment after resetting.

        obs, info, maze_number = eval_env.reset(seed=42, options={})
        print(
            "PRINT MAZE NUMBER IM LEARN-------------------------------------------",
            maze_number,
            "---------------",
        )
        start = time.time()
        # This code runs a loop to simulate the environment using the trained model and logs the results.
        # Loop: Runs for a specified number of steps.
        # Predict Action: Uses the model to predict the next action.
        # Step: Takes the action in the environment and receives the next observation, reward, and termination status.
        # Log: Logs the state and action if the observation type is KIN.
        # Render: Renders the environment.
        # Sync: Synchronizes the simulation.
        # Reset: Resets the environment if terminated.
        for i in range((eval_env.EPISODE_LEN_SEC + 2) * eval_env.CTRL_FREQ):

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action, maze_number)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print(
                "Obs:",
                obs,
                "\tAction",
                action,
                "\tReward:",
                reward,
                "\tTerminated:",
                terminated,
                "\tTruncated:",
                truncated,
            )
            # if DEFAULT_OBS == ObservationType.KIN:
            #     if not multiagent:
            #         logger.log(drone=0, timestamp=i / test_env.CTRL_FREQ, state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]), control=np.zeros(12))
            #     else:
            #         for d in range(DEFAULT_AGENTS):
            #             logger.log(drone=d, timestamp=i / test_env.CTRL_FREQ, state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]), control=np.zeros(12))
            eval_env.render()
            print(terminated)
            sync(i, start, eval_env.CTRL_TIMESTEP)
            if terminated:
                obs = eval_env.reset(seed=42, options={})
        eval_env.close()

        # if plot and DEFAULT_OBS == ObservationType.KIN:
        #     logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning example script")
    parser.add_argument(
        "--multiagent",
        default=DEFAULT_MA,
        type=str2bool,
        help="Whether to use example LeaderFollower instead of Hover (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--gui_Train",
        default=DEFAULT_GUI_TRAIN,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--gui_Test",
        default=DEFAULT_GUI_TEST,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    parser.add_argument(
        "--pyb_freq",
        default=DEFAULT_PYB_FREQ,
        type=int,
        help="Physics frequency (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--ctrl_freq",
        default=DEFAULT_CTRL_FREQ,
        type=int,
        help="Control frequency (default: 60)",
        metavar="",
    )
    parser.add_argument(
        "--reward_and_action_change_freq",
        default=DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,
        type=int,
        help="Control frequency (default: 60)",
        metavar="",
    )
    parser.add_argument(
        "--drone_model",
        default=DEFAULT_DRONE_MODEL,
        type=str,
        help="Control frequency (default: 60)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="set to True if you want to see the debug GUI, only for showing the frame in training!(default: False)",
        metavar="",
    )
    parser.add_argument(
        "--advanced_status_plot",
        default=DEFAULT_ADVANCED_STATUS_PLOT,
        type=str2bool,
        help="set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)",
        metavar="",
    )
    parser.add_argument(
        "--target_position",
        default=DEFAULT_TARGET_POSITION,
        type=str,
        help="set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)",
        metavar="",
    )
    parser.add_argument(
        "--EPISODE_LEN_SEC",
        default=DEFAULT_EPISODE_LEN_SEC,
        type=int,
        help="set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)",
        metavar="",
    )
    parser.add_argument(
        "--dash_active",
        default=DEFAULT_DASH_ACTIVE,
        type=str2bool,
        help="set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)",
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
