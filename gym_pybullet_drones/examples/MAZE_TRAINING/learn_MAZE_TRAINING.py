"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning library `stable-baselines3`.

"""

import argparse
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np

# Importiere benötigte Module für CNN-DQN
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ACHTUNG: es können nicht beide Werte auf TRUE gesetzt werden (nicht GUI_TRAIN und GUI_TEST zusammen)!
DEFAULT_GUI_TRAIN = True
Default_Train = True
Default_Test = False
Default_Test_filename_test = "Model_test"
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_ADVANCED_STATUS_PLOT = False

DEFAULT_GUI_TEST = False

DEFAULT_USE_PRETRAINED_MODEL = False
# DEFAULT_PRETRAINED_MODEL_PATH = '/home/florian/Documents/gym-pybullet-drones/results/durchgelaufen-DQN/final_model.zip'
# DEFAULT_PRETRAINED_MODEL_PATH = "/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/results/save-03.07.2025_02.23.46/best_model.zip"
DEFAULT_PRETRAINED_MODEL_PATH = "/home/moritz_s/Documents/RKIM_1/F_u_E_Drohnenrennen/GitRepo/gym-pybullet-drones/results/save-03.19.2025_22.33.41/best_model.zip"


DEFAULT_EVAL_FREQ = 5 * 1e4
DEFAULT_EVAL_EPISODES = 1

DEFAULT_TRAIN_TIMESTEPS = 10 * 1e5  # nach 100000 Steps sollten schon mehrbahre Erkenntnisse da sein
DEFAULT_TARGET_REWARD = 99999999999999
DEFAULF_NUMBER_LAST_ACTIONS = 20


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

# INIT_RPYS = {}
INIT_RPYS = np.array(
    [
        [0, 0, np.random.uniform(0, 2 * np.pi)],
    ]
)

# Iterieren Sie über die Maps und speichern Sie die Werte im Dictionary
for map_name, map_values in Target_Start_Values.items():
    INIT_XYZS[map_name] = (map_values["initial_xyzs"],)
    DEFAULT_TARGET_POSITION[map_name] = (map_values["target_position"],)
    # INIT_RPYS = map_values['initial_rpys']


DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_PYB_FREQ = 100
DEFAULT_CTRL_FREQ = 50
DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ = 3  # mit 5hz fliegt die Drohne noch zu oft an die Wand, ohne das das Pushback aktiv werden kann (mit Drehung aktiv) -> 10 HZ
DEFAULT_EPISODE_LEN_SEC = 10 * 60  # 15 * 60
DEFAULT_DRONE_MODEL = DroneModel("cf2x")
DEFAULT_PUSHBACK_ACTIVE = False

DEFAULT_OBS = ObservationType("kin")  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType("vel")  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

DEFAULT_DASH_ACTIVE = False

DEFAULT_Multiplier_Collision_Penalty = 1

DEFAULT_VelocityScale = 1

# Bei wie viel Prozent der Fläche einen Print ausgeben
DEFAULT_Procent_Step = 0.01


"""MODEL_Versionen: 
- M1:   PPO
- M2:   DQN_CNNPolicy_StandardFeatureExtractor
- M3:   DQN_MLPPolicy
- M4:   DQN_CNNPolicy_CustomFeatureExtractor
- M5:   DQN_NN_MIPolicy mit fullyConnectLayer
- SAC:  
"""
MODEL_VERSION = "M5"

"""REWARD_VERSIONen: siehe BaseAviary_MAZE_TRAINING.py für Details
- R1:   Standard-Reward-Version: nur neue entdeckte Felder werden einmalig belohnt
- R2:   Zusätzlich Bestrafung für zu nah an der Wand
- R3:   Collision zieht je nach Wert auf Heatmap diesen von der Reward ab (7 etwa Wand, 2 nahe Wand, 0.)
- R4:   Collision zieht je nach Wert auf Heatmap diesen von der Reward ab (7 etwa Wand, 2 nahe Wand, 0.) und Abzug für jeden Step
"""
REWARD_VERSION = "R4"

"""ObservationType:
- O1: X, Y, Yaw, Raycast readings (nur PPO)
- O2: 5 Kanäliges Bild CNN
- O3: 5 Kanäliges Bild CNN mit zweimal last Clipped Actions
- 04: Kanal 1: Normalisierte SLAM Map (Occupancy Map)
- O5: XYZ Position, Yaw, Raycast readings, 3 last clipped actions 


"""

OBSERVATION_TYPE = "O5"

"""ActionType:'
- A1: Vier Richtungen und zwei Drehungen
- A2: Vier Richtungen
"""

ACTION_TYPE = "A2"

# TODO: Implementierung Actionspace und ObsSpace Auswahl

################################################################################


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
    Action_Type=ACTION_TYPE,
    Pushback_active=DEFAULT_PUSHBACK_ACTIVE,
    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
    DEFAULT_VelocityScale=DEFAULT_VelocityScale,
    Procent_Step=DEFAULT_Procent_Step,
    TRAIN=Default_Train,
    TEST=Default_Test,
    filename_test=Default_Test_filename_test,
    number_last_actions=DEFAULF_NUMBER_LAST_ACTIONS,
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
                    Pushback_active=Pushback_active,
                    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                    VelocityScale=DEFAULT_VelocityScale,
                    Procent_Step=Procent_Step,
                    number_last_actions=number_last_actions,
                ),
                n_envs=1,
                seed=0,
            )
            # if 'train_env' in locals():
            # train_env.close()

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
                    Pushback_active=Pushback_active,
                    DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                    VelocityScale=DEFAULT_VelocityScale,
                    Procent_Step=Procent_Step,
                    number_last_actions=number_last_actions,
                ),
                n_envs=1,
                seed=0,
            )
            # if 'eval_env' in locals():
            # eval_env.close()

        #### Check the environment's spaces ########################
        print("[INFO] Action space:", train_env.action_space)
        print("[INFO] Observation space:", train_env.observation_space)
        # NOTE - FIX THE ACTION SPACE
        #### Load existing model or create new one ###################
        match MODEL_Version:
            case "M1":  # M1: PPO
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_Version} with {REWARD_VERSION}")
                    model = PPO.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new model {MODEL_Version} with {REWARD_VERSION}")
                    model = PPO(
                        "MlpPolicy",
                        train_env,
                        verbose=1,
                        # Learning-Rate 0,0002 zu gering -> auf 0.0004 erhöht -> auf 0.0005 erhöht --> auf 0.0004 reduziert, da die Policy zu stark angepasst wurde, obwohl es schon 5s am Ziel war..
                    )

            case "M2":  # M2: DQN_CNNPolicy_StandardFeatureExtractor
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH} for {MODEL_Version} with {REWARD_VERSION}")
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print(f"[INFO] Creating new model {MODEL_Version} with {REWARD_VERSION}")
                    model = DQN(
                        "CnnPolicy",
                        train_env,
                        # learning_rate=0.0004, #nicht verwendet --> erst mal standard fürs Training
                        device="cuda:0",
                        verbose=1,
                        buffer_size=5000,
                    )

            case "M3":  # M3: DQN_MLPPolicy
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH}")
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print("[INFO] Creating new model with CNN-DQN with custom feature extractor")
                    model = DQN(
                        "MlpPolicy",
                        train_env,
                        device="cuda:0",
                        # learning_rate=0.0004,
                        # policy_kwargs=dict(net_arch=[128, 64, 32]),
                        learning_rate=0.004,
                        verbose=1,
                        seed=42,
                        buffer_size=5000,
                        gamma=0.8,
                    )

            case "M4":  # M4: DQN_CNNPolicy_CustomFeatureExtractor
                # ANCHOR - CNN-DQN
                # Setze die policy_kwargs, um deinen Custom Feature Extractor zu nutzen:
                policy_kwargs = dict(features_extractor_class=CustomCNNFeatureExtractor(), features_extractor_kwargs=dict(features_dim=4))
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH}")
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    print("[INFO] Creating new model with CNN-DQN with custom feature extractor")
                    model = DQN(
                        "CnnPolicy",
                        train_env,
                        policy_kwargs=policy_kwargs,
                        device="cuda:0",
                        # learning_rate=0.0004,
                        learning_rate=0.001,
                        verbose=1,
                        seed=42,
                        buffer_size=5000,
                    )  # Reduced from 1,000,000 to 10,000 nochmal reduziert auf 5000 da zu wenig speicher
            # NOTE - OHNE ZUFALLSWERTE AM ANFANG

            case "M5":  # M5: DQN_NN_MIPolicy
                # ANCHOR - NN-DQN-MI
                # Setze die policy_kwargs, um deinen Custom Feature Extractor zu nutzen:
                # policy_kwargs = dict(features_extractor_class=CustomNNFeatureExtractor, features_extractor_kwargs=dict(features_dim=4))
                if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
                    print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH}")
                    model = DQN.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
                else:
                    model = DQN(
                        "MultiInputPolicy",
                        train_env,
                        device="cuda:0",
                        # learning_rate=0.0004,
                        policy_kwargs=dict(net_arch=[128, 64, 32]),
                        learning_rate=0.004,
                        verbose=1,
                        seed=42,
                        buffer_size=5000,
                        gamma=0.8,
                    )  # Reduced from 1,000,000 to 10,000 nochmal reduziert auf 5000 da zu wenig speicher

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

        # Definiere den Speicherpfad und die Häufigkeit der Checkpoints (z.B. alle 10.000 Schritte)
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,  # Speichert alle 10.000 Schritte
            save_path=filename + "/",  # Speicherpfad für die Modelle
            name_prefix=filename + "/",  # Präfix für die Modell-Dateien
            save_replay_buffer=True,  # Speichert auch den Replay Buffer
            save_vecnormalize=True,  # Falls VecNormalize genutzt wird, wird es mitgespeichert
        )

        # Kombiniere beide Callbacks mit CallbackList
        callback_list = CallbackList([checkpoint_callback, eval_callback])

        model.learn(total_timesteps=DEFAULT_TRAIN_TIMESTEPS, callback=callback_list, log_interval=1000, progress_bar=True)  # shorter training in GitHub Actions pytest

        #### Save the model ########################################
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

        # if os.path.isfile(filename+'/final_model.zip'):
        #     path = filename+'/final_model.zip'
        if os.path.isfile(filename_test + "/best_model.zip"):
            path = filename_test + "/best_model.zip"
        else:
            print("[ERROR]: no model under the specified path", filename_test)
        model = DQN.load(path)

        #### Show (and record a video of) the model's performance ##

        test_env = make_vec_env(
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
                Pushback_active=Pushback_active,
                DEFAULT_Multiplier_Collision_Penalty=DEFAULT_Multiplier_Collision_Penalty,
                VelocityScale=DEFAULT_VelocityScale,
                Procent_Step=Procent_Step,
            ),
            n_envs=1,
            seed=0,
        )

        # logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=DEFAULT_AGENTS if multiagent else 1, output_folder=output_folder, colab=colab)
        # The evaluate_policy function is used to evaluate the performance of the trained model.
        # model: The trained model to be evaluated.
        # test_env_nogui: The environment used for evaluation without GUI.
        # n_eval_episodes=10: The number of episodes to run for evaluation.
        # In your code, the function will evaluate the model over 10 episodes and return the mean and standard deviation of the rewards.
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
        print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
        # The reset function is used to reset the environment to its initial state.
        # seed=42: The seed for the random number generator to ensure reproducibility.
        # options={}: Additional options for resetting the environment.
        # In your code, obs will contain the initial observation, and info will contain additional information provided by the environment after resetting.

        obs, info, maze_number = test_env.reset(seed=42, options={})
        print("PRINT MAZE NUMBER IM LEARN-------------------------------------------", maze_number, "---------------")
        start = time.time()
        # This code runs a loop to simulate the environment using the trained model and logs the results.
        # Loop: Runs for a specified number of steps.
        # Predict Action: Uses the model to predict the next action.
        # Step: Takes the action in the environment and receives the next observation, reward, and termination status.
        # Log: Logs the state and action if the observation type is KIN.
        # Render: Renders the environment.
        # Sync: Synchronizes the simulation.
        # Reset: Resets the environment if terminated.
        for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action, maze_number)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
            # if DEFAULT_OBS == ObservationType.KIN:
            #     if not multiagent:
            #         logger.log(drone=0, timestamp=i / test_env.CTRL_FREQ, state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]), control=np.zeros(12))
            #     else:
            #         for d in range(DEFAULT_AGENTS):
            #             logger.log(drone=d, timestamp=i / test_env.CTRL_FREQ, state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]), control=np.zeros(12))
            test_env.render()
            print(terminated)
            sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated:
                obs = test_env.reset(seed=42, options={})
        test_env.close()

        # if plot and DEFAULT_OBS == ObservationType.KIN:
        #     logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning example script")
    parser.add_argument("--multiagent", default=DEFAULT_MA, type=str2bool, help="Whether to use example LeaderFollower instead of Hover (default: False)", metavar="")
    parser.add_argument("--gui_Train", default=DEFAULT_GUI_TRAIN, type=str2bool, help="Whether to use PyBullet GUI (default: True)", metavar="")
    parser.add_argument("--gui_Test", default=DEFAULT_GUI_TEST, type=str2bool, help="Whether to use PyBullet GUI (default: True)", metavar="")
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VIDEO, type=str2bool, help="Whether to record a video (default: False)", metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar="")
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar="")
    parser.add_argument("--pyb_freq", default=DEFAULT_PYB_FREQ, type=int, help="Physics frequency (default: 240)", metavar="")
    parser.add_argument("--ctrl_freq", default=DEFAULT_CTRL_FREQ, type=int, help="Control frequency (default: 60)", metavar="")
    parser.add_argument("--reward_and_action_change_freq", default=DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ, type=int, help="Control frequency (default: 60)", metavar="")
    parser.add_argument("--drone_model", default=DEFAULT_DRONE_MODEL, type=str, help="Control frequency (default: 60)", metavar="")
    parser.add_argument(
        "--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool, help="set to True if you want to see the debug GUI, only for showing the frame in training!(default: False)", metavar=""
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
        "--dash_active", default=DEFAULT_DASH_ACTIVE, type=str2bool, help="set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)", metavar=""
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
