"""
Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface with SAC.

This example uses the SAC algorithm for single or multi-agent learning tasks.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
#1D (identical input to all motors) with RPMs
DEFAULT_ACT = ActionType('one_d_rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

#Training und Auswertung des Models
def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
    #Ergebnisse werden in einem Ordner gespeichert
    filename = os.path.join(output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')
    #Erstellt die Trainings- und Evaluierungsumgebungen basierend auf dem Einzel- oder Mehragentenmodus.
    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=True)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False)

    #### Check the environment's spaces ########################
    # Der Action-Space definiert die Steuerbefehle, die die Drohne an die Umgebung senden kann.
    # Box(-1.0, 1.0, (1, 1), float32):
    # Box: Der Aktionsraum ist ein kontinuierlicher Raum, der durch eine Box definiert wird.
    # -1.0: Die untere Grenze des Aktionsraums. Der minimale Wert, den eine Aktion annehmen kann, ist -1.0.
    # 1.0: Die obere Grenze des Aktionsraums. Der maximale Wert, den eine Aktion annehmen kann, ist 1.0.
    # (1, 1): Die Form des Aktionsraums. In diesem Fall ist es ein 2D-Array mit den Dimensionen (1, 1), was bedeutet, dass es eine einzelne Aktion gibt, die einen Wert hat.
    # float32: Der Datentyp der Aktionen. In diesem Fall sind die Aktionen float32-Werte.
    print('[INFO] Action space:', train_env.action_space)
    # Position (x, y, z): Die ersten drei Werte geben die Position der Drohne in den x-, y- und z-Koordinaten an.
    # Orientierung (Quaternion q1, q2, q3, q4): Die nächsten vier Werte geben die Orientierung der Drohne als Quaternion an.
    # Geschwindigkeit (vx, vy, vz): Die nächsten drei Werte geben die Geschwindigkeit der Drohne in den x-, y- und z-Richtungen an.
    # Winkelgeschwindigkeit (wx, wy, wz): Die nächsten drei Werte geben die Winkelgeschwindigkeit der Drohne um die x-, y- und z-Achsen an.
    # Aktionen aus dem Aktionspuffer: Die restlichen 17 Werte stammen aus dem Aktionspuffer und geben die letzten Aktionen an, die von der Drohne ausgeführt wurden.
    #Der Observation-Space beschreibt die Informationen, die die Drohne über die Umgebung erhält.
    #     Box: Der Beobachtungsraum ist ein kontinuierlicher Raum, der durch eine Box definiert wird.
    # [[-inf -inf 0. -inf -inf -inf -inf -inf -inf -inf -inf -inf -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]]: Die untere Grenze des Beobachtungsraums. 
    # Die Werte -inf und 0 bedeuten, dass es keine unteren Grenzen für diese Zustandsvariablen gibt, außer für die Höhe (z), die nicht negativ sein kann. Die Werte -1 beziehen sich auf die Aktionen aus dem Aktionspuffer.
    # [[inf inf inf inf inf inf inf inf inf inf inf inf 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]: Die obere Grenze des Beobachtungsraums. Die Werte inf bedeuten, dass es keine oberen Grenzen für diese Zustandsvariablen gibt. 
    # Die Werte 1 beziehen sich auf die Aktionen aus dem Aktionspuffer.
    # (1, 27): Die Form des Beobachtungsraums. In diesem Fall ist es ein 2D-Array mit den Dimensionen (1, 27), was bedeutet, dass es eine einzelne Beobachtung gibt, die 27 Werte hat.
    print('[INFO] Observation space:', train_env.observation_space)

    #     Position (x, y, z): Die ersten drei Werte geben die Position der Drohne in den x-, y- und z-Koordinaten an.
    # Orientierung (Quaternion q1, q2, q3, q4): Die nächsten vier Werte geben die Orientierung der Drohne als Quaternion an.
    # Geschwindigkeit (vx, vy, vz): Die nächsten drei Werte geben die Geschwindigkeit der Drohne in den x-, y- und z-Richtungen an.
    # Winkelgeschwindigkeit (wx, wy, wz): Die nächsten drei Werte geben die Winkelgeschwindigkeit der Drohne um die x-, y- und z-Achsen an.
    # Aktionen aus dem Aktionspuffer: Die restlichen 17 Werte stammen aus dem Aktionspuffer und geben die letzten Aktionen an, die von der Drohne ausgeführt wurden.

    #### Train the model #######################################
    ## CNN Policy wäre bei RBG-Bildern geeignet
    model = SAC('MlpPolicy',
                train_env,
                verbose=1,
                #tensorboard_log=filename + '/tb/'
                )
    # Initialisiere das SAC-Modell mit zusätzlichen Parametern
    # model = SAC(
    # 'MlpPolicy',  # oder 'CnnPolicy', 'MultiInputPolicy' je nach Bedarf
    # train_env,
    # verbose=1,
    # tensorboard_log=filename + '/tb/',
    # learning_rate=3e-4,  # Lernrate für den Optimierer
    # buffer_size=1_000_000,  # Größe des Replay-Puffers
    # learning_starts=100,  # Anzahl der Schritte, bevor das Lernen beginnt
    # batch_size=256,  # Größe der Mini-Batches für das Lernen
    # tau=0.005,  # Polyak-Glättungsfaktor für das Zielnetzwerk
    # gamma=0.99,  # Diskontierungsfaktor
    # train_freq=1,  # Frequenz des Trainings
    # gradient_steps=1,  # Anzahl der Optimierungsschritte nach jedem Rollout
    # ent_coef='auto',  # Koeffizient für den Entropieverlust
    # target_update_interval=1,  # Anzahl der Schritte, nach denen das Zielnetzwerk aktualisiert wird
    # target_entropy='auto',  # Zielentropie für die Entropieregulierung
    # use_sde=False,  # Ob stochastische differenzierbare Exploration verwendet werden soll
    # sde_sample_freq=-1,  # Frequenz der Neuprobenahme der SDE
    # use_sde_at_warmup=False,  # Ob SDE während des Warmups verwendet werden soll
    # create_eval_env=False,  # Ob eine Evaluierungsumgebung erstellt werden soll
    # policy_kwargs=None,  # Zusätzliche Argumente für die Politik
    # seed=None,  # Seed für die Zufallszahlengenerierung
    # device='auto'  # Gerät, auf dem das Modell trainiert wird
    # )
    
    #Definiert einen Callback, um das Training zu stoppen, wenn ein bestimmter 
    #Belohnungsschwellenwert erreicht wird, und einen Evaluierungs-Callback.

    target_reward = 474.15 if not multiagent else 949.5
    # return True if the mean reward is greater than the threshold
    # continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    # Die Klasse EvalCallback ermöglicht es, ein trainiertes Modell in regelmäßigen Abständen zu evaluieren, 
    # um zu überprüfen, wie gut es auf einer Evaluierungsumgebung abschneidet. Dies ist nützlich, um sicherzustellen, dass das Modell während des 
    # Trainings Fortschritte macht und um das beste Modell basierend auf der Evaluierungsleistung zu speichern.
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    #Startet das Training des Modells für eine bestimmte Anzahl von Zeitschritten.
    # model.learn(total_timesteps=int(1e4) if local else int(1e2),
    #             callback=eval_callback,
    #             log_interval=100)
     
    #### Save the model ########################################
    # model.save(filename + '/final_model.zip')
    # print(filename)

    #### Load the pre-trained model ############################
    model_path = '/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/results/save-12.09.2024_18.42.46/final_model.zip'
    if os.path.isfile(model_path):
        model = SAC.load(model_path)
    else:
        print("[ERROR]: no model under the specified path", model_path)

    # #### Print training progression ############################
    # with np.load(filename + '/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    # if os.path.isfile(filename + '/best_model.zip'):
    #     path = filename + '/best_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # model = SAC.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab)
    # Bewertet die Leistung des Modells in der Testumgebung.
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env,
                                              n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #Führt eine Simulation durch, bei der das trainierte Modell in der Testumgebung verwendet wird, um Aktionen vorherzusagen und die Umgebung zu steuern. 
    # Die Ergebnisse werden geloggt und die Umgebung wird gerendert.
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        #Eingabeparameter: Die Methode nimmt die aktuelle Beobachtung, den optionalen versteckten Zustand und die Episode-Start-Masken als Eingabeparameter.
        # Vorhersage der Aktion: Die Methode ruft die predict-Methode der policy-Instanz auf, um die Aktion basierend auf der Beobachtung vorherzusagen. Die policy-Instanz ist eine Instanz der Politikklasse, die die Logik zur Vorhersage von Aktionen enthält.
        # Rückgabe: Die Methode gibt die vorhergesagte Aktion und den nächsten versteckten Zustand zurück.
        # Die predict-Methode ist eine Schnittstelle, die es ermöglicht, basierend auf einer gegebenen Beobachtung und optionalen versteckten Zuständen Aktionen vorherzusagen. Sie ist besonders nützlich in Reinforcement-Learning-Szenarien, in denen der Agent 
        # kontinuierlich Entscheidungen treffen muss, basierend auf den Beobachtungen aus der Umgebung.
        action, _states = model.predict(obs, deterministic=True)
        #The input action for one or more drones, translated into RPMs by
        # the specific implementation of `_preprocessAction()` in each subclass.
        # obs: Die Beobachtung des aktuellen Zustands der Umgebung.
            # Beobachtungstyp KIN:
            # Kinematische Beobachtungen: Der Code extrahiert kinematische Zustandsvariablen (Position, Geschwindigkeit, Orientierung, Winkelgeschwindigkeit) für jede Drohne und speichert sie in einem Array.
            # Aktionspuffer: Der Code fügt die letzten Aktionen aus dem Aktionspuffer zur Beobachtung hinzu.
            # Rückgabe: Fügt die Aktionen aus dem Aktionspuffer zur Beobachtung hinzu. Der Aktionspuffer speichert die letzten Aktionen, die von den Drohnen ausgeführt wurden.
        # reward: Der Belohnungswert des aktuellen Schritts.
            #HOVER AVIARY Zustand der Drohne abrufen: Der Zustand der Drohne mit der ID 0 wird abgerufen, wobei state[0:3] die x-, y- und z-Position der Drohne enthält.
            # Belohnung berechnen: Die Belohnung wird basierend auf der Entfernung der Drohne zur Zielposition berechnet. Die Belohnung ist maximal 2 und nimmt exponentiell ab, je weiter die Drohne von der Zielposition entfernt ist. Wenn die Entfernung zu groß ist, wird die Belohnung auf 0 gesetzt.
        # terminated: Ein boolescher Wert, der angibt, ob die Episode beendet ist.
            # Episode: Eine vollständige Sequenz von Interaktionen zwischen einem Agenten und einer Umgebung, die mit einem Endzustand endet.
            #Überprüfen, ob die Episode beendet ist: Wenn die Entfernung kleiner als ein sehr kleiner Schwellenwert ist, wird True zurückgegeben, was bedeutet, dass die Episode beendet ist. Andernfalls wird False zurückgegeben.
        # truncated: Ein boolescher Wert, der angibt, ob die Episode abgebrochen wurde.
        # info: Zusätzliche Informationen als Dictionary.
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i / test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                           control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i / test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                               control=np.zeros(12))
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC reinforcement learning example script')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, help='Use MultiHoverAviary (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder for logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Run in Colab (default: False)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
