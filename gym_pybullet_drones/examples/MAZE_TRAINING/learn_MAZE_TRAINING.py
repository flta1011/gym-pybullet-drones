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
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.examples.MAZE_TRAINING.Logger_MAZE_TRAINING_BUGGY import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.examples.MAZE_TRAINING.BaseRLAviary_MAZE_TRAINING import BaseRLAviary_MAZE_TRAINING
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# ACHTUNG: es können nicht beide Werte auf TRUE gesetzt werden (nicht GUI_TRAIN und GUI_TEST zusammen)!
DEFAULT_GUI_TRAIN = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_ADVANCED_STATUS_PLOT = True

DEFAULT_GUI_TEST = False

DEFAULT_USE_PRETRAINED_MODEL = False

DEFAULT_PRETRAINED_MODEL_PATH = ''

DEFAULT_EVAL_FREQ = 5*1e4
DEFAULT_EVAL_EPISODES = 1

DEFAULT_TRAIN_TIMESTEPS = 1*1e5 # nach 100000 Steps sollten schon mehrbahre Erkenntnisse da sein
DEFAULT_TARGET_REWARD = 20000

DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_PYB_FREQ = 100
DEFAULT_CTRL_FREQ = 50
DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ = 5
DEFAULT_DRONE_MODEL = DroneModel("cf2x")

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

DEFAULT_ALTITUDE = 0.5

INIT_XYZS = np.array([
                          [0, 0, DEFAULT_ALTITUDE],
                          ])
INIT_RPYS = np.array([
                          [0, 0, 0],
                          ])





def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui_Train=DEFAULT_GUI_TRAIN, gui_Test=DEFAULT_GUI_TEST, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, pyb_freq=DEFAULT_PYB_FREQ, ctrl_freq=DEFAULT_CTRL_FREQ, user_debug_gui=DEFAULT_USER_DEBUG_GUI, reward_and_action_change_freq=DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ, drone_model=DEFAULT_DRONE_MODEL, advanced_status_plot=DEFAULT_ADVANCED_STATUS_PLOT):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
        
        # ANCHOR - learn_MAZE_TRAINING ENVS
        train_env = make_vec_env(BaseRLAviary_MAZE_TRAINING,
                        env_kwargs=dict(
                            drone_model=drone_model,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            physics=Physics.PYB,
                            gui=gui_Train,
                            user_debug_gui=user_debug_gui,
                            pyb_freq=pyb_freq,
                            ctrl_freq=ctrl_freq, # Ansatz: von 60 auf 10 reduzieren, damit die gewählte Action länger wirkt
                            reward_and_action_change_freq=reward_and_action_change_freq, # Ansatz: neu hinzugefügt, da die Step-Funktion vorher mit der ctrl_freq aufgerufen wurde, Problem war dann, dass bei hoher Frequenz die Raycasts keine Änderung hatten, dafür die Drohne aber sauber geflogen ist (60). Wenn der Wert niedriger war, hat es mit den Geschwindigkeiten und Actions besser gepasst, dafür ist die Drohne nicht sauber geflogen, weil die Ctrl-Frequenz für das erreichen der gewählten Action zu niedrig war (10/20).
                            act=ActionType.VEL
                            ),
                        n_envs=1,
                        seed=0
                        )
        #if 'train_env' in locals():
            #train_env.close()
        
        eval_env = make_vec_env(BaseRLAviary_MAZE_TRAINING,
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
                            act=ActionType.VEL
                            ),
                        n_envs=1,
                        seed=0
                        )
        #if 'eval_env' in locals():
           # eval_env.close()

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Load existing model or create new one ###################
    if DEFAULT_USE_PRETRAINED_MODEL and os.path.exists(DEFAULT_PRETRAINED_MODEL_PATH):
        print(f"[INFO] Loading existing model from {DEFAULT_PRETRAINED_MODEL_PATH}")
        model = PPO.load(DEFAULT_PRETRAINED_MODEL_PATH, env=train_env)
    else:
        print("[INFO] Creating new model")
        model = PPO('MlpPolicy',
                   train_env,
                   verbose=1,
                   learning_rate=0.0004, # 0,0002 zu gering -> auf 0.0004 erhöht -> auf 0.0005 erhöht --> auf 0.0004 reduziert, da die Policy zu stark angepasst wurde, obwohl es schon 5s am Ziel war..
                   )
    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = DEFAULT_TARGET_REWARD
    print(target_reward)
    #The StopTrainingOnRewardThreshold callback is used to stop the training once a certain reward threshold is reached.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    #The EvalCallback is used to evaluate the agent periodically during training.
    # eval_env: The environment used for evaluation.
    # callback_on_new_best: Callback to trigger when a new best model is found.
    # verbose=1: Info messages will be printed during evaluation.
    # best_model_save_path: Path to save the best model.
    # log_path: Path to save evaluation logs.
    # eval_freq: Frequency of evaluations (every 1000 steps in this case).
    # deterministic=True: Use deterministic actions during evaluation.
    # render=False: Do not render the environment during evaluation.
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=DEFAULT_EVAL_FREQ, # alle 10000 Schritte wird die Evaluation durchgeführt (mit Frequenz reward_and_action_change_freq)
                                 deterministic=True, 
                                 render=False , # nicht auf True setzbar, da dem RL-Environment keine render_mode="human"übergeben werden kann
                                 n_eval_episodes=1)# neu eingefügt, dass es schneller durch ist mit der Visu
    #The model.learn function is used to train the model.
    # total_timesteps: The total number of timesteps to train for. It is set to 1e7 (10 million) if local is True, otherwise 1e2 (100) for shorter training in GitHub Actions pytest.
    # callback: The callback to use during training, in this case, eval_callback.
    # log_interval: The number of timesteps between logging events.
    # In your code, the model will train for a specified number of timesteps, using the eval_callback for periodic evaluation, and log information every 100 timesteps.
    model.learn(total_timesteps=DEFAULT_TRAIN_TIMESTEPS, # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=1000,
                progress_bar=True)
    
    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##

    test_env = BaseRLAviary_TestFlytoWall(gui=gui_Test,
                            act=DEFAULT_ACT,
                            record=record_video)
    test_env_nogui = BaseRLAviary_TestFlytoWall(act=DEFAULT_ACT)
   
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )
    #The evaluate_policy function is used to evaluate the performance of the trained model.
    # model: The trained model to be evaluated.
    # test_env_nogui: The environment used for evaluation without GUI.
    # n_eval_episodes=10: The number of episodes to run for evaluation.
    # In your code, the function will evaluate the model over 10 episodes and return the mean and standard deviation of the rewards.
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    #The reset function is used to reset the environment to its initial state.
    # seed=42: The seed for the random number generator to ensure reproducibility.
    # options={}: Additional options for resetting the environment.
    # In your code, obs will contain the initial observation, and info will contain additional information provided by the environment after resetting.
    
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    #This code runs a loop to simulate the environment using the trained model and logs the results.
    # Loop: Runs for a specified number of steps.
    # Predict Action: Uses the model to predict the next action.
    # Step: Takes the action in the environment and receives the next observation, reward, and termination status.
    # Log: Logs the state and action if the observation type is KIN.
    # Render: Renders the environment.
    # Sync: Synchronizes the simulation.
    # Reset: Resets the environment if terminated.  
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):

        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12
                                     )
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui_Train',                default=DEFAULT_GUI_TRAIN,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--gui_Test',                default=DEFAULT_GUI_TEST,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--pyb_freq',          default=DEFAULT_PYB_FREQ,    type=int,           help='Physics frequency (default: 240)', metavar='')
    parser.add_argument('--ctrl_freq',          default=DEFAULT_CTRL_FREQ,    type=int,           help='Control frequency (default: 60)', metavar='')
    parser.add_argument('--reward_and_action_change_freq',          default=DEFAULT_REWARD_AND_ACTION_CHANGE_FREQ,    type=int,           help='Control frequency (default: 60)', metavar='')
    parser.add_argument('--drone_model',          default=DEFAULT_DRONE_MODEL,    type=str,           help='Control frequency (default: 60)', metavar='')
    parser.add_argument('--user_debug_gui',          default=DEFAULT_USER_DEBUG_GUI,    type=str2bool,           help='set to True if you want to see the debug GUI, only for showing the frame in training!(default: False)', metavar='')
    parser.add_argument('--advanced_status_plot',          default=DEFAULT_ADVANCED_STATUS_PLOT,    type=str2bool,           help='set to True if you want to see the advanced status plot, only for showing the frame in training!(default: False)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
