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
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.examples.Test_Flo.BaseRLAviary_TestFlytoWall import BaseRLAviary_TestFlytoWall

# best_model_save_path = "/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/gym_pybullet_drones/examples/Test_Flo/results/save-01.09.2025_18.03.25"

DEFAULT_ALTITUDE = 0.5

INIT_XYZS = np.array(
    [
        [0, 0, DEFAULT_ALTITUDE],
    ]
)
INIT_RPYS = np.array(
    [
        [0, 0, 0],
    ]
)


def train(output_folder="results", guiAfterTraining=False, plot=True):
    # Create output directory
    filename = os.path.join(output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    # Create training environment
    train_env = make_vec_env(
        BaseRLAviary_TestFlytoWall,
        env_kwargs=dict(
            drone_model=DroneModel("cf2x"), initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS, physics=Physics.PYB, gui=False, ctrl_freq=48, reward_and_action_change_freq=0, act=ActionType.VEL
        ),
        n_envs=1,
        seed=0,
    )

    # Create evaluation environment
    eval_env = BaseRLAviary_TestFlytoWall(
        drone_model=DroneModel("cf2x"), initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS, gui=False, physics=Physics.PYB, ctrl_freq=48, reward_and_action_change_freq=0, act=ActionType.VEL
    )

    #### Check the environment's spaces ########################
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    # Initialize PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=filename + "/tb/")

    # Setup callbacks
    target_reward = 2800

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
        eval_env, callback_on_new_best=callback_on_best, verbose=1, best_model_save_path=filename + "/", log_path=filename + "/", eval_freq=int(1000), deterministic=True, render=False
    )
    # The model.learn function is used to train the model.
    # total_timesteps: The total number of timesteps to train for. It is set to 1e7 (10 million) if local is True, otherwise 1e2 (100) for shorter training in GitHub Actions pytest.
    # callback: The callback to use during training, in this case, eval_callback.
    # log_interval: The number of timesteps between logging events.
    # In your code, the model will train for a specified number of timesteps, using the eval_callback for periodic evaluation, and log information every 100 timesteps.

    # Train the model
    model.learn(total_timesteps=int(3000), callback=eval_callback, log_interval=100)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save final model
    model_path = os.path.join(results_dir, "final_model.zip")
    model.save(model_path)
    print(f"Training completed. Model saved to {model_path}")

    if guiAfterTraining:
        evaluate(model_path=model_path, gui=True)

    return filename


def evaluate(model_path, gui=True):
    """Evaluate a trained model"""
    model = PPO.load(model_path)

    # Create test environment
    env = BaseRLAviary_TestFlytoWall(
        drone_model=DroneModel("cf2x"),
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        gui=gui,
        physics=Physics.PYB,
        ctrl_freq=48,
        reward_and_action_change_freq=0,
        act=ActionType.VEL,
    )

    # Run evaluation episodes
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while steps < env.EPISODE_LEN_SEC * env.CTRL_FREQ:
        action, _states = model.predict(obs, deterministic=True)
        print(f"Raw action from model: {action}")
        action = env._preprocessAction(action)
        print(f"Processed action: {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

        if gui:
            env.render()
            time.sleep(1.0 / env.CTRL_FREQ)

    env.close()
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--evaluate", type=str, help="Path to model to evaluate")
    parser.add_argument("--gui", action="store_true", help="Show GUI during evaluation")
    # args = parser.parse_args()

    # if args.train:
    #     model_path = train(gui=False)
    #     print(f"Training completed. Model saved to {model_path}")

    # if args.evaluate:
    #     reward = evaluate(args.evaluate, gui=args.gui)
    #     print(f"Evaluation complete. Total reward: {reward}")

    model_path = train(guiAfterTraining=True)

    # reward = evaluate(model_path="/home/florian/Documents/gym-pybullet-drones/gym_pybullet_drones/examples/Test_Flo/results/final_model.zip", gui=True)
    # reward = evaluate(model_path='/home/alex/Documents/RKIM/Semester_1/F&E_1/Dronnenrennen_Group/gym-pybullet-drones/gym_pybullet_drones/examples/results/save-01.10.2025_15.57.01/final_model.zip', gui=True)
