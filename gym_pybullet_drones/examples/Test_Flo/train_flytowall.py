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

def train(output_folder="results", gui=False, plot=True):
    # Create output directory
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Create training environment
    train_env = make_vec_env(
        BaseRLAviary_TestFlytoWall,
        env_kwargs=dict(
            drone_model=DroneModel("cf2x"),
            initial_xyzs=np.array([[0., 0., 0.5]]),
            initial_rpys=np.array([[0., 0., 0.]]),
            gui=False,
            physics="PYB",
            ctrl_freq=48
        ),
        n_envs=1,
        seed=0
    )

    # Create evaluation environment
    eval_env = BaseRLAviary_TestFlytoWall(
        drone_model=DroneModel("cf2x"),
        initial_xyzs=np.array([[0., 0., 0.5]]),
        initial_rpys=np.array([[0., 0., 0.]]),
        gui=False,
        physics="PYB",
        ctrl_freq=48
    )

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=filename+'/tb/'
    )

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(
        total_timesteps=int(1e2),
        callback=eval_callback,
        log_interval=100
    )

    # Save final model
    model.save(filename+'/final_model.zip')
    print(f"Training completed. Model saved to {filename}")
    
    return filename

def evaluate(model_path, gui=True):
    """Evaluate a trained model"""
    model = PPO.load(model_path)
    
    # Create test environment
    env = BaseRLAviary_TestFlytoWall(
        drone_model="cf2x",
        initial_xyzs=np.array([[0., 0., 0.5]]),
        initial_rpys=np.array([[0., 0., 0.]]),
        gui=gui,
        physics="PYB",
        ctrl_freq=48
    )
    
    # Run evaluation episodes
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < env.EPISODE_LEN_SEC * env.CTRL_FREQ:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
            
        if gui:
            env.render()
            time.sleep(1./env.CTRL_FREQ)
    
    env.close()
    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', type=str, help='Path to model to evaluate')
    parser.add_argument('--gui', action='store_true', help='Show GUI during evaluation')
    args = parser.parse_args()

    if args.train:
        model_path = train(gui=False)
        print(f"Training completed. Model saved to {model_path}")
    
    if args.evaluate:
        reward = evaluate(args.evaluate, gui=args.gui)
        print(f"Evaluation complete. Total reward: {reward}")