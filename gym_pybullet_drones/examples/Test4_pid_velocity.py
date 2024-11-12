"""Script demonstrating the joint use of velocity input.

The simulation is run by a `Test4_VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.Test4_VelocityAviary import VelocityAviary

DEFAULT_DRONE = DroneModel("cf2x") # x steht für x configuration der Rotorblätter
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True # Plot the simulation results
DEFAULT_USER_DEBUG_GUI = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_CONTROL_MODE = 'Keyboard' # Keyboard for manual control, 'PID' for automated PID control --> Automatic parts will be deleted
DEFAULT_ALTITUDE = 0.5  # Altitude at which the drone will hover in meters)

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        control_mode=DEFAULT_CONTROL_MODE
        ):
        #### Initialize the simulation ############################# # 3 Drohnen raustgeschmissen
    INIT_XYZS = np.array([
                          [ 0, 0, .1],
                          ])
    INIT_RPYS = np.array([
                          [0, 0, 0], # yaw, pitch, roll (in Radiant), es könnte z.B. np.pi/2 sein
                          ])
    PHY = Physics.PYB
    
    
  
    

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=1,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()
    
    
    


    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz*PERIOD
    wegpunkt_counters = np.array([0 for i in range(1)])
    

    #### Initialize the velocity target ########################   # die 3 anderen Drohen sind rausgeschmissen
    TARGET_VEL = np.zeros((1,NUM_WP,4))
    for i in range(NUM_WP): # für jeden Calculationschritt gibt es ein definiertes Set an Velocities 
        TARGET_VEL[0, i, :] = [-0.5, 1, 0, 0.99] if i < (NUM_WP/8) else [0.5, -1, 0, 0.99]


    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Run the simulation ####################################
    action = np.zeros((1,4)) # nur noch 1 Drohne!
    START = time.time()
    
    # Initiale Kameraeinstellungen
    camera_distance = 1
    camera_yaw = 50
    camera_pitch = -35
    camera_target_position = [0, 0, 1]
    
    i = 0
    while True:

        #### Per Keyboard debuggen ################################
        desired_state_roll = 0
        desired_state_pitch = 0
        desired_state_vx = 0
        desired_state_vy = 0
        desired_state_yaw_rate = 0
        desired_state_altitude = DEFAULT_ALTITUDE
        
        #Keyboard inputs werden hierin gespeichert
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0
        
        #keyboard inputs lesen
        keys = p.getKeyboardEvents()
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
            forward_desired = 0.5
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED:
            forward_desired = -0.5
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
            sideways_desired = -0.5
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
            sideways_desired = 0.5
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            yaw_desired = 1.0
        if ord('e') in keys and keys[ord('e')] & p.KEY_WAS_TRIGGERED:
            yaw_desired = -1.0
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            break
        
        #write desired from keyboard into the desired state
        desired_state_yaw_rate = yaw_desired
        desired_state_vy = sideways_desired
        desired_state_vy  = forward_desired
        
        
        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(1): # nur noch 1 Drohne!
            action[j, :] = TARGET_VEL[j, wegpunkt_counters[j], :] 

        #### Go to the next way point and loop #####################
        for j in range(1): # nur noch 1 Drohne!
            wegpunkt_counters[j] = wegpunkt_counters[j] + 1 if wegpunkt_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(1): # nur noch 1 Drohne!
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state= obs[j],
                       control=np.hstack([TARGET_VEL[j, wegpunkt_counters[j], 0:3], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        
        print(f"Step {i}")
        
        i += 1
        
    

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("vel") # Optional CSV save
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Velocity control example using VelocityAviary')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,      type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

