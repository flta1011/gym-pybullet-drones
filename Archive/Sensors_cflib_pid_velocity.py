"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use internal PID control to track a target velocity.

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

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_DRONE = DroneModel("cf2x") # x = Configuration of the rotors
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True # plot the simulation results
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_CONTROL_MODE = 'Keyboard'  # 'Keyboard' for manual control using keyboard inputs,
                                   # 'PID' for automated PID control (automatic parts will be deleted)
DEFAULT_ALTITUDE = 0.5  # Altitude at which the drone will hover in meters)
DEFAULT_NUM_DRONES = 1


# Define key mappings
# [X, Y, Z, Scaling factor]
KEY_MAPPING = {
    'up': np.array([1, 0, 0, 0.99]), # Up
    'down': np.array([-1, 0, 0, 0.99]), # Down
    'left': np.array([0, 1, 0, 0.99]), # Yaw left
    'right': np.array([0, -1, 0, 0.99]) # Yaw right
    }

#def get_keyboard_events():
#    keys = p.getKeyboardEvents()
#    movement = np.zeros(4)
#    for k, v in keys.items():
#        if (v & p.KEY_WAS_TRIGGERED) or (v & p.KEY_IS_DOWN):
#            if chr(k) in KEY_MAPPING:
#                movement += KEY_MAPPING[chr(k)]
#            elif k == p.B3G_UP_ARROW:
#                movement += KEY_MAPPING['up']
#            elif k == p.B3G_DOWN_ARROW:
#                movement += KEY_MAPPING['down']
#            elif k == p.B3G_LEFT_ARROW:
#                movement += KEY_MAPPING['left']
#            elif k == p.B3G_RIGHT_ARROW:
#                movement += KEY_MAPPING['right']
#    return movement

def get_keyboard_events():
    keys = p.getKeyboardEvents()
    movement = np.zeros(4)
    
    if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED or keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
        movement += KEY_MAPPING['up']
    if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED or keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
        movement += KEY_MAPPING['down']
    if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED or keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
        movement += KEY_MAPPING['left']
    if p.B3G_RIGHT_ARROW in keys and (p.KEY_WAS_TRIGGERED or keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
        movement += KEY_MAPPING['right']
    
    for k, v in keys.items():
        if chr(k) in KEY_MAPPING and ((v & p.KEY_WAS_TRIGGERED) or (v & p.KEY_IS_DOWN)):
            movement += KEY_MAPPING[chr(k)]

    return movement


# Get Sensor Data
def check_distance_sensors(self, crazyflie_id):
        """
        Check the distance sensors of the Crazyflie drone.
        Args:
            crazyflie_id (int): The PyBullet body ID of the Crazyflie drone.
        Returns:
            list: Sensor readings for each direction (forward, backward, left, right, up, down).
                Each reading is the distance to the nearest obstacle or max_distance if no obstacle is detected.
        """
        pos, ori = p.getBasePositionAndOrientation(crazyflie_id)
        
        local_directions = np.array([
            [1, 0, 0],    # Forward
            [-1, 0, 0],   # Backward
            [0, 1, 0],    # Left
            [0, -1, 0],   # Right
            [0, 0, 1],    # Up
            [0, 0, -1],   # Down
        ])
        
        max_distance = 4  # meters
        sensor_readings = []
        
        # Convert quaternion to rotation matrix using NumPy
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        
        for direction in local_directions:
            # Transform local direction to world direction
            world_direction = rot_matrix.dot(direction)
            
            to_pos = pos + world_direction * max_distance
            
            ray_result = p.rayTest(pos, to_pos)
            hit_object_id = ray_result[0][0]
            hit_fraction = ray_result[0][2]
            
            if hit_object_id != -1 and hit_fraction > 0:
                distance = hit_fraction * max_distance
            else:
                distance = None  # No obstacle detected within max_distance
            
            sensor_readings.append(distance)
        
        return sensor_readings



#################################
# Simulierte Sensoren integrieren
#################################

# Get Sensor Data
def simulate_sensors(drone_id):
    # Hole Drohnenposition und Geschwindigkeit
    pos, _ = p.getBasePositionAndOrientation(drone_id)
    linear_velocity, _ = p.getBaseVelocity(drone_id)
    
    # Simuliere optischen Fluss (FlowDeck)
    optical_flow_x = linear_velocity[0]  # Bewegung in X-Richtung
    optical_flow_y = linear_velocity[1]  # Bewegung in Y-Richtung
    
    # Simuliere Abstandsmessungen (MultiRanger)
    ray_directions = [
        [1, 0, 0], [-1, 0, 0],  # Vorne, Hinten
        [0, 1, 0], [0, -1, 0],  # Links, Rechts
        [0, 0, -1], [0, 0, 1],  # Unten, Oben
    ]
    ray_results = p.rayTestBatch(
        [pos] * len(ray_directions),
        [[pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]] for d in ray_directions]
    )
    distances = [r[2] for r in ray_results]  # Trefferpositionen auswerten
    
    return {
        "flow_x": optical_flow_x,
        "flow_y": optical_flow_y,
        "ranger": distances
    }

#########################################################
# Crazyflie-Bibliothek mit simulierten Sensoren verbinden
#########################################################

from cflib.crazyflie import Crazyflie

class SimulatedDeck:
    def __init__(self, simulation_fn):
        self.simulation_fn = simulation_fn

    def read_sensors(self):
        return self.simulation_fn()

###################################
# Verbindung zur Crazyflie Firmware
###################################

def main():
    cf = Crazyflie(rw_cache="./cache")
    cf.open_link("simulated://")  # Virtueller Link für die Simulation

    # Initialisiere dein Simulated Deck
    simulated_deck = SimulatedDeck(simulate_sensors)
    
    # Schreibe eine Endlosschleife für die Sensorabfrage
    while True:
        sensors = simulated_deck.read_sensors()
        print("Simulierte Sensoren:", sensors)

if __name__ == "__main__":
    main()


#####################################
# Simulationserweiterung mit Bewegung
#####################################

from cflib.crazyflie.commander import Commander

def control_drone(commander: Commander):
    # Sende Steuerbefehle an die Drohne
    commander.send_velocity_world_setpoint(0.1, 0, 0, 0)  # Bewege in X-Richtung


#######################################
#
#######################################




# Run the Simulation
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
        #### Initialize the simulation #############################
    INIT_XYZS = np.array([
                          [ 0, 0, DEFAULT_ALTITUDE],
                          ])
    INIT_RPYS = np.array([
                          [0, 0, 0],
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
    wp_counters = np.array([0 for i in range(1)])

    ##### Initialize the velocity target ########################
    #TARGET_VEL = np.zeros((4,NUM_WP,4))
    #for i in range(NUM_WP):
    #    TARGET_VEL[0, i, :] = [-0.5, 1, 0, 0.99] if i < (NUM_WP/8) else [0.5, -1, 0, 0.99]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Run the simulation ####################################
    action = np.zeros((DEFAULT_NUM_DRONES,4))
    START = time.time()

    # initial target position
    TARGET_POS = np.copy(INIT_XYZS)

    # initial camera position
    #camera_distance = 1
    #camera_yaw = 50
    #camera_pitch = -35
    #camera_target_position = [0, 0, 1]

    for i in range(0, NUM_WP): # for each control step

        #### Capture keyboard events ############################
        movement = get_keyboard_events()
        print(f"Movement: {movement}")

        for j in range(DEFAULT_NUM_DRONES):
            action[j, :] = movement
            print(f"Action for drone {action[j, :]}")

        
        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        #### Alex:tbd Update the camera position ###########################
        # events = p.getMouseEvents()
        # for e in events:
        #     if e[0] == p.MOUSE_WHEEL:
        #         cameraDistance -= e[2] * 0.1  # Adjust zoom speed as needed

        drone_pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
        p.resetDebugVisualizerCamera(cameraDistance=2,
                                 cameraYaw=-90,
                                 cameraPitch=-30,
                                 cameraTargetPosition=drone_pos,
                                 physicsClientId=env.CLIENT
                                 )
        
        ##### PID ### Compute control for the current way point #############
        #for j in range(num_drones):
        #    action[j, :] = TARGET_VEL[j, wp_counters[j], :] 

        # Get ToF sensor readings for the first drone TBD ALEX
        # tof_readings = env._getToFSensorReadings(env.DRONE_IDS[0])
        # print(f"ToF Sensor Readings: {tof_readings}")

        #### Go to the next way point and loop #####################
        for j in range(DEFAULT_NUM_DRONES):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(DEFAULT_NUM_DRONES):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state= obs[j],
                       #control=np.hstack([TARGET_VEL[j, wp_counters[j], 0:3], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        
        print(f"Step {i}")

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

