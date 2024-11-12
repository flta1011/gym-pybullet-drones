import numpy as np
import pybullet as p
import time
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Simulation parameters
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Initial camera settings
camera_distance = 1
camera_yaw = 50
camera_pitch = -35
camera_target_position = [0, 0, 1]

# Initialize PyBullet
if DEFAULT_GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

# Load the environment
env = CtrlAviary(drone_model=DEFAULT_DRONES,
                 num_drones=DEFAULT_NUM_DRONES,
                 physics=DEFAULT_PHYSICS,
                 gui=DEFAULT_GUI,
                 record=DEFAULT_RECORD_VISION,
                 obstacles=DEFAULT_OBSTACLES,
                 user_debug_gui=DEFAULT_USER_DEBUG_GUI)

# Initialize PID controller
pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)

# Initial target position and orientation
target_pos = np.array([0.0, 0.0, 1.0])
target_rpy = np.array([0.0, 0.0, 0.0])

# Simulation parameters
time_step = 1 / DEFAULT_SIMULATION_FREQ_HZ
control_step = 1 / DEFAULT_CONTROL_FREQ_HZ

# Main loop
start_time = time.time()
while time.time() - start_time < DEFAULT_DURATION_SEC:
    keys = p.getKeyboardEvents()
    
    # Update target position based on keyboard input
    if ord('w') in keys:
        target_pos[1] += 0.1  # Forward
    if ord('s') in keys:
        target_pos[1] -= 0.1  # Backward
    if ord('a') in keys:
        target_pos[0] -= 0.1  # Left
    if ord('d') in keys:
        target_pos[0] += 0.1  # Right
    if ord('q') in keys:
        target_rpy[2] += 0.1  # Yaw left
    if ord('e') in keys:
        target_rpy[2] -= 0.1  # Yaw right
    if ord('r') in keys:
        target_pos[2] += 0.1  # Up
    if ord('f') in keys:
        target_pos[2] -= 0.1  # Down

    # Get current state of the drone
    cur_pos, cur_quat = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
    cur_vel, cur_ang_vel = p.getBaseVelocity(env.DRONE_IDS[0])

    # Compute control action using PID controller
    rpm, pos_e, yaw_e = pid_controller.computeControl(
        control_timestep=control_step,
        cur_pos=cur_pos,
        cur_quat=cur_quat,
        cur_vel=cur_vel,
        cur_ang_vel=cur_ang_vel,
        target_pos=target_pos,
        target_rpy=target_rpy
    )

    # Apply the computed RPMs to the drone motors
    for i in range(4):
        p.setMotorControl2(env.DRONE_IDS[0], i, p.VELOCITY_CONTROL, targetVelocity=rpm[i])

    # Step the simulation
    p.stepSimulation()
    time.sleep(time_step)

    # Update camera position
    drone_pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
    camera_target_position = drone_pos
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

# Disconnect from PyBullet
p.disconnect()