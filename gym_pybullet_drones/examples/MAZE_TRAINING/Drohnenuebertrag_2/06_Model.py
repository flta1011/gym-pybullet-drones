#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2017 Bitcraze AB
#
#  Crazyflie Python Library
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Example scripts that allows a user to "push" the Crazyflie 2.0 around
using your hands while it's hovering.

This examples uses the Flow and Multi-ranger decks to measure distances
in all directions and tries to keep away from anything that comes closer
than 0.2m by setting a velocity in the opposite direction.

The demo is ended by either pressing Ctrl-C or by holding your hand above the
Crazyflie.

For the example to run the following hardware is needed:
 * Crazyflie 2.0
 * Crazyradio PA
 * Flow deck
 * Multiranger deck
"""
import logging
import sys
import time

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
from stable_baselines3 import DQN, PPO, SAC

URI = uri_helper.uri_from_env(default="radio://0/60/2M/E7E7E7E7E7")

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def is_close(range):
    MIN_DISTANCE = 0.3  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE


def is_close_up(range):
    MIN_DISTANCE = 1.0  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE


if __name__ == "__main__":
    last_actions = np.zeros(100)
    model = SAC.load("/home/moritz_s/Desktop/Test_Series/04/M6_R6_O10_A3_TR1_T1_20250427_003400_SAC_O10_035_Abstand_5Hz_100LA_neue_Mazes_ohne_SLAM_neu_ueber_nacht/_2200000_steps.zip")
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache="./cache")
    with SyncCrazyflie(URI, cf=cf) as scf:
        # Arm the Crazyflie
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        with MotionCommander(scf) as motion_commander:
            with Multiranger(scf) as multiranger:
                keep_flying = True

                while keep_flying:
                    VELOCITY = 0.25
                    velocity_x = 0.0
                    velocity_y = 0.0
                    # Observation space
                    # Replace None values with a default distance (e.g., 5.0 meters)
                    front = multiranger.front if multiranger.front is not None else 2.1
                    back = multiranger.back if multiranger.back is not None else 2.1
                    left = multiranger.left if multiranger.left is not None else 2.1
                    right = multiranger.right if multiranger.right is not None else 2.1

                    obs_list = [front, back, left, right]
                    obs_list.extend(last_actions)
                    # Convert the whole list to a numpy array
                    observation = np.array(obs_list, dtype=np.float32)  # Explicitly set dtype
                    # Prediction
                    action, _ = model.predict(observation, deterministic=True)
                    velocity_x = action[0] * 0.4
                    velocity_y = action[1] * 0.4

                    # Pushback
                    if is_close(multiranger.front):
                        velocity_x -= VELOCITY
                    if is_close(multiranger.back):
                        velocity_x += VELOCITY

                    if is_close(multiranger.left):
                        velocity_y -= VELOCITY
                    if is_close(multiranger.right):
                        velocity_y += VELOCITY

                    if is_close_up(multiranger.up):
                        keep_flying = False

                    # write last action
                    last_actions = np.roll(last_actions, 2)
                    # Handle different action formats (scalar or array)
                    last_actions[0] = velocity_x
                    last_actions[1] = velocity_y

                    motion_commander.start_linear_motion(velocity_x, velocity_y, 0)

                    time.sleep(0.1)

            print("Terminated!")
