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
from threading import Event

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
from stable_baselines3 import DQN, PPO, SAC

URI = uri_helper.uri_from_env(default="radio://0/60/2M/E7E7E7E7E7")

if len(sys.argv) > 1:
    URI = sys.argv[1]

DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0, 0]


def is_close(range):
    MIN_DISTANCE = 0.3  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE


def is_close(range):
    MIN_DISTANCE = 1.0  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE


def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()


def log_pos_callback(timestamp, data, logconf):
    # print(data)
    global position_estimate
    position_estimate[0] = data["stateEstimate.x"]
    position_estimate[1] = data["stateEstimate.y"]
    position_estimate[2] = data["stabilizer.yaw"]


def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print("Deck is attached!")
    else:
        print("Deck is NOT attached!")


class SimpleSlam:
    def __init__(self, map_size=18, cropped_map_size=6.4, resolution=0.05, init_position=None):
        """Erstellt eine leere Occupancy-Grid Map.
        Args:
            map_size (float): Seitengröße der Map in Metern (z. B. 8 m).
            resolution (float): Seitengröße einer Zelle, sodass grid_size ~60 ergibt.
            init_position (tuple): (x, y)-Startposition der Drohne in Metern.
        """
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.cropped_map_size_grid = int((cropped_map_size / resolution) / 2)

        # Initialisiere die Map-Werte:
        self.unbekannt_value = 0
        self.frei_value = 200
        self.wand_value = 50
        self.besucht_value = 125
        self.actual_Position_value = 255
        self.init = init_position
        self.resolution = resolution
        self.previous_grid_x = 0
        self.previous_grid_y = 0
        self.interest_values = np.zeros(4, dtype=int)  # [front, back, left, right]

        # Berechne die Map-Offsets basierend auf der init_position
        if init_position is not None:
            init_x, init_y = init_position
            self.offset_x = int(init_x / resolution)
            self.offset_y = int(init_y / resolution)
            if not (0 <= self.offset_x < self.grid_size and 0 <= self.offset_y < self.grid_size):
                raise ValueError("init_position muss innerhalb der Map-Grenzen liegen.")
        else:
            self.offset_x = self.grid_size // 2
            self.offset_y = self.grid_size // 2

        # Initialisiere die Map:
        self.occupancy_grid = self.unbekannt_value * np.ones((self.grid_size, self.grid_size, 1))
        self.cropped_grid = self.unbekannt_value * np.ones((self.cropped_map_size_grid, self.cropped_map_size_grid, 1))
        self.center = self.grid_size // 2
        self.center_cropped = self.cropped_map_size_grid // 2
        self.path = []  # speichert besuchte Zellen
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []
        self.counter_free_space = 0

    def reset(self):
        """Reset the SLAM map to its initial state."""
        self.occupancy_grid = self.unbekannt_value * np.ones((self.grid_size, self.grid_size, 1))
        self.path = []
        self.DrohnePosition = []
        self.Prev_DrohnePosition = []

    def world_to_grid(self, x, y):
        grid_x = int(self.center + x / self.resolution)
        grid_y = int(self.center + y / self.resolution)
        return grid_x, grid_y

    def is_within_bounds(self, grid_x, grid_y):
        """Überprüft, ob die Drohne innerhalb der Map-Grenzen bleibt."""
        return 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size

    def update(self, drone_pos, drone_yaw, raycast_results):

        # NOTE - Werte anpassen wie ich welches Szenario darstellen möchte TBD 24.03.2025
        # NOTE - 0.2: unbekannt, 0.9: frei, 0.0: Wand, 0.5: besucht (Sensor oben frei), 0.7 = aktuelle Position

        """
        Aktualisiert die Map anhand der Sensorwerte.
        Args:
            drone_pos (tuple): (x, y, z)-Position der Drohne.
            drone_yaw (float): Yaw-Winkel (in Radiant).
            raycast_results (dict): z. B. { 'front': d_front, 'back': d_back,
                                            'left': d_left, 'right': d_right, 'up': d_up }
        """
        x, y, _ = drone_pos

        #!SECTION Erkenntnis damit im Training Koordinaten in Globalen waren und hier aber die Drohne Koordinaten hat
        neues_x = x
        neues_y = y

        # Iterate through 5x5 grid centered on current position --> 3x3 grid

        grid_x, grid_y = self.world_to_grid(neues_x, neues_y)

        # FALLBACK für Errror "index 210 is out of bounds for axis 0 with size 180"
        if grid_x > self.grid_size:
            grid_x = self.grid_size
        if grid_y > self.grid_size:
            grid_y = self.grid_size
        # Fallback für Cases < 0 (sollten eigentlich nie auftreten)
        if grid_x < 0:
            grid_x = 0
        if grid_y < 0:
            grid_y = 0

        # self.path.append((grid_x, grid_y))
        # Markiere aktuelle Zelle als frei:
        self.occupancy_grid[grid_x, grid_y] = self.frei_value
        self.grid_x = grid_x
        self.grid_y = grid_y
        # self.counter_free_space += 1

        # Definiere Richtungswinkel:
        angles = {"front": drone_yaw, "back": drone_yaw + np.pi, "left": drone_yaw + np.pi / 2, "right": drone_yaw - np.pi / 2}
        for direction in ["front", "back", "left", "right"]:
            distance = raycast_results.get(direction, 9999)
            if distance < 2.1:  # Treffer – Wand erkannt
                angle = angles[direction]
                end_x = neues_x + distance * np.cos(angle)
                end_y = neues_y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                # Markiere Zellen entlang der Strahlbahn als frei:
                for cx, cy in cells[:-1]:  # alles bis auf den Endpunkt frei markieren
                    if (
                        0 <= cx < self.grid_size
                        and 0 <= cy < self.grid_size
                        and self.occupancy_grid[cx, cy] != self.wand_value
                        and self.occupancy_grid[cx, cy] != self.besucht_value
                        and self.occupancy_grid[cx, cy] != self.actual_Position_value
                    ):
                        self.occupancy_grid[cx, cy] = self.frei_value
                        self.counter_free_space += 1
                    if distance == 0.05:
                        if distance == 0.05 and direction == "front":
                            self.occupancy_grid[cx + 1, cy] = self.wand_value
                            self.occupancy_grid[cx + 2, cy] = self.wand_value
                            self.occupancy_grid[cx + 3, cy] = self.wand_value
                        elif distance == 0.05 and direction == "back":
                            self.occupancy_grid[cx - 1, cy] = self.wand_value
                            self.occupancy_grid[cx - 2, cy] = self.wand_value
                            self.occupancy_grid[cx - 3, cy] = self.wand_value
                        elif distance == 0.05 and direction == "left":
                            self.occupancy_grid[cx, cy - 1] = self.wand_value
                            self.occupancy_grid[cx, cy - 2] = self.wand_value
                            self.occupancy_grid[cx, cy - 3] = self.wand_value
                        elif distance == 0.05 and direction == "right":
                            self.occupancy_grid[cx, cy + 1] = self.wand_value
                            self.occupancy_grid[cx, cy + 2] = self.wand_value
                            self.occupancy_grid[cx, cy + 3] = self.wand_value

                # Markiere den Endpunkt als Wand:
                if 0 <= end_grid_x < self.grid_size and 0 <= end_grid_y < self.grid_size:
                    self.occupancy_grid[end_grid_x, end_grid_y] = self.wand_value

            elif distance >= 2.1:  # Distanz ist auf 4 m gekappt, da das die Range des Sensors ist --> alles als frei markieren
                angle = angles[direction]
                end_x = neues_x + distance * np.cos(angle)
                end_y = neues_y + distance * np.sin(angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                cells = self.get_line(grid_x, grid_y, end_grid_x, end_grid_y)
                # Markiere Zellen entlang der Strahlbahn als frei:
                for cx, cy in cells[:]:  # alles frei markieren
                    if (
                        0 <= cx < self.grid_size
                        and 0 <= cy < self.grid_size
                        and self.occupancy_grid[cx, cy] != self.wand_value
                        and self.occupancy_grid[cx, cy] != self.besucht_value
                        and self.occupancy_grid[cx, cy] != self.actual_Position_value
                    ):
                        self.occupancy_grid[cx, cy] = self.frei_value
                        self.counter_free_space += 1

        self.Prev_DrohnePosition = self.DrohnePosition

        self.DrohnePosition = []
        # Überschreibe die vorherige aktuelle Position mit besucht_value
        if hasattr(self, "i_value_previous") and hasattr(self, "j_value_previous"):
            for i_prev, j_prev in zip(self.i_value_previous, self.j_value_previous):
                self.occupancy_grid[i_prev, j_prev] = self.besucht_value

        # self.cropped_grid = self.occupancy_grid[]

        # Speichere die aktuelle Position als vorherige Position
        self.i_value_previous = []
        self.j_value_previous = []

        for i in range(max(0, grid_x - 1), min(self.grid_size, grid_x + 2)):
            for j in range(max(0, grid_y - 1), min(self.grid_size, grid_y + 2)):
                self.path.append((i, j))
                self.DrohnePosition.append((i, j))
                if (
                    self.occupancy_grid[i, j] == self.besucht_value
                    or self.occupancy_grid[i, j] == self.unbekannt_value
                    or self.occupancy_grid[i, j] == self.frei_value
                    or self.occupancy_grid[i, j] == self.actual_Position_value
                    and self.occupancy_grid[i, j] != self.wand_value
                ):
                    # self.occupancy_grid[i, j] = self.besucht_value
                    self.occupancy_grid[i, j] = self.actual_Position_value

                    self.i_value_previous.append(i)
                    self.j_value_previous.append(j)

                # if self.occupancy_grid[i, j] == self.besucht_value:
                #     self.occupancy_grid[i, j] = self.actual_Position_value

        # Compare previous and current positions to determine movement direction
        if self.grid_y and self.grid_x and self.previous_grid_x and self.previous_grid_y:
            # # Calculate the center of the previous and current positions
            # delta_x = self.grid_x - self.previous_grid_x
            # delta_y = self.grid_y - self.previous_grid_y

            # if abs(delta_x) > abs(delta_y):  # Movement is primarily horizontal
            #     if delta_x > 0:
            #         movement = "right"
            #     elif delta_x < 0:
            #         movement = "left"
            # else:  # Movement is primarily vertical
            #     if delta_y > 0:
            #         movement = "forward"
            #     elif delta_y < 0:
            #         movement = "backward"

            # Update cropped grid based on movement
            cropped_start_x = int(max(0, self.grid_x - self.center_cropped))
            cropped_end_x = int(min(self.grid_size, self.grid_x + self.center_cropped))
            cropped_start_y = int(max(0, self.grid_y - self.center_cropped))
            cropped_end_y = int(min(self.grid_size, self.grid_y + self.center_cropped))

            # Ensure the cropped grid is exactly 64x64
            if cropped_end_x - cropped_start_x != 64:
                cropped_end_x = cropped_start_x + 64
            if cropped_end_y - cropped_start_y != 64:
                cropped_end_y = cropped_start_y + 64

            self.cropped_grid = self.occupancy_grid[cropped_start_x:cropped_end_x, cropped_start_y:cropped_end_y]

        self.previous_grid_x = self.grid_x
        self.previous_grid_y = self.grid_y

    def get_line(self, x0, y0, x1, y1):
        """Berechnet Zellen entlang einer Linie (Bresenham-Algorithmus)."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x, y))
        return cells

    def _compute_interest_values(self):
        """
        Compute interest values based on current drone position.
        """
        drone_position = np.argwhere(self.occupancy_grid[:, :, 0] == 255)  # Get the drone position

        # Check if drone position was found
        if len(drone_position) == 0:
            print("Warning: Drone position not found in occupancy grid")
            return self.interest_values

        free_areas = np.argwhere(self.occupancy_grid == 200)  # Get the free areas

        min_x_y = drone_position[0]
        max_x_y = [min_x_y[0] + 5, min_x_y[1] + 5]

        # Reset interest values
        self.interest_values = np.zeros(4, dtype=int)

        # Iterate through free areas and calculate their relation to the drone
        for area in free_areas:
            if area[0] < min_x_y[0]:
                self.interest_values[0] += 1  # "front"
            elif area[0] > max_x_y[0]:
                self.interest_values[1] += 1  # "back"
            if area[1] < min_x_y[1]:
                self.interest_values[3] += 1  # "left"
            elif area[1] > max_x_y[1]:
                self.interest_values[2] += 1  # "right"

        return self.interest_values


if __name__ == "__main__":
    last_actions = np.zeros(100)
    model = SAC.load("/home/moritz_s/Desktop/Test_Series/01/M6_R6_O8_A3_TR1_T1_20250415_211929_schwere_Mazes_SAC_alt_2Hz_100LA/save-04.15.2025_21.19.29/final_model.zip")

    slam = SimpleSlam()

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache="./cache")
    with SyncCrazyflie(URI, cf=cf) as scf:
        # Arm the Crazyflie
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name="Position", period_in_ms=10)
        logconf.add_variable("stateEstimate.x", "float")
        logconf.add_variable("stateEstimate.y", "float")
        logconf.add_variable("stabilizer.yaw", "float")
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        # if not deck_attached_event.wait(timeout=5):
        #     print("No flow deck detected!")
        #     sys.exit(1)

        logconf.start()

        # take_off_simple(scf)

        with MotionCommander(scf) as motion_commander:
            with Multiranger(scf) as multiranger:
                keep_flying = True

                while keep_flying:
                    # Replace None values with a default distance (e.g., 5.0 meters)
                    front = multiranger.front if multiranger.front is not None else 2.1
                    back = multiranger.back if multiranger.back is not None else 2.1
                    left = multiranger.left if multiranger.left is not None else 2.1
                    right = multiranger.right if multiranger.right is not None else 2.1

                    # Updating SLAM
                    drone_pos = position_estimate
                    drone_yaw = position_estimate[2]
                    raycast_results = {
                        "front": front,
                        "back": back,
                        "left": left,
                        "right": right,
                    }
                    slam.update(drone_pos, drone_yaw, raycast_results)
                    slam._compute_interest_values()

                    # preparing observation
                    # X-Pos, Y-Pos, Raycast Readings, Interest Values, Last Actions
                    obs_list = [position_estimate[0], position_estimate[1], front, back, left, right]
                    obs_list.extend(slam.interest_values)
                    obs_list.extend(last_actions)
                    observation = np.array(obs_list)

                    # Flight Code
                    VELOCITY = 0.25
                    velocity_x = 0.0
                    velocity_y = 0.0

                    # Prediction
                    action, _ = model.predict(observation, deterministic=True)
                    velocity_x = action[0] * 0.25
                    velocity_y = action[1] * 0.25

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
