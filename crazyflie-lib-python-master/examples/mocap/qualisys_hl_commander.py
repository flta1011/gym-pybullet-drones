# -*- coding: utf-8 -*-
#
# ,---------,       ____  _ __
# |  ,-^-,  |      / __ )(_) /_______________ _____  ___
# | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
# Copyright (C) 2019 Bitcraze AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, in version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Example of how to connect to a Qualisys QTM system and feed the position to a
Crazyflie. It uses the high level commander to upload a trajectory to fly a
figure 8.

Set the uri to the radio settings of the Crazyflie and modify the
rigid_body_name to match the name of the Crazyflie in QTM.
"""
import asyncio
import math
import time
import xml.etree.cElementTree as ET
from threading import Thread

import qtm_rt
from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

# The name of the rigid body in QTM that represents the Crazyflie
rigid_body_name = "cf"

# True: send position and orientation; False: send position only
send_full_pose = True

# When using full pose, the estimator can be sensitive to noise in the orientation data when yaw is close to +/- 90
# degrees. If this is a problem, increase orientation_std_dev a bit. The default value in the firmware is 4.5e-3.
orientation_std_dev = 8.0e-3

# The trajectory to fly
# See https://github.com/whoenig/uav_trajectories for a tool to generate
# trajectories

# Duration,x^0,x^1,x^2,x^3,x^4,x^5,x^6,x^7,y^0,y^1,y^2,y^3,y^4,y^5,y^6,y^7,z^0,z^1,z^2,z^3,z^4,z^5,z^6,z^7,yaw^0,yaw^1,yaw^2,yaw^3,yaw^4,yaw^5,yaw^6,yaw^7
figure8 = [
    [
        1.050000,
        0.000000,
        -0.000000,
        0.000000,
        -0.000000,
        0.830443,
        -0.276140,
        -0.384219,
        0.180493,
        -0.000000,
        0.000000,
        -0.000000,
        0.000000,
        -1.356107,
        0.688430,
        0.587426,
        -0.329106,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.710000,
        0.396058,
        0.918033,
        0.128965,
        -0.773546,
        0.339704,
        0.034310,
        -0.026417,
        -0.030049,
        -0.445604,
        -0.684403,
        0.888433,
        1.493630,
        -1.361618,
        -0.139316,
        0.158875,
        0.095799,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.620000,
        0.922409,
        0.405715,
        -0.582968,
        -0.092188,
        -0.114670,
        0.101046,
        0.075834,
        -0.037926,
        -0.291165,
        0.967514,
        0.421451,
        -1.086348,
        0.545211,
        0.030109,
        -0.050046,
        -0.068177,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.700000,
        0.923174,
        -0.431533,
        -0.682975,
        0.177173,
        0.319468,
        -0.043852,
        -0.111269,
        0.023166,
        0.289869,
        0.724722,
        -0.512011,
        -0.209623,
        -0.218710,
        0.108797,
        0.128756,
        -0.055461,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.560000,
        0.405364,
        -0.834716,
        0.158939,
        0.288175,
        -0.373738,
        -0.054995,
        0.036090,
        0.078627,
        0.450742,
        -0.385534,
        -0.954089,
        0.128288,
        0.442620,
        0.055630,
        -0.060142,
        -0.076163,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.560000,
        0.001062,
        -0.646270,
        -0.012560,
        -0.324065,
        0.125327,
        0.119738,
        0.034567,
        -0.063130,
        0.001593,
        -1.031457,
        0.015159,
        0.820816,
        -0.152665,
        -0.130729,
        -0.045679,
        0.080444,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.700000,
        -0.402804,
        -0.820508,
        -0.132914,
        0.236278,
        0.235164,
        -0.053551,
        -0.088687,
        0.031253,
        -0.449354,
        -0.411507,
        0.902946,
        0.185335,
        -0.239125,
        -0.041696,
        0.016857,
        0.016709,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.620000,
        -0.921641,
        -0.464596,
        0.661875,
        0.286582,
        -0.228921,
        -0.051987,
        0.004669,
        0.038463,
        -0.292459,
        0.777682,
        0.565788,
        -0.432472,
        -0.060568,
        -0.082048,
        -0.009439,
        0.041158,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        0.710000,
        -0.923935,
        0.447832,
        0.627381,
        -0.259808,
        -0.042325,
        -0.032258,
        0.001420,
        0.005294,
        0.288570,
        0.873350,
        -0.515586,
        -0.730207,
        -0.026023,
        0.288755,
        0.215678,
        -0.148061,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
    [
        1.053185,
        -0.398611,
        0.850510,
        -0.144007,
        -0.485368,
        -0.079781,
        0.176330,
        0.234482,
        -0.153567,
        0.447039,
        -0.532729,
        -0.855023,
        0.878509,
        0.775168,
        -0.391051,
        -0.713519,
        0.391628,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],  # noqa
]


class QtmWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while self._stay_open:
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        qtm_instance = await self._discover()
        host = qtm_instance.host
        print("Connecting to QTM on " + host)
        self.connection = await qtm_rt.connect(host)

        params = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text.strip() for index, label in enumerate(xml.findall("*/Body/Name"))]

        await self.connection.stream_frames(components=["6D"], on_packet=self._on_packet)

    async def _discover(self):
        async for qtm_instance in qtm_rt.Discover("0.0.0.0"):
            return qtm_instance

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        if self.body_name not in self.qtm_6DoF_labels:
            print("Body " + self.body_name + " not found.")
        else:
            index = self.qtm_6DoF_labels.index(self.body_name)
            temp_cf_pos = bodies[index]
            x = temp_cf_pos[0][0] / 1000
            y = temp_cf_pos[0][1] / 1000
            z = temp_cf_pos[0][2] / 1000

            r = temp_cf_pos[1].matrix
            rot = [
                [r[0], r[3], r[6]],
                [r[1], r[4], r[7]],
                [r[2], r[5], r[8]],
            ]

            if self.on_pose:
                # Make sure we got a position
                if math.isnan(x):
                    return

                self.on_pose([x, y, z, rot])

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()


def wait_for_position_estimator(scf):
    print("Waiting for estimator to find position...")

    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data["kalman.varPX"])
            var_x_history.pop(0)
            var_y_history.append(data["kalman.varPY"])
            var_y_history.pop(0)
            var_z_history.append(data["kalman.varPZ"])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (max_y - min_y) < threshold and (max_z - min_z) < threshold:
                break


def _sqrt(a):
    """
    There might be rounding errors making 'a' slightly negative.
    Make sure we don't throw an exception.
    """
    if a < 0.0:
        return 0.0
    return math.sqrt(a)


def send_extpose_rot_matrix(cf, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    quat = Rotation.from_matrix(rot).as_quat()

    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat[0], quat[1], quat[2], quat[3])
    else:
        cf.extpos.send_extpos(x, y, z)


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")

    # time.sleep(1)
    wait_for_position_estimator(cf)


def adjust_orientation_sensitivity(cf):
    cf.param.set_value("locSrv.extQuatStdDev", orientation_std_dev)


def activate_kalman_estimator(cf):
    cf.param.set_value("stabilizer.estimator", "2")

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value("locSrv.extQuatStdDev", 0.06)


def activate_mellinger_controller(cf):
    cf.param.set_value("stabilizer.controller", "2")


def upload_trajectory(cf, trajectory_id, trajectory):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    trajectory_mem.trajectory = []

    total_duration = 0
    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
        total_duration += duration

    trajectory_mem.write_data_sync()
    cf.high_level_commander.define_trajectory(trajectory_id, 0, len(trajectory_mem.trajectory))
    return total_duration


def run_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3.0)
    relative = True
    commander.start_trajectory(trajectory_id, 1.0, relative)
    time.sleep(duration)
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()


if __name__ == "__main__":
    cflib.crtp.init_drivers()

    # Connect to QTM
    qtm_wrapper = QtmWrapper(rigid_body_name)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        trajectory_id = 1

        # Set up a callback to handle data from QTM
        qtm_wrapper.on_pose = lambda pose: send_extpose_rot_matrix(cf, pose[0], pose[1], pose[2], pose[3])

        adjust_orientation_sensitivity(cf)
        activate_kalman_estimator(cf)
        # activate_mellinger_controller(cf)
        duration = upload_trajectory(cf, trajectory_id, figure8)
        print("The sequence is {:.1f} seconds long".format(duration))
        reset_estimator(cf)

        # Arm the Crazyflie
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        run_sequence(cf, trajectory_id, duration)

    qtm_wrapper.close()
