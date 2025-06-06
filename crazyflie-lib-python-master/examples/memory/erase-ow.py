# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
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
Warning: you will have to write new data to the memory to make it
usable again. Use with caution.

Simple example that connects to the first Crazyflie found, looks for
EEPROM memories and erases its contents.
"""
import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

# Only output errors from the logging framework
logging.basicConfig(level=logging.INFO)


class EEPROMExample:
    """
    Simple example listing the EEPROMs found and erases its contents.
    """

    def __init__(self, link_uri):
        """Initialize and run the example with the specified link_uri"""

        # Create a Crazyflie object without specifying any cache dirs
        self._cf = Crazyflie()

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print("Connecting to %s" % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print("Connected to %s" % link_uri)

        mems = self._cf.mem.get_mems(MemoryElement.TYPE_1W)
        print("Found {} 1-wire memories".format(len(mems)))
        if len(mems) > 0:
            print("Erasing memory {}".format(mems[0].id))
            mems[0].erase(self._data_written)

    def _data_written(self, mem, addr):
        print("Data written, reading back...")
        mem.update(self._data_updated)

    def _data_updated(self, mem):
        print("Updated id={}".format(mem.id))
        print("\tType      : {}".format(mem.type))
        print("\tSize      : {}".format(mem.size))
        print("\tValid     : {}".format(mem.valid))
        print("\tName      : {}".format(mem.name))
        print("\tVID       : 0x{:02X}".format(mem.vid))
        print("\tPID       : 0x{:02X}".format(mem.pid))
        print("\tPins      : 0x{:02X}".format(mem.pins))
        print("\tElements  : ")

        for key in mem.elements:
            print("\t\t{}={}".format(key, mem.elements[key]))

        self._cf.close_link()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print("Error when logging %s: %s" % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        print("[%d][%s]: %s" % (timestamp, logconf.name, data))

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print("Connection to %s failed: %s" % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print("Connection to %s lost: %s" % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print("Disconnected from %s" % link_uri)
        self.is_connected = False


if __name__ == "__main__":
    input("Warning, this will erase EEPROM memory, press enter to continue...")

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    le = EEPROMExample(uri)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    try:
        while le.is_connected:
            time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(1)
