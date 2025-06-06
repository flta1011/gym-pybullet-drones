# -*- coding: utf-8 -*-
#
# ,---------,       ____  _ __
# |  ,-^-,  |      / __ )(_) /_______________ _____  ___
# | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
# Copyright (C) 2020 Bitcraze AB
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
This class is used to turn the power of the Crazyflie on and off via
a Crazyradio.
"""
import time

import cflib.crtp
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.radiodriver import RadioManager


class PowerSwitch:
    BOOTLOADER_CMD_ALLOFF = 0x01
    BOOTLOADER_CMD_SYSOFF = 0x02
    BOOTLOADER_CMD_SYSON = 0x03
    BOOTLOADER_CMD_RESET_INIT = 0xFF
    BOOTLOADER_CMD_RESET = 0xF0

    def __init__(self, uri):
        self.uri = uri
        uri_augmented = uri + "?safelink=0&autoping=0&ackfilter=0"
        self.link = cflib.crtp.get_link_driver(uri_augmented)
        # Switch to legacy mode, if uri options are not available or old Python backend is used
        if not self.link or self.link.get_name() == "radio":
            uri_parts = cflib.crtp.RadioDriver.parse_uri(uri)
            self.devid = uri_parts[0]
            self.channel = uri_parts[1]
            self.datarate = uri_parts[2]
            self.address = uri_parts[3]
            if self.link:
                self.link.close()
                self.link = None

    def platform_power_down(self):
        """Power down the platform, both NRF and STM MCUs.
        Same as turning off the platform with the power button."""
        self._send(self.BOOTLOADER_CMD_ALLOFF)

    def stm_power_down(self):
        """Power down the STM MCU, the NRF will still be powered and handle
        basic radio communication. The STM can be restarted again remotely.
        Note: the power to expansion decks is also turned off."""
        self._send(self.BOOTLOADER_CMD_SYSOFF)

    def stm_power_up(self):
        """Power up (boot) the STM MCU and decks."""
        self._send(self.BOOTLOADER_CMD_SYSON)

    def stm_power_cycle(self):
        """Restart the STM MCU by powering it off and on.
        Expansion decks will also be rebooted."""
        self.stm_power_down()
        time.sleep(1)
        self.stm_power_up()

    def reboot_to_fw(self):
        """Reboot the platform and start in firmware mode"""
        self._send(self.BOOTLOADER_CMD_RESET_INIT)
        self._send(self.BOOTLOADER_CMD_RESET, [1])

    def reboot_to_bootloader(self):
        """Reboot the platform and start the bootloader"""
        self._send(self.BOOTLOADER_CMD_RESET_INIT)
        self._send(self.BOOTLOADER_CMD_RESET, [0])

    def close(self):
        if self.link:
            self.link.close()

    def _send(self, cmd, data=[]):
        if not self.link:
            packet = [0xF3, 0xFE, cmd] + data

            cr = RadioManager.open(devid=self.devid)
            cr.set_channel(self.channel)
            cr.set_data_rate(self.datarate)
            cr.set_address(self.address)
            cr.set_arc(3)

            success = False
            for i in range(50):
                res = cr.send_packet(packet)
                if res and res.ack:
                    success = True
                    break

                time.sleep(0.01)

            cr.close()

            if not success:
                raise Exception("Failed to connect to Crazyflie at {}".format(self.uri))
        else:

            # send command (will be repeated until acked)
            pk = CRTPPacket(0xFF, [0xFE, cmd] + data)
            self.link.send_packet(pk)
            # wait up to 1s
            pk = self.link.receive_packet(0.1)
            if pk is None:
                raise Exception("Failed to connect to Crazyflie at {}".format(self.uri))
