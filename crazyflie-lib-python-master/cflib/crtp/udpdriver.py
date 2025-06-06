#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2011-2013 Bitcraze AB
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
""" CRTP UDP Driver. Work either with the UDP server or with an UDP device
See udpserver.py for the protocol"""
import queue
import re
import socket
import struct
from urllib.parse import urlparse

from .crtpdriver import CRTPDriver
from .crtpstack import CRTPPacket
from .exceptions import WrongUriType

__author__ = "Bitcraze AB"
__all__ = ["UdpDriver"]


class UdpDriver(CRTPDriver):

    def __init__(self):
        None

    def connect(self, uri, linkQualityCallback, linkErrorCallback):
        if not re.search("^udp://", uri):
            raise WrongUriType("Not an UDP URI")

        parse = urlparse(uri)

        self.queue = queue.Queue()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (parse.hostname, parse.port)
        self.socket.connect(self.addr)

        self.socket.sendto("\xFF\x01\x01\x01".encode(), self.addr)

    def receive_packet(self, time=0):
        data, addr = self.socket.recvfrom(1024)

        if data:
            data = struct.unpack("B" * (len(data) - 1), data[0 : len(data) - 1])
            pk = CRTPPacket()
            pk.port = data[0]
            pk.data = data[1:]
            return pk

        try:
            if time == 0:
                return self.rxqueue.get(False)
            elif time < 0:
                while True:
                    return self.rxqueue.get(True, 10)
            else:
                return self.rxqueue.get(True, time)
        except queue.Empty:
            return None

    def send_packet(self, pk):
        raw = (pk.port,) + struct.unpack("B" * len(pk.data), pk.data)

        cksum = 0
        for i in raw:
            cksum += i

        cksum %= 256

        data = "".join(chr(v) for v in (raw + (cksum,)))

        # print tuple(data)
        self.socket.sendto(data.encode(), self.addr)

    def close(self):
        # Remove this from the server clients list
        self.socket.sendto("\xFF\x01\x02\x02".encode(), self.addr)

    def get_name(self):
        return "udp"

    def scan_interface(self, address):
        return []
