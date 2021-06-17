#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

from beam_simulator import station

class station_test(unittest.TestCase):
    """A collection of unit tests for the station module"""

    def test_station(self):
        """Test loading a Station object"""
        try:
            from lsl.common.paths import DATA as dataPath
            _ssmifsv = os.path.join(dataPath, 'lwasv-ssmif.txt')

            lwasv = station.load_LWA(_ssmifsv)
            self.assertTrue(isinstance(lwasv, station.Station))

        except ImportError:
            s = station.load_station('../beam_simulator/station_template.txt')
            self.assertTrue(isinstance(s, station.Station))

    def test_antenna(self):
        """Test the Antenna object of a Station object"""
        try:
            from lsl.common.paths import DATA as dataPath
            _ssmifsv = os.path.join(dataPath, 'lwasv-ssmif.txt')

            lwasv = station.load_LWA(_ssmifsv)
            ant = lwasv.antennas[0]
            self.assertTrue(isinstance(ant, station.Antenna))

        except ImportError:
            s = station.load_station('../beam_simulator/station_template.txt')
            ant = s.antennas[0]
            self.assertTrue(isinstance(ant, station.Antenna))

    def test_cable(self):
        """Test the Cable object of an Antenna object"""
        try:
            from lsl.common.paths import DATA as dataPath
            _ssmifsv = os.path.join(dataPath, 'lwasv-ssmif.txt')

            lwasv = station.load_LWA(_ssmifsv)
            ant = lwasv.antennas[0]
            cable = ant.cable
            self.assertTrue(isinstance(cable, station.Cable))
        
        except ImportError:
            s = station.load_station('../beam_simulator/station_template.txt')
            ant = s.antennas[0]
            cable = ant.cable
            self.assertTrue(isinstance(cable, station.Cable))


if __name__ == '__main__':
    unittest.main()

