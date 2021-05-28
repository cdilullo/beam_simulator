#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np

from beam_simulator import station, beamformer

class beamformer_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the
    beamformer module."""

    def test_generate_uniform_weights(self):
        """Check the beamformer.generate_uniform_weights function."""
        
        s = station.load_station('../beam_simulator/station_template.txt')
        w = beamformer.generate_uniform_weights(s)

        self.assertEqual(len(s.antennas), w.sum())

    def test_generate_gaussian_weights(self):
        """Check to see if the beamformer.generate_gaussian_weights
        function runs."""

        s = station.load_station('../beam_simulator/station_template.txt')
        w = beamformer.generate_gaussian_weights(s)

        self.assertEqual(w.size, len(s.antennas))

    def test_calc_geometric_delays(self):
        """Check to see if the beamformer.calc_geometric_delays 
        function runs."""

        s = station.load_station('../beam_simulator/station_template.txt')
        d = beamformer.calc_geometric_delays(s)

        self.assertEqual(d.size, len(s.antennas))

    def test_beamform(self):
        """Test to see if the beamformer.beamform function
        runs and produces a nonzero power pattern."""

        s = station.load_station('../beam_simulator/station_template.txt')
        w = beamformer.generate_uniform_weights(s)

        pwr = beamformer.beamform(station=s, w=w)

        self.assertNotEqual(pwr.sum(), 0)
        
if __name__ == '__main__':
    unittest.main()
