#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from beam_simulator import nec

class nec_tests(unittest.TestCase):
    """A unittest.testCase collection to test the 
    beam_simulator.nec module."""

    def test_AntennaPattern(self):
        """Test to see if nec.AntennaPattern creates 
        an isotropic pattern."""

        a = nec.AntennaPattern()
        ap = a.antenna_pat

        self.assertEqual(ap.sum(), ap.size)

if __name__ == '__main__':
    unittest.main()
    
