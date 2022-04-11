Beam Simulator
==============
[![Build and Test](https://github.com/cdilullo/beam_simulator/actions/workflows/main.yml/badge.svg)](https://github.com/cdilullo/beam_simulator/actions/workflows/main.yml) [![codecov](https://codecov.io/gh/cdilullo/beam_simulator/branch/master/graph/badge.svg?token=VHRPYN6Y98)](https://codecov.io/gh/cdilullo/beam_simulator)
 [![Documentation Status](https://readthedocs.org/projects/beam-simulator/badge/?version=latest)](https://beam-simulator.readthedocs.io/en/latest/?badge=latest)

[![Paper](https://img.shields.io/badge/arXiv-2112.13899-blue.svg)](https://arxiv.org/abs/2112.13899)

Description
-----------
This package contains tools for simulating the beam pattern of an arbitray phased array. It consists of 3 modules:
* nec.py - Tools for reading in an output file from a NEC4 simulation and representing the gain pattern of an individual antenna.
* station.py - Collection of classes used to build objects representing the different levels of an array.
* beamformer.py - Beamforming related functions for a given `<Station>` object.
* driftcurve.py - Collection of functions which can simulate the observed sky for a `<Station>` object and simulate its response.

Information about a station can be supplied in a text file which can be loaded in to generate a fully populated `<Station>` object.
A template for such text files is provided. LWA SSMIF's are also supported to quickly load in a full LWA station.

Requirements
------------
* python >= 3.6
* numpy >= 1.19.2
* scipy >= 1.5.4
* astropy >= 4.1
* matplotlib >= 3.3.2
* numba >= 0.51.2
* pygdsm >= 1.3.0
* tqdm >= 4.62.2
* ephem >= 4.1
* healpy >= 1.15.0
* lsl >= 2.0.2 (for LWA SSMIF compatability)

Example
-------
Below is an example showing how to:
1. Populate a `<Station>` object describing LWA-SV using its Station Static MIB Initialization File (SSMIF).
1. Represent the gain pattern of a LWA dipole.
1. Simulate the beam pattern for LWA-SV for a given pointing center and frequency.

NOTES:
* "lwasv-ssmif.txt" can be found [within the LWA Software Library](https://github.com/lwa-project/lsl/tree/master/lsl/data).
* The NEC4 output files for the LWA dipole can be found [here](http://fornax.phys.unm.edu/lwa/trac/browser/trunk/DipoleResponse)

### Setting up the Station.
```
from beam_simulator import station

lwasv = station.load_LWA("lwasv-ssmif.txt")
```

### Representing the LWA dipole gain pattern and fitting spherical harmonics to it as a function of frequency.
The LWA dipole gain pattern has been modeled using NEC4 at multiple frequencies. These can be loaded into 
Beam Simulator in order to build a model of the dipole gain pattern.
```
from beam_simulator import nec

#We need to read in all NEC4 output files for a series of frequencies.
freqs = [10, 20, 30, 40, 50, 60, 70, 80, 88] #MHz

p1 = [f'lwa1_xep_{freq}.out' for freq in freqs]
t1 = [f'lwa1_xet_{freq}.out' for freq in freqs]
p2 = [f'lwa1_yep_{freq}.out' for freq in freqs]
t2 = [f'lwa1_yet_{freq}.out' for freq in freqs]

#Fit the spherical harmonic decompisition as a polynomial in frequency.
nec.fit_antenna_response(freqs, p1, t1, p2, t2, lmax=12)  
```

This will create a file named "beam_cofficients.npz" which can be used by the `beamformer` module.

### Simulating the beam pattern of LWA-SV.
```
from beam_simulator import beamformer

#Generate the weighting vector for the station.
w = beamformer.generate_uniform_weights(lwasv) #All antennas have the same weighting of 1.0

#Simulate the beam for a pointing center of az = 180 deg, el = 75 deg at 74 MHz with 1 degree resolution.
pwr = beamformer.beamform(lwasv, w, freq=74e6, azimuth=180.0, elevation=75.0, resolution=1.0,
                          antGainFile='beam_coefficients.npz', dB=False)
```
