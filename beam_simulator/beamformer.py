"""
Beamformer functions.
"""
import os
import sys
import aipy
import numpy as np

from numba import jit
from astropy.constants import c

import time

__version__ = '1.0'
__authors__ = ['Chris DiLullo', 'Jayce Dowell']
__all__ = ['generate_uniform_weights', 'generate_gaussian_weights',
           'calc_geometric_delays', 'beamform']

def generate_uniform_weights(station):
    """
    Return a weighting vector with uniform weights (w = 1.0 for all elements)
    for a given Station object.
    
    Inputs:
     * Station object

    Returns:
     * Array of size len(station.antennas)
    """

    w = np.ones( len(station.antennas) )
    return w

def generate_gaussian_weights(station, freq=60e6, azimuth=0.0, elevation=90.0, fwhm=5.0):
    """
    Return a weighting vector with Gaussian weights relative
    to the station geometric center for a custom beam with
    a given main lobe FWHM (deg) at a given frequency (Hz)
    and az/el pointing center (deg).

    See DiLullo, Taylor, and Dowell (2020) for a description of the method.

    Inputs:
     * station - Station object
     * freq - Frequency [Hz]
     * azimuth - Azimuth Eastward of North [deg]
     * elevation - Elevation above horizon [deg]
     * fwhm - FWHM of the main lobe [deg]

    Returns:
     * Array of size length len(station.antennas)
    """

    #Make sure the pointing center is valid.
    if azimuth < 0 or azimuth > 360:
        raise ValueError("Pointing center azimuth is out of the valid range [0, 360]")
    if elevation < 0 or elevation > 90:
        raise ValueError("Pointing center elevation is out of the valid range [0, 90]")

    #Create the arrays containing antenna positions, cable delays,
    #and cable attenuations.
    xyz = np.array([(a.x, a.y, a.z) for a in station.antennas]).T
    att = np.sqrt( np.array([a.cable.attenuation(freq) for a in station.antennas]) )

    #Compute the center of the array and reframe the station.
    center = np.mean(xyz, axis=1)

    xyz2 = xyz - np.array([ [center[0]], [center[1]], [0] ])

    #Compute the rotation matrix which describes the transformation from xyz to XYZ.
    rot = np.array([ [np.cos(azimuth*np.pi/180), -np.sin(azimuth*np.pi/180), 0.0],
                     [np.sin(azimuth*np.pi/180), np.cos(azimuth*np.pi/180), 0.0],
                     [0.0, 0.0, 1.0] ])

    #Transform from xyz to XYZ.
    XYZ = np.matmul(rot, xyz2)

    #Compute the necessary baselines in the perpendicular (X) and parallel (Y)
    #directions of the station for the given pointing center.
    d_X = (c.value / freq) / (fwhm * np.pi/180)
    d_Y = d_X / np.sin(elevation * np.pi/180)

    #Compute the standard deviation in both the X and Y directions
    #for a Gaussian whose full width at fifth maximum (FWFM) is d_X
    #and d_Y.
    sigma_X = d_X / (2.0 * np.sqrt(2.0 * np.log(5.0)) )
    sigma_Y = d_Y / (2.0 * np.sqrt(2.0 * np.log(5.0)) )

    #Compute the weights.
    w = att * np.exp( -(XYZ[0,:]**2 / (2.0*sigma_X**2) + XYZ[1,:]**2 / (2.0*sigma_Y**2)) )

    return w

def calc_geometric_delays(station, freq=60e6, azimuth=0.0, elevation=90.0):
    """
    Calculate the geometric delays between station elements
    for a given frequency (Hz) and az/el pointing center (deg).

    Inputs:
     * station - Station object
     * freq - Frequency [Hz]
     * azimuth - Azimuth Eastward of North [deg]
     * elevation - Elevation above horizon [deg]

    Returns:
     * Delays array of size length len(station.antennas)
    """

    #Make sure the pointing center is valid.
    if azimuth < 0 or azimuth > 360:
        raise ValueError("Pointing center azimuth is out of the valid range [0, 360]")
    if elevation < 0 or elevation > 90:
        raise ValueError("Pointing center elevation is out of the valid range [0, 90]")
    
    #Create the array that contains the positions of the station elements.
    xyz = np.array([(a.x, a.y, a.z) for a in station.antennas]).T

    #Compute the station center.
    center = np.mean(xyz, axis=1)

    #Build a unit vector that points in the direction of the pointing center.
    k = np.array([np.cos(elevation*np.pi/180)*np.sin(azimuth*np.pi/180),
                np.cos(elevation*np.pi/180)*np.cos(azimuth*np.pi/180),
                np.sin(elevation*np.pi/180)])

    #Compute the element poitions with respect to the center
    #and then compute the time delays relative to that pointing
    #direction in seconds.
    xyz2 = (xyz.T - center).T

    delays = np.dot(k, xyz2) / c.value

    #Get the cable delays.
    cbl = np.array([a.cable.delay(freq) for a in station.antennas])

    #Take cable delays into account.
    delays = cbl - delays

    return delays

@jit(nopython=True)
def _computeBeamformedSignal(freq, az, el, xyz, cbl, t, w, att, pol1, pol2, vLight):

    pwr1 = az*0.0
    pwr2 = az*0.0

    for i in range(az.shape[0]):
        for j in range(az.shape[1]):
            #Get the pixel coordinates.
            a, e = az[i,j], el[i,j]
            
            #Convert this to a pointing vector and compute
            #the physical delays across the array.
            k = np.array([np.cos(e)*np.sin(a),np.cos(e)*np.cos(a), np.sin(e)])
            t_p = cbl - np.dot(k, xyz) / vLight

            #Calculate the beamformed signal in this direction.
            sig = w*np.exp(-2j*np.pi*freq*(t_p - t)) / att

            #Sum and square the results.
            pwr1[i,j] = np.abs( np.sum(sig[pol1]) )**2
            pwr2[i,j] = np.abs( np.sum(sig[pol2]) )**2

    return pwr1, pwr2

def beamform(station, w, freq=60e6, azimuth=0.0, elevation=90.0, resolution=1.0, antGainFile=None, dB=False):
    """
    Given a weighting vector and beam_simulator.Station object,
    simulate the beam pattern on the sky for a given frequency
    and pointing.

    Inputs:
     * station - Station object
     * w - Weighting array
     * freq - Frequency [Hz]
     * azimuth - Azimuth Eastward of North [deg]
     * elevation - Elevation above horizon [deg]
     * resolution - Simulation resolution [deg]
     * antGainFile - Antenna gain file output by nec.combine_harmonic_fits (.npz)
     * dB - Convert final simulated power to dB

    Returns:
     * 2-D array of shape (360/resolution X 90/resolution)
    """

    #Make sure the pointing center is valid.
    if azimuth < 0 or azimuth > 360:
        raise ValueError("Pointing center azimuth is out of the valid range [0, 360]")
    if elevation < 0 or elevation > 90:
        raise ValueError("Pointing center elevation is out of the valid range [0, 90]")
    
    #Create the arrays containing antenna positions, cable delays,
    #and cable attenuations.
    xyz = np.array([(a.x, a.y, a.z) for a in station.antennas]).T
    cbl = np.array([a.cable.delay(freq) for a in station.antennas])
    att = np.sqrt( np.array([a.cable.attenuation(freq) for a in station.antennas]) )
    
    #Create the azimuth, elevation, and power arrays at the desired resolution.
    ires = int(round(1.0/min([1.0, resolution])))
    az = np.arange(0,360*ires+1,1) / ires
    el = np.arange(0,90*ires+1,1) / ires
    el, az = np.meshgrid(el,az)

    #Convert az and el to radians.
    az, el = az*np.pi/180.0, el*np.pi/180.0

    #Compute the delays across the array for the desired pointing center.
    t = calc_geometric_delays(station=station, freq=freq, azimuth=azimuth, elevation=elevation)

    #Figure out which elements correspond to each polarization and ignore bad antennas.
    pol1 = [i for i,a in enumerate(station.antennas) if a.status == 1 and a.pol == 1]
    pol2 = [i for i,a in enumerate(station.antennas) if a.status == 1 and a.pol == 2]
    w[[i for i, a in enumerate(station.antennas) if a.status == 0]] = 0.0

    pol1 = np.array(pol1)
    pol2 = np.array(pol2)

    #Beamform.
    print(f'Simulating beam pattern for a pointing at {azimuth:.2f} deg azimuth, {elevation:.2f} deg elevation at {freq/1e6:.2f} MHz')

    pwr1, pwr2 = _computeBeamformedSignal(freq=freq, az=az, el=el, xyz=xyz, cbl=cbl, t=t, w=w, att=att, pol1=pol1, pol2=pol2, vLight=c.value)

    #Multiply by the dipole gain pattern (this assumes pattern multiplication is valid).
    try:
        #Check to see if a file generated by the utility functions in nec.py is present.
        beam = np.load(antGainFile)
        deg = beam['deg']
        lmax = beam['lmax']
        
        #Get the polarization.
        coeffs1 = beam['coeffs1']
        coeffs2 = beam['coeffs2']
        
        #Read the coefficients and input them to AIPY to represent the gain pattern.
        for p, coeffs in zip([pwr1, pwr2], [coeffs1, coeffs2]):
            beamShapeDict = {}
            for j in range(deg+1):
                beamShapeDict[j] = np.squeeze(coeffs[-1-j,:])
            
            antGain = aipy.amp.BeamAlm(np.array([freq/1e9]), lmax=lmax, mmax=lmax, 
                                       deg=deg, nside=256, coeffs=beamShapeDict)
            
            antGain = antGain.response(aipy.coord.azalt2top(np.concatenate([[az.ravel()], [el.ravel()]])))
        
        
            #Multiply the power array by the antenna gain pattern.
            antGain = antGain.reshape(az.shape)
            antGain /= antGain.max()
            
            p *= antGain
        
    except:
        print('No antenna gain pattern given. The output power array only accounts for the geometry of the array!')

    #Return the beam pattern.
    pwr1 /= pwr1.max()
    pwr2 /= pwr2.max()
    pwr = np.stack((pwr1, pwr2), axis=0)
    
    if dB:
        return 10*np.log10(pwr)
    else:
        return pwr
