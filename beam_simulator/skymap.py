"""
Functions related to simulating the sky above a specific observer and simulating driftcurves and spectra for a given beam pointing.
"""

import ephem
import astropy
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from . import beamformer

from tqdm import tqdm
from datetime import datetime, timedelta
from pygdsm import pygsm, pygsm2016, lfsm    

def generate_GSM_observer(station, freq=60e6, date=None, skymap='GSM2008', plot=False):
    """
    Generate a model of the observed sky for a Station object at a given time and frequency.

    Inputs:
     * station - Station object
     * date - UTC date as either a datetime.datetime object or Unix timestamp. If None, defaults to the current time
     * skymap - Sky map to use. Valid options are 'GSM2008', 'GSM2016', and 'LFSM'
     * freq - Frequency in Hz
     * plot - Plot the observed sky on an orthographic projection

    Returns:
     * Observer object containing observed sky information
    """
    
    #Check to make sure a valid sky map was queried.
    try:
        assert skymap in {'GSM2008', 'GSM2016', 'LFSM'}
    except AssertionError:
        raise RuntimeError("Unknown sky map selected. Please choose 'GSM2008', 'GSM2016', or 'LFSM'")

    #Set up the observer information with the right GSM parameters.
    if skymap == 'GSM2008':
        observer = pygsm.GSMObserver()
    elif skymap == 'GSM2016':
        observer = pygsm2016.GSMObserver2016()
        observer.gsm.data_unit = 'TRJ'
    else:
        observer = lfsm.LFSMObserver()

    #Date setup.
    if date is None:
        date = datetime.utcnow()
    elif not isinstance(date, datetime):
        date = datetime.utcfromtimestamp(date)

    #Fill in the relevant info.
    observer.lat = station.latitude*(np.pi/180.0)
    observer.lon = station.longitude*(np.pi/180.0)
    observer.date = date

    #Generate the observed sky.
    freq_MHz = freq / 1e6
    observer.generate(freq_MHz)

    if plot:
        #Plot the observed portion sky model on a Molleweide projection.
        #Note: The 'logged' option sets the scale to log base 2.
        title = f'The Sky as seen from {station.name} at {observer.date} UTC'
        observer.view(logged=True, show=True, title=title)

    return observer

def generate_driftcurve(station, beam, freq=60e6, start=None, stop=None, step=30, skymap='GSM2008', verbose=True, plot=False):
    """
    Simulate a driftcurve for a given station beam pointing for a specific time range at a frequency.

    Inputs:
     * station - Station object
     * beam - Beam power pattern output by the beamformer.beamform function
     * freq - Frequency in Hz at which to generate the sky model. Should match that of the supplied beam pattern simulation
     * start - UTC start time as either a datetime.datetime object or Unix timestamp
     * stop - UTC stop time as either a datetime.datetime object or Unix timestamp
     * step - Time step in minutes at which to simulate the driftcurve between start and stop
     * skymap - Sky map to use. Valid options are 'GSM2008', 'GSM2016', or 'LFSM'
     * verbose - Show progress bar
     * plot - Plot the driftcurve

    Returns:
     * Two 1-D arrays containing the local sidreal times (LSTs) and simulated temperatures of the simulation

     .. note::
      If start is None, default is to begin at the current time.
      If stop is None, default is to end at 1 day after the current time.
    """

    #Set up the times.
    if start is None:
        start = datetime.utcnow()
    elif not isinstance(start, datetime):
        start = datetime.utcfromtimestamp(start)
    
    if stop is None:
        stop = start + timedelta(days=1)
    elif not isinstance(stop, datetime):
        stop = datetime.utcfromtimestamp(stop)

    step = timedelta(minutes=step)

    times = []
    i = 0
    while True:
        times.append(start + i*step)
        if times[-1] >= stop:
            times[-1] = stop
            break
        i += 1

    #Set up the function to query the beam pattern.
    ires = beam.shape[1] // 360
    def Beam_Pattern(az, alt, pol, beam=beam, ires=ires):
        iAz = (np.round(az*ires)).astype(np.int32)
        iAlt = (np.round(alt*ires)).astype(np.int32)

        return beam[pol, iAz, iAlt] 

    #Simulate the observed sky at each time.
    LSTs = np.zeros(len(times), dtype=np.float64)
    drift = np.zeros((beam.shape[0],len(times)), dtype=np.float64)
    for i,t in enumerate(tqdm(times)) if verbose else enumerate(times):
        #Generate the observed sky at the given time.
        observer = generate_GSM_observer(station=station, date=t, skymap=skymap, freq=freq)

        sky = observer.observed_sky.data
        mask = observer.observed_sky.mask
        sky = sky[~mask]

        #Equatorial coordinates of each observed HEALPix pixel in degrees.
        ra, dec = observer._observed_ra[~mask], observer._observed_dec[~mask]

        ra *= (np.pi/180.0)
        dec *= (np.pi/180.0)

        az, alt = [], []
        for r,d in zip(ra, dec):
            b = ephem.FixedBody()
            b._ra = r
            b._dec = d

            b.compute(observer)
            az.append(b.az)
            alt.append(b.alt)

        az = np.array(az)*(180.0/np.pi)
        alt = np.array(alt)*(180/np.pi)

        #Query the beam pattern at the visible local coordinates
        # for each polarization.
        for j in range(drift.shape[0]):
            gain = Beam_Pattern(az, alt, pol=j)
        
            #Multiply the observed sky by the given beam pattern.
            beam_T = sky * gain

            #Sum to find total observed power.
            drift[j,i] = np.sum(beam_T) / gain.sum()

        #Convert the time to LST.
        LSTs[i] = astropy.time.Time(t, location=(station.longitude, station.latitude)).sidereal_time('apparent').value

    #Plot the driftcurve, if requested.
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.set_title(f'{station.name} Driftcurve at {freq/1e6} MHz Using {skymap}', fontsize=14)
        ax.plot(LSTs, drift[0,:], 'o', label='Pol 1')
        ax.plot(LSTs, drift[1,:], 'o', label='Pol 2')
        ax.set_xlabel('Local Sidereal Time [h]', fontsize=12)
        ax.set_ylabel('Temperature [K]', fontsize=12)
        ax.legend(loc=0)
        ax.tick_params(which='both', direction='in', length=8, labelsize=12)

        plt.show()

    return LSTs, drift


def generate_spectrum(station, weights, freqs, azimuth=90.0, elevation=90.0, resolution=1.0, ant_gain_file=None, date=None, skymap='GSM2008', verbose=True, plot=False):
    """
    Simulate a spectrum across a set of frequencies for a given station beam pointing and time.

    Inputs:
     * station - Station obect
     * weights - weighting array for the station elements
     * freqs - Frequencies in Hz
     * azimuth - Azimuth East of North in degrees
     * elevation - Elevation above the horizon in degrees
     * resolution - Resolution of the beam simulations in degrees
     * ant_gain_file - .npz file which stores the antenna gain pattern fit across frequency
     * date - UTC date/time as either a datetime.datetime object or a Unix timestamp
     * skymap - Sky map to use. Valid options are 'GSM2008', 'GSM2016', or 'LFSM'
     * verbose - Show progress bar
     * plot - Plot the final spectra
    """

    #Set up the observer information.
    if date is None:
        date = datetime.utcnow()
    elif not isinstance(date, datetime):
        date = datetime.utcfromtimestamp(date)

    #Set up the function to query the beam pattern.
    ires = int(round(1.0/min([1.0, resolution])))
    def Beam_Pattern(az, alt, pol, beam, ires=ires):
        iAz = (np.round(az*ires)).astype(np.int32)
        iAlt = (np.round(alt*ires)).astype(np.int32)

        return beam[pol, iAz, iAlt] 

    #Generate the beam at each desired frequency and combine it with the sky.
    spec = np.zeros((2, freqs.size), dtype=np.float64)
    for j, freq in enumerate(tqdm(freqs)) if verbose else enumerate(freqs):
        beam = beamformer.beamform(station=station, weights=weights, freq=freq, azimuth=azimuth, 
                elevation=elevation, resolution=resolution, ant_gain_file=ant_gain_file, verbose=False)

        observer = generate_GSM_observer(station=station, freq=freq, date=date, skymap=skymap)

        sky, mask = observer.observed_sky.data, observer.observed_sky.mask
        sky = sky[~mask]

        #Equatorial coordinates of each observed HEALPix pixel in degrees.
        ra, dec = observer._observed_ra[~mask], observer._observed_dec[~mask]

        ra *= (np.pi/180.0)
        dec *= (np.pi/180.0)

        az, alt = [], []
        for r,d in zip(ra, dec):
            b = ephem.FixedBody()
            b._ra = r
            b._dec = d

            b.compute(observer)
            az.append(b.az)
            alt.append(b.alt)

        az = np.array(az)*(180.0/np.pi)
        alt = np.array(alt)*(180/np.pi)

        for i in range(spec.shape[0]): 
            gain = Beam_Pattern(az, alt, pol=i, beam=beam)
        
            #Multiply the observed sky by the given beam pattern.
            beam_T = sky * gain

            #Sum to find total observed power.
            spec[i,j] = np.sum(beam_T) / gain.sum()

    #Plot the spectrum, if requested.
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.set_title(f'{station.name} Spectrum at {date:%Y/%m/%d %H:%M:%S} UTC for Beam Pointing ({azimuth} deg Az, {elevation} deg El) Using {skymap}', fontsize=14)
        ax.plot(freqs/1e6, spec[0,:], '-', label='Pol 1')
        ax.plot(freqs/1e6, spec[1,:], '-', label='Pol 2')
        ax.set_xlabel('Frequency [MHz]', fontsize=12)
        ax.set_ylabel('Temperature [K]', fontsize=12)
        ax.legend(loc=0)
        ax.tick_params(which='both', direction='in', length=8, labelsize=12)

        plt.show()

    return spec
