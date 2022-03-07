"""
Driftcurve related classes and functions.
"""

import tqdm
import ephem
import astropy
import numpy as np
import healpy as hp

from datetime import datetime, timedelta
from pygdsm import pygsm, pygsm2016, lfsm    

def generate_observed_sky(station, date=None, skymap='GSM2008', freq=60e6, return_observer=False, plot=False):
    """
    Generate a model of the observed sky for a Station object at a given time and frequency.

    Inputs:
     * station - Station object
     * date - UTC date as either a datetime.datetime object or Unix timestamp. If None, defaults to the current time
     * skymap - Sky map to use. Valid options are 'GSM2008', 'GSM2016', and 'LFSM'
     * freq - Frequency in Hz
     * return_observer - Returns the Observer object along with the observed sky array
     * plot - Plot the observed sky

    Returns:
     * Observer object (optional)
     * Array representing the temperature of the sky in HEALPix format, with points under the horizon masked
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
    observer.lat = station.latitude
    observer.lon = station.longitude
    observer.date = date

    #Get the observed sky.
    freq_MHz = freq / 1e6
    sky = observer.generate(freq_MHz)

    if plot:
        #Plot the observed portion sky model on a Molleweide projection.
        #Note: The 'logged' option sets the scale to log base 2.
        observer.view_observed_gsm(logged=True, show=True)

    if return_observer:
        return observer, sky
    else:
        return sky

def generate_driftcurve(station, beam, start=None, stop=None, step=30, skymap='GSM2008', freq=60e6):
    """
    Simulate a driftcurve for a given station beam pointing for a specific time range at a frequency.

    Inputs:
     * station - Station object
     * beam - Beam power pattern output by the beamformer.beamform function
     * start - UTC start time as either a datetime.datetime object or Unix timestamp
     * stop - UTC stop time as either a datetime.datetime object or Unix timestamp
     * step - Time step in minutes at which to simulate the driftcurve between start and stop
     * skymap - Sky map to use. Valid options are 'GSM2008', 'GSM2016', or 'LFSM'
     * freq - Frequency in Hz at which to generate the sky model. Should match that of the supplied beam pattern simulation

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
        if times[-1] == stop:
            break
        i += 1

    #Set up function to query the beam pattern.
    ires = beam.shape[1] // 360
    def Beam_Pattern(az, alt, pol, beam=beam, ires=ires):
        iAz = (np.round(az*ires)).astype(np.int32)
        iAlt = (np.round(alt*ires)).astype(np.int32)

        return beam[pol, iAz, iAlt] 

    #Simulate the observed sky at each time.
    drift = np.zeros((beam.shape[0],len(times)), dtype=np.float64)
    for i in range(drift.shape[0]):
        print(f'Working on pol {i}')
        for j,t in tqdm.tqdm(enumerate(times)):
            observer, sky = generate_observed_sky(station=station, date=t, skymap=skymap, freq=freq, return_observer=True)

            #Convert sky HEALPix coords from Galactic to Equatorial in degrees.
            b, l = hp.pix2ang(observer._n_side, observer._pix0, lonlat=True)
            b, l = b[observer._mask], l[observer._mask]
            rot = hp.rotator.Rotator(coord=['G','C'])
            dec, ra = rot(b, l)

            #Convert Equatorial coordinates to local.
            az, alt = [], []
            for r, d in zip(ra, dec):
                b = ephem.FixedBody()
                b._ra = r
                b._dec = d

                b.compute(observer)
                az.append(b.az)
                alt.append(b.alt)
            
            az = np.array(az)
            alt = np.array(alt)

            #Query the beam pattern at the visible local coordinates. 
            gain = Beam_Pattern(az, alt, pol=i)

            #Multiply the observed sky by the given beam pattern.
            sky *= gain

            #Sum to find total observed power.
            drift[i,j] = np.sum(sky) / gain.sum()

    #Convert the time to LST.
    LSTs = np.zeros(len(times), dtype=np.float64)
    for i in range(len(times)):
        LSTs[i] = astropy.time.Time(times[i], location=(station.longitude, station.latitude)).sidereal_time('apparent').value

    return LSTs, drift
