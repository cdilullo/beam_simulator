"""
Beam object which stores beam pattern properties.
"""

import numpy as np
import scipy.optimize as scop

__version__ = '1.0'
__authors__ = ['Chris DiLullo', 'Jayce Dowell']
__all__ = []

def _Gauss2D(az_el, az0, el0, sigma_az, sigma_el):
    az, el = az_el #az_el should be a tuple
    return np.exp( -((np.cos(el*np.pi/180.0)*(az-az0))**2/(2.0*sigma_az**2) + (el-el0)**2/(2.0*sigma_el**2)) )

class Beam(object):
    """
    Object to store a simulated beam pattern.
    """

    def __init__(self, power=None, freq=60e6, resolution=1.0, 
                 azimuth=None, elevation=None, dB=False):

        self.dB = dB
        self.freq = freq
        self.resolution = resolution
        
        # Make sure the pointing center is valid.
        if azimuth < 0 or azimuth > 360:
            raise ValueError("Pointing center azimuth is out of the valid range [0, 360]")
        if elevation < 0 or elevation > 90:
            raise ValueError("Pointing center elevation is out of the valid range [0, 90]")

        self.azimuth = azimuth
        self.elevation = elevation

        # Set the power array.
        if power is not None:
            self._power = power
        else:
            ires = int(round(1.0/min([1.0, resolution])))
            az = np.arange(0,360*ires+1,1) / ires
            el = np.arange(0,90*ires+1,1) / ires
            self._power = np.ones((az.size, el.size), dtype=np.float64)

    def __repr__(self) -> str:
        n = self.__class__.__name__
        a = []
        for attr in ['power', 'freq', 'resolution', 'azimuth', 'elevation', 'dB']:
            a.append((attr, getattr(self, attr)))

        output = f'<{n}:'
        first = True
        for key, value in a:
            output += '%s %s=%s' % (('' if first else ','), key, value)
            first = False
        output += '>'
        
        return fill(output, subsequent_indent='         ')
    
    @property
    def power(self):
        """
        Return the array containing the beam power pattern.
        """

        return self._power
    
    def fit_mainlobe(self, return_sigmas=False):
        """
        Fit a 2D Gaussian to the main lobe of the beam pattern.
        """
        
        ires = int(round(1.0/min([1.0, self.resolution])))
        az = np.arange(0,360*ires+1,1) / ires
        el = np.arange(0,90*ires+1,1) / ires
        el, az = np.meshgrid(el, az)

        p, _ = scop.curve_fit(_Gauss2D, (az.flatten(), el.flatten()), self._power, 
                              p0=[self.azimuth, self.elevation, 1.0, 1.0])
        
        mainlobe = _Gauss2D((az,el), p[0], p[1], p[2], p[3])

        if return_sigmas:
            return mainlobe, p[2], p[3]
        else:
            return mainlobe