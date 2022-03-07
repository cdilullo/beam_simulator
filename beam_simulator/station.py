"""
Collection of classes that represent the various levels of an array.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from textwrap import fill
from scipy.interpolate import interp1d
from astropy.constants import c, mu0, eps0

__version__ = '1.0'
__authors__ = ['Chris DiLullo', 'Jayce Dowell']
__all__     = ['Station', 'Antenna', 'Cable', 'LMR200', 'LMR400',
               'load_station', 'load_LWA']

class Station(object):
    """
    Object to store information about an antenna array. Stores:
     * Station Name
     * Antenna Objects
     * Array Latitude (float degrees)
     * Array Longitude (float degrees)
    """

    def __init__(self, name=None, latitude=None, longitude=None, antennas=None):

        self.name = name
        self.latitude = latitude
        self.longitude = longitude

        if antennas is None:
            self._antennas = []
        else:
            self._antennas = list(antennas)

    def __repr__(self):
        n = self.__class__.__name__
        a = [(attr, getattr(self, attr)) for attr in ('name', 'antennas')]
        
        if a[1][1] != []:
            a[1] = ('antennas', ['...'])
        
        output = '<%s:' % n
        first = True
        for key, value in a:
            output += '%s %s=%s' % (('' if first else ','), key, value)
            first = False
        output += '>'
        return fill(output)
    
    @property
    def antennas(self):
        """
        Return a list of `station.Antenna` objects that the `station.Station` object is comprised of.
        """
        return self._antennas
    
    def plot_antennas(self):
        """
        Plot the locations of the antennas.
        """
        f, ax = plt.subplots(1,1)
        ax.set_title('%s' % self.name, fontsize='x-large')
        ax.set_xlabel('X [m]', fontsize='large')
        ax.set_ylabel('Y [m]', fontsize='large')
        ax.tick_params(direction='in',size=5)
        ax.plot([a.x for a in self._antennas], [a.y for a in self._antennas], '+')

        plt.show()

class Antenna(object):
    """
    Object to store information about individual antennas. Stores:
     * ID Number
     * Position relative to origin in meters (x(East), y(North), z(Up))
     * Polarization (1 or 2)
     * Status (1 is good, 0 is bad)
    """

    def __init__(self, id, x, y, z, status=1, pol=1, cable=None):
        self.id = int(id)
        self.pol = int(pol)
        self.status = int(status)
        self.x = x
        self.y = y
        self.z = z

        if cable is None:
            self._cable = []
        else:
            self._cable = cable

    def __repr__(self):
        n = self.__class__.__name__
        a = [(attr, getattr(self, attr)) for attr in ('id', 'pol', 'status', 'x', 'y', 'z', 'cable')]
        
        if a[-1][1] != []:
            a[-1] = ('cable', repr(a[-1][1]).replace(',\n    ', ', '))

        output = '<%s:' % n
        first = True
        for key, value in a:
            output += '%s %s=%s' % (('' if first else ','), key, value)
            first = False
        output += '>'
        return fill(output, subsequent_indent='          ')
        
    @property
    def cable(self):
        """
        Return the `station.Cable` object associated with this antenna.
        """
        return self._cable

class Cable(object):
    """
    Object to store information about a cable. Stores:
     * ID Number
     * Length [m]
     * Velocity Factor (vf)
     * Inner conductor radius (a) [m]
     * Outer conductor radius (b) [m]
     * Inner conductor conductivity (sigma_a) [S/m]
     * Outer conductor conductivity (sigma_b) [S/m]
     * Relative permittity (dielectric constant) (k)
     * Reference frequency (f0) [Hz]

    .. note::
     Attenuation and delay methods are described in LWA Memo #187.
    """

    def __init__(self, id, length, vf, a, b, sigma_a, sigma_b, k, f0):
        self.id = id
        self.length = length
        self.vf = vf
        self.a = a
        self.b = b
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.k = k
        self.f0 = f0
    
    def __repr__(self):
        n = self.__class__.__name__
        a = [(attr, getattr(self, attr)) for attr in ('id', 'length', 'vf', 'a', 'b', 'sigma_a', 'sigma_b', 'k', 'f0')]
        
        output = '<%s:' % n
        first = True
        for key, value in a:
            output += '%s %s=%s' % (('' if first else ','), key, value)
            first = False
        output += '>'
        return fill(output, subsequent_indent='         ')
    
    def attenuation(self, frequency, dB=False):
        """
        Compute the signal attenuation due to cable loss.

        Inputs:
         * frequency - Frequency in Hz.
         * dB - Return the cable attenuation in dB.

        Returns:
         * Cable attenuation.
        """
        a0 = ( np.sqrt(np.pi*self.k*eps0.value / 4.0) * ((self.sigma_a**(-0.5) / self.a) + (self.sigma_b**(-0.5) / self.b))
             * (np.log(self.b/self.a))**(-1) * self.f0**(0.5) )
        
        att = np.exp(2 * a0 * self.length * np.sqrt(frequency/self.f0))
        if dB:
            att = 10.0 * np.log10(att)

        return att

    def delay(self, frequency):
        """
        Compute the signal delay, both bulk and dispersive terms, induced by the cable.

        Inputs:
         * frequency - Frequency in Hz.

        Returns:
         * Cable delay in seconds. 
        """
        #Bulk delay term.
        bulk = self.length / (self.vf * c.value)

        #Dispersive delay term.
        disp = ( (self.length / (self.vf*c.value))* (1.0 / (8.0*np.sqrt(np.pi*mu0.value))) *
                 (self.sigma_a**(-0.5) / self.a + self.sigma_b**(-0.5) / self.b) * 
                 (np.log(self.b/self.a))**(-1) * frequency**(-0.5) )

        delay = bulk + disp

        return delay

class LMR200(Cable):
    """
    Convenience subclass to populate a Cable object for a LMR200 cable. Inputs:
     * Cable ID
     * Cable length [m]

    .. note::
     The values of the attenuation method are set to match the Times Mircowave LMR-200 data sheet.
     See https://www.timesmicrowave.com/DataSheets/CableProducts/LMR-200.pdf for more information.
    """

    def __init__(self, id, length):
        self.id = id 
        self.length=length
        self.vf = 0.83
        self.a = 5.600e-4
        self.b = 1.535e-3
        self.sigma_a = 5.96e7 #Copper
        self.sigma_b = 1.73e7 #Aluminum tape
        self.k = 1.45
        self.f0 = 5.0e7

        super().__init__(id=self.id, length=self.length, vf=self.vf, a=self.a, b=self.b, sigma_a=self.sigma_a, sigma_b=self.sigma_b, k=self.k, f0=self.f0)

    def attenuation(self, frequency, dB=False):
        """
        Compute the signal attenuation due to cable loss.

        Inputs:
         * frequency - Frequency in Hz.
         * dB - Return the cable attenuation in dB.

        Returns:
         * Cable attenuation.
        """

        freqs = np.array([30, 50, 150, 220, 450, 900, 1500, 1800, 2000, 2500, 5800, 8000])*1e6
        ref   = 10**( np.array([5.8, 7.5, 13.1, 15.9, 22.8, 32.6, 42.4, 46.6, 49.3, 55.4, 86.5, 102.8]) * (self.length / 100.0) / 10.0 )

        intp = interp1d(freqs, ref, kind='linear')
        att = intp(frequency)
        if dB:
            att = 10.0 * np.log10(att)

        return att

class LMR400(Cable):
    """
    Convenience subclass to populate a Cable object for a LMR400 cable. Inputs:
     * Cable ID
     * Cable length [m]

    .. note::
     The values of the attenuation method are set to match the Times Mircowave LMR-400 data sheet.
     See https://www.timesmicrowave.com/DataSheets/CableProducts/LMR-400.pdf for more information.
    """
   
    def __init__(self, id, length):
        self.id = id
        self.length = length
        self.vf = 0.84
        self.a = 1.37e-3
        self.b = 3.70e-3
        self.sigma_a = 5.96e7 #Copper
        self.sigma_b = 1.73e7 #Aluminum tape
        self.k = 1.38
        self.f0 = 5.0e7

        super().__init__(id=self.id, length=self.length, vf=self.vf, a=self.a, b=self.b, sigma_a=self.sigma_a, sigma_b=self.sigma_b, k=self.k, f0=self.f0)

    def attenuation(self, frequency, dB=False):
        """
        Compute the signal attenuation due to cable loss.

        Inputs:
         * frequency - Frequency in Hz.
         * dB - Return the cable attenuation in dB.

        Returns:
         * Cable attenuation.
        """

        freqs = np.array([30, 50, 150, 220, 450, 900, 1500, 1800, 2000, 2500, 5800, 8000])*1e6
        ref   = 10**( np.array([2.2, 2.9, 5.0, 6.1, 8.9, 12.8, 16.8, 18.6, 19.6, 22.2, 35.5, 42.7]) * (self.length / 100.0) / 10.0 )

        intp = interp1d(freqs, ref, kind='linear')
        att = intp(frequency)
        if dB:
            att = 10.0 * np.log10(att)

        return att

def load_station(arg):
    """
    Load in a template file (.txt) which contains information
    about the station. See the README and station_template.txt
    files for more information.

    Inputs:
     * arg - Station template file (.txt)

    Returns:
     * `station.Station` object
    """ 
    f = open(arg, 'r')
    while True:
        line = f.readline()
        
        #Check to see if we're at the end of the file.
        if line == '':
            break

        #Read the Station Information section.
        if re.match('^###', line) and re.search('Station', line):
            print('Reading Station Information')
            continue

        if re.match('^NAME', line):
            name = line.split('"')[1]
            continue
        elif re.match('^LAT', line):
            lat = float(line.split(' ')[-1])
        elif re.match('^LON', line):
            lon = float(line.split(' ')[-1])
        elif re.match('^ANTS', line):
            numAnts = int(line.split(' ')[-1])
            print("Number of antennas is %i" % numAnts)
            continue

        #Read the Antenna Information section.
        if re.match('^###', line) and re.search('Antenna', line):
            antSection = True
            antID, antX, antY, antZ, antPol, antStat = [], [], [], [], [], []
            print('Reading Antenna Information')
            continue
        
        if re.match('^1', line) and antSection:
            #We've found the first line with antenna data.
            l = line.split(' ')
            while '' in l:
                l.remove('')
            for param, value in zip([antID, antX, antY, antZ, antPol, antStat], l[1:]):
                param.append( float(value) )
            
            #Read the rest of the antennas.
            for i in range(numAnts-1):
                line = f.readline()
    
                l = line.split(' ')
                while '' in l:
                    l.remove('')
                for param, value in zip([antID, antX, antY, antZ, antPol, antStat], l[1:]):
                    param.append( float(value) )
            continue

        #Read the Cable Information section.
        if re.match('^###', line) and re.search('Cable', line):
            cblSection = True
            antSection = False
            cables = []
            print('Reading Cable Information')
            continue

        if re.match('^1', line) and cblSection:
            #We've found the first line with cable data. 
            l = line.split(' ')
            while '' in l:
                l.remove('')
            #Figure out if it is a LMR200/400 or a custom cable.
            if l[2] == 'LMR200':
                cables.append( LMR200(id=l[1], length=float(l[3])) )
            elif l[2] == 'LMR400':
                cables.append( LMR400(id=l[1], length=float(l[3])) )
            else:
                cables.append( Cable(id=l[1], length=float(l[3]), vf=float(l[4]), a=float(l[5]), b=float(l[6]), 
                                    sigma_a=float(l[7]), sigma_b=float(l[8]), k=float(l[9]), f0=float(l[10])) )
            #Read the rest of the cable lines.
            for i in range(numAnts-1):
                line = f.readline()

                l = line.split(' ')
                while '' in l:
                    l.remove('')
                #Figure out if it is a LMR200/400 or a custom cable.
                if l[2] == 'LMR200':
                    cables.append( LMR200(id=l[1], length=float(l[3])) )
                elif l[2] == 'LMR400':
                    cables.append( LMR400(id=l[1], length=float(l[3])) )
                else:
                    cables.append( Cable(id=l[1], length=float(l[3]), vf=float(l[4]), a=float(l[5]), b=float(l[6]), 
                                        sigma_a=float(l[7]), sigma_b=float(l[8]), k=float(l[9]), f0=float(l[10])) )
            continue

    #Put it all together.
    antennas = []
    for i in range(numAnts):
        antennas.append(Antenna(id=antID[i], x=antX[i], y=antY[i], z=antZ[i],
                                status=antStat[i], pol=antPol[i], cable=cables[i]))

    station = Station(name=name, latitude=lat, longitude=lon, antennas=antennas)
    return station

try:
    from lsl.common.stations import parse_ssmif

    def load_LWA(ssmif):
        """
        Read in a LWA SSMIF file and return a fully populated `station.Station` object.
        
        Inputs:
         * ssmif - LWA SSMIF file (.txt)

        Returns:
         * `station.Station` object.
        
        .. note:: This function requires the LWA Software Library (LSL)
        """
        site = parse_ssmif(ssmif)

        lat, lon = site.lat*(180.0/np.pi), site.lon*(180.0/np.pi)

        antennas = []
        for ant in site.antennas:
            #Antenna information.
            antID   = ant.id
            antPol  = ant.pol + 1
            antStat = 1 if ant.combined_status == 33 else 0
            antX, antY, antZ = ant.stand.x, ant.stand.y, ant.stand.z
            
            #Cable information.
            cblID   = ant.cable.id
            cblLen  = ant.cable.length * ant.cable.stretch

            antennas.append( Antenna(id=antID, x=antX, y=antY, z=antZ, status=antStat, pol=antPol, 
                            cable=LMR200(id=cblID, length=cblLen)) )

        station = Station(name=site.name, latitude=lat, longitude=lon, antennas=antennas)

        return station

except ImportError:
    import warnings
    warnings.simplefilter('always', ImportWarning)
    warnings.warn('Cannot import lsl, loading from a LWA SSMIF file disabled', ImportWarning)

    def load_LWA(ssmif):
        """
        Read in a LWA SSMIF file and return a fully populated `station.Station` object.
        
        Inputs:
         * ssmif - LWA SSMIF file (.txt)

        Returns:
         * `station.Station` object.
        
        .. note:: This function requires the LWA Software Library (LSL)
        """
        raise RuntimeError('Not supported without lsl')

