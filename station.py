"""
Collection of classes that represent the various levels of an array.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from astropy.constants import c, mu0, eps0

__version__ = '1.0'
__authors__ = ['Chris DiLullo', 'Jayce Dowell']

class Station(object):

    def __init__(self, name=None, antennas=None):

        self.name = name

        if antennas is None:
            self._antennas = []
        else:
            self._antennas = list(antennas)

    @property
    def antennas(self):
        return self._antennas
    
    def plot_antennas(self):
        
        f, ax = plt.subplots(1,1)
        ax.set_title('%s' % self.name, fontsize='x-large')
        ax.set_xlabel('X [m]', fontsize='large')
        ax.set_ylabel('Y [m]', fontsize='large')
        ax.tick_params(direction='in',size=5)
        ax.plot([a.x for a in self._antennas], [a.y for a in self._antennas], '+')

        plt.show()

class Antenna(object):
    """
    Object to store information about individual antennas.
    Stores:
     * ID Number
     * Position relative to origin in meters (x,y,z)
     * Polarization (1 or 2)
     * Status (1 is good, 0 is bad)
    """

    def __init__(self, id, x, y, z, status=1, pol=1, cable=None):
        self.id = int(id)
        self.pol = pol
        self.status = status
        self.x = x
        self.y = y
        self.z = z

        if cable is None:
            self._cable = []
        else:
            self._cable = cable

    @property
    def cable(self):
        return self._cable

class Cable(object):
    """
    Object to store information about a cable.
    Stores:
    * ID Number
    * Length [m]
    * Velocity Factor (vf)
    * Inner conductor radius (a) [m]
    * Outer conductor radius (b) [m]
    * Inner conductor conductivity (sigma_a) [S/m]
    * Outer conductor conductivity (sigma_b) [S/m]
    * Relative permittity (dielectric constant) (k)
    * Reference frequency (f0) [Hz]
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
    
    def attenuation(self, frequency, dB=False):
        a0 = ( np.sqrt(np.pi*self.k*eps0.value / 4.0) * ((self.sigma_a**(-0.5) / self.a) + (self.sigma_b**(-0.5) / self.b))
             * (np.log(self.b/self.a))**(-1) * self.f0**(0.5) )
        
        att = np.exp(2 * a0 * self.length * np.sqrt(frequency/self.f0))
        if dB:
            att = 10.0 * np.log10(att)

        return att

    def delay(self, frequency):

        #Bulk delay term.
        bulk = self.length / (self.vf * c.value)

        #Dispersive delay term.
        disp = ( (self.length / (self.vf*c.value))* (1.0 / (8.0*np.sqrt(np.pi*mu0.value))) *
                 (self.sigma_a**(-0.5) / self.a + self.sigma_b**(-0.5) / self.b) * 
                 (np.log(self.b/self.a))**(-1) * frequency**(-0.5) )

        delay = bulk + disp

        return delay

class LMR200(Cable):
    
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

class LMR400(Cable):
    """
    NOTE: This is not functioning properly yet.
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

def loadStation(arg):
    """
    Load in a template file (.txt) which contains information
    about the station. See the README and station_template.txt
    files for more information.
    """

    f = open(arg,'r')
    lines = f.readlines()

    #Find the station name and number of antennas.
    name = lines[5].split('"')[1]
    numAnts = int(lines[6].split(' ')[-1])

    #Get the antenna data.
    antID, x, y, z, pol, stat = [], [], [], [], [], []
    for l in lines[11:11+numAnts]:
            #Quickly clean up the lines.
            l = l.split(' ')
            while '' in l:
                l.remove('')
            antID.append(int(l[0]))
            x.append(float(l[1]))
            y.append(float(l[2]))
            z.append(float(l[3]))
            pol.append(int(l[4]))
            stat.append(int(l[5]))

    #Get the cable data (starts at index 11 + numAnts + 4)
    start = 15 + numAnts
    cables = []
    for l in lines[start:start+numAnts]:
        l = l.split(' ')
        while '' in l:
            l.remove('')
        if l[1] == 'LMR200':
            cables.append(LMR200(id=int(l[0]),length=float(l[2])))
        elif l[1] == 'LMR400':
            cables.append(LMR400(id=int(l[0]),length=float(l[2])))

    #Put it all together.
    antennas = []
    for i in range(numAnts):
        antennas.append(Antenna(id=antID[i], x=x[i], y=y[i], z=z[i],
                                pol=pol[i],cable=cables[i]))

    station = Station(name=name,antennas=antennas)
    return station

def loadLWA(arg):
    """
    Load in an LWA SSMIF (.txt) file and generate the equivalent
    station object.
    """
    #Read in and parse the SSMIF (see lsl.common.stations.py)
    kwdRE = re.compile(r'(?P<keyword>[A-Z_0-9]+)(\[(?P<id1>[0-9]+?)\])?(\[(?P<id2>[0-9]+?)\])?(\[(?P<id3>[0-9]+?)\])?')
    
    # Loop over the lines in the file
    with open(arg, 'r') as fh:
        for line in fh:
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            if len(line) == 0 or line.isspace():
                continue
            if line[0] == '#':
                continue
                
            keywordSection, value = line.split(None, 1)
            value = value.split('#', 1)[0]
            
            mtch = kwdRE.match(keywordSection)
            keyword = mtch.group('keyword')
            
            ids = [-1, -1, -1]
            for i in range(3):
                try:
                    ids[i] = int(mtch.group('id%i' % (i+1)))
                except TypeError:
                    pass
            
            # Station Name
            if keyword == 'STATION_ID':
                idn = str(value)
                continue
            
            # Stand & Antenna Data
            if keyword == 'N_STD':
                nStand = int(value)
                
                stdPos = [[0.0, 0.0, 0.0] for n in range(nStand)]
                stdAnt = [n//2+1 for n in range(2*nStand)]
                stdStat = [3 for n in range(2*nStand)]     
                continue
                
            if keyword == 'STD_LX':
                stdPos[ids[0]-1][0] = float(value)
                continue
            if keyword == 'STD_LY':
                stdPos[ids[0]-1][1] = float(value)
                continue
            if keyword == 'STD_LZ':
                stdPos[ids[0]-1][2] = float(value)
                continue
                
            if keyword == 'ANT_STD':
                stdAnt[ids[0]-1] = int(value)
                continue
            if keyword == 'ANT_STAT':
                stdStat[ids[0]-1] = int(value)
                continue
                
            # FEE, Cable, & SEP Data
            
            if keyword == 'N_FEE':
                nFee = int(value)
                
                feeID = ["UNK" for n in range(nFee)]
                feeStat = [3 for n in range(nFee)]
                feeDesi = [1 for n in range(nFee)]
                feeAnt1 = [2*n+1 for n in range(nFee)]
                feeAnt2 = [2*n+2 for n in range(nFee)]
                
                continue
                
            if keyword == 'FEE_ID':
                feeID[ids[0]-1] = value
                continue
                
            if keyword == 'FEE_STAT':
                feeStat[ids[0]-1] = int(value)
                continue
                
            if keyword == 'FEE_DESI':
                feeDesi[ids[0]-1] = int(value)
                continue
                
            if keyword == 'FEE_ANT1':
                feeAnt1[ids[0]-1] = int(value)
                continue
            if keyword == 'FEE_ANT2':
                feeAnt2[ids[0]-1] = int(value)
                continue
                
                
            if keyword == 'N_RPD':
                nRPD = int(value)
                
                rpdID = ['UNK' for n in range(nRPD)]
                rpdStat = [3 for n in range(nRPD)]
                rpdLeng = [0.0 for n in range(nRPD)]
                rpdStr = [1.0 for n in range(nRPD)]
                rpdAnt = [n+1 for n in range(nRPD)]
                
                continue
                
            if keyword == 'RPD_ID':
                rpdID[ids[0]-1] = value
                continue
                
            if keyword == 'RPD_STAT':
                rpdStat[ids[0]-1] = int(value)
                continue
                
            if keyword == 'RPD_LENG':
                rpdLeng[ids[0]-1] = float(value)
                continue
                
            if keyword == 'RPD_STR':
                if ids[0] == -1:
                    rpdStr = [float(value) for n in range(nRPD)]
                else:
                    rpdStr[ids[0]-1] = float(value)
                continue
                
            if keyword == 'RPD_ANT':
                rpdAnt[ids[0]-1] = int(value)
                continue

    status = np.ones(2*nStand)
    for i in range(2*nStand):
        if feeStat[i//2] != 3 or stdStat[i] != 3:
            status[i] = 0
            
    antennas = []
    for i in range(2*nStand):
        antennas.append( Antenna(id=i, x=stdPos[i//2][0], y=stdPos[i//2][1], z=stdPos[i//2][2], status=status[i], pol=i%2+1, 
                                cable=LMR200(id=rpdID[i], length=rpdLeng[i]*rpdStr[i])) )
    
    station = Station(name='LWA-'+idn, antennas=antennas)
    return station
