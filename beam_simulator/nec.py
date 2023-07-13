"""
Utilities related to representing an indiviudal antenna
radiation pattern from a NEC4 output file.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import sph_harm

__version__ = '1.0'
__authors_ = ['Chris DiLullo', 'Jayce Dowell']
__all__ = ['AntennaPattern', 'fit_antenna_response']

class AntennaPattern(object):
    """
    Object to store the antenna gain pattern
    for a single frequency from a NEC4 output 
    file. If no file is given, then an isotropic 
    gain pattern will be generated.
    
    .. note:: 
     The NEC4 file must contain an EXCITATION section and be
     at 1 degree resolution.
    """
    
    def __init__(self, name=None):
        
        if name is not None:
            self.antenna_pat = np.zeros((361,91), dtype=np.complex64)
        
            fh = open(name, 'r')
            lines = fh.readlines()
        
            #Look for 'EXCITATION'.
            excitation = None
            for line in lines:
                if line.find('EXCITATION') >= 0:
                    excitation = True
                    break
                
            if excitation:
                self._read_excitation(lines)
            
            else:
                raise RuntimeError("The provided NEC4 file doesn't have an EXCITATION section!")
            
            fh.close()

        else:
            self.antenna_pat = np.ones((361,91), dtype=np.complex64)
        
    def _read_excitation(self, lines):
        """
        Private function to read an EXCITATION section 
        of a NEC output file.
        """
        
        for i, line in enumerate(lines):
            if line.find('EXCITATION') >= 0:
                theta = 90 - int(float(lines[i+2].split()[3]))
                phi = int(float(lines[i+2].split()[6]))
                powcurr = float(lines[i+12].split()[8])
                phscurr = float(lines[i+12].split()[9])
                self.antenna_pat[phi, theta] = powcurr*np.exp(1j*phscurr*np.pi/180.0)
        self.antenna_pat[-1, :] = self.antenna_pat[0, :]

    def plot_pattern(self, dB=False):
        """
        Function to plot the normalized antenna gain pattern.
        """
        normGain = (np.abs(self.antenna_pat)**2) / (np.abs(self.antenna_pat)**2).max()
        vmax, vmin = 1.0, 0.0
        if dB:
            normGain = 10*np.log10(normGain)
            vmax, vmin = 0.0, -30.0
        
        f, ax = plt.subplots(1,1)
        ax.set_title('Antenna Gain Pattern', fontsize='x-large')
        c = ax.imshow(normGain.T, origin='lower', interpolation='nearest', vmax=vmax, vmin=vmin)
        cb = f.colorbar(c,ax=ax, orientation='horizontal')
        cb.set_label('Normalized Gain' +(' [dB]' if dB else ' [lin.]'), fontsize='large')
        ax.set_xlabel('Azimuth [deg.]', fontsize='large')
        ax.set_ylabel('Elevation [deg.]', fontsize='large')
        ax.tick_params(direction='in',size=5)
        plt.show()

def fit_antenna_response(freqs, p1, t1, p2, t2, lmax=None):
    """
    Fit the gain response of an antenna as a polynomial in frequency.
 
    Parameters:
     * freqs: List or array of frequencies in MHz
     * p1: List or array containing the names of NEC4 outputs for polarization 1 parallel response
     * t1: List or array containing the names of NEC4 outputs for polarization 1 transverse response
     * p2: List or array containing the names of NEC4 outputs for polarization 2 parallel response
     * t2: List or array containing the names of NEC4 outputs for polarization 2 transverse response
     * lmax: Maximum degree of spherical harmonics to use for decomposition (None = No spherical harmonic decomposition)

    Returns:
     * A .npz file containing the coefficients of the polynomial in frequency
    """
    
    if lmax is None:
        pol1, pol2 = [], []
        for i in range(freqs.size):
            extP = AntennaPattern(p1[i])
            extT = AntennaPattern(t1[i])
            ext1 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
            ext1 /= ext1.max()

            extP = AntennaPattern(p2[i])
            extT = AntennaPattern(t2[i])
            ext2 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
            ext2 /= ext2.max()

            pol1.append(ext1)
            pol2.append(ext2)

        pol1, pol2 = np.array(pol1), np.array(pol2)
        pol1 = pol1.reshape((pol1.shape[0], pol1.shape[1]*pol1.shape[2]))
        pol2 = pol2.reshape((pol2.shape[0], pol2.shape[1]*pol2.shape[2]))

        coeffs1 = np.polyfit(freqs/1e3, pol1, deg=freqs.size-1)
        coeffs2 = np.polyfit(freqs/1e3, pol2, deg=freqs.size-1)

        np.savez('beam_coefficients.npz', coeffs1=coeffs1, coeffs2=coeffs2, deg=freqs.size-1)

    else:
        #Az is 0 at +Y (North), Phi is zero at +X (East)
        #El is 0 at horizon, Theta is 0 at +Z (Up)
        az = np.arange(0, 361, 1)/1.0 * (np.pi/180.0)
        el = np.arange(0, 91, 1)/1.0 * (np.pi/180.0)
        phi = (-(az - np.pi/2) + 2*np.pi) % (2*np.pi)
        theta = (np.pi/2.0) - el
        theta, phi = np.meshgrid(theta, phi)

        pol1, pol2 = [], []
        for i in range(freqs.size):
            extP = AntennaPattern(p1[i])
            extT = AntennaPattern(t1[i])
            ext1 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
            ext1 /= ext1.max()

            extP = AntennaPattern(p2[i])
            extT = AntennaPattern(t2[i])
            ext2 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
            ext2 /= ext2.max()

            nTerms = int((lmax*(lmax+3)+2)/2)
            terms1 = np.zeros(nTerms, dtype=np.complex64)
            terms2 = np.zeros(nTerms, dtype=np.complex64)
        
            t = 0
            for l in range(lmax+1):
                for m in range(0, l+1):
                    Ylm = sph_harm(m, l, phi, theta)
                    terms1[t] = (ext1*np.sin(theta)*Ylm.conj()).sum() * (phi[1,0]-phi[0,0])*(theta[0,1]-theta[0,0])
                    terms2[t] = (ext2*np.sin(theta)*Ylm.conj()).sum() * (phi[1,0]-phi[0,0])*(theta[0,1]-theta[0,0])
                    t += 1
        
            pol1.append(terms1)
            pol2.append(terms2)

        pol1, pol2 = np.array(pol1), np.array(pol2)
        coeffs1 = np.polyfit(freqs/1e3, pol1, deg=pol1.shape[0]-1)
        coeffs2 = np.polyfit(freqs/1e3, pol2, deg=pol2.shape[0]-1)

        np.savez('beam_coefficients.npz', coeffs1=coeffs1, coeffs2=coeffs2, deg=freqs.size-1, lmax=lmax)    
