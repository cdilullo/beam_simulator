"""
Utilities related to representing an indiviudal antenna
radiation pattern from a NEC4 output file.
"""
import os
import sys
import aipy
import numpy as np
import matplotlib.pyplot as plt


from scipy.special import sph_harm


__version__ = '1.0'
__authors_ = ['Chris DiLullo', 'Jayce Dowell']
__all__ = ['AntennaPattern']

class AntennaPattern(object):
    """
    Object to store the antenna gain pattern
    for a single frequency from a NEC4 output 
    file.
    
    .. note:: 
     The NEC4 file must contain an EXCITATION section.
    """
    
    def __init__(self, name):
        
        self.antenna_pat = np.zeros((360,91), dtype=np.complex64)
        
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

def sphfit(az, alt, data, lmax=5, degrees=False, real_only=False):
    """
    Decompose a spherical or semi-spherical data set into spherical harmonics.  
    Inputs
    ------
      * az: 2-D numpy array of azimuth coordinates in radians or degrees if the 
            `degrees` keyword is set
      * alt: 2-D numpy array of altitude coordinates in radian or degrees if the 
             `degrees` keyword is set
      * data: 2-D numpy array of the data to be fit.  If the data array is purely
             real, then the `real_only` keyword can be set which speeds up the 
             decomposition
      * lmax: integer setting the maximum order harmonic to fit
    Keywords
    --------
      * degrees: boolean of whether or not the input azimuth and altitude coordinates
         are in degrees or not
      * real_only: boolean of whether or not the input data is purely real or not.  If
        the data are real, only coefficients for modes >=0 are computed.
    Returned is a 1-D complex numpy array with the spherical harmonic coefficients 
    packed packed in order of increasing harmonic order and increasing mode, i.e.,
    (0,0), (1,-1), (1,0), (1,1), (2,-2), etc.  If the `real_only` keyword has been 
    set, the negative coefficients for the negative modes are excluded from the 
    output array.
    
    .. note::
        sphfit was designed to fit the LWA dipole response pattern as a function of
        azimuth and elevation.  Elevation angles are mapped to theta angles by adding
        pi/2 so that an elevation of 90 degrees corresponds to a theta of 180 degrees.
        To fit in terms of spherical coordianates, subtract pi/2 from the theta values
        before running.
    """
    
    if degrees:
        rAz = az*np.pi/180.0
        rAlt = alt*np.pi/180.0
    else:
        rAz = 1.0*az
        rAlt = 1.0*alt
    rAlt += np.pi/2
    sinAlt = np.sin(rAlt)
    
    if real_only:
        nTerms = int((lmax*(lmax+3)+2)/2)
        terms = np.zeros(nTerms, dtype=np.complex64)
        
        t = 0
        for l in range(lmax+1):
            for m in range(0, l+1):
                Ylm = sph_harm(m, l, rAz, rAlt)
                terms[t] = (data*sinAlt*Ylm.conj()).sum() * (rAz[1,0]-rAz[0,0])*(rAlt[0,1]-rAlt[0,0])
                t += 1
                
    else:
        nTerms = int((lmax+1)**2)
        terms = np.zeros(nTerms, dtype=np.complex64)
        
        t = 0
        for l in range(lmax+1):
            for m in range(-l, l+1):
                Ylm = sph_harm(m, l, rAz, rAlt)
                terms[t] = (data*sinAlt*Ylm.conj()).sum()
                t += 1
                
    return terms

def sphval(terms, az, alt, degrees=False, real_only=False):
    """
    Evaluate a set of spherical harmonic coefficents at a specified set of
    azimuth and altitude coordinates.
    Inputs
    ------
      * terms: 1-D complex numpy array, typically from sphfit
      * az: 2-D numpy array of azimuth coordinates in radians or degrees if the 
            `degrees` keyword is set
      * alt: 2-D numpy array of altitude coordinates in radian or degrees if the 
             `degrees` keyword is set
    Keywords
    --------
      * degrees: boolean of whether or not the input azimuth and altitude coordinates
                 are in degrees or not
      * real_only: boolean of whether or not the input data is purely real or not.  If
                  the data are real, only coefficients for modes >=0 are computed.
    Returns a 2-D numpy array of the harmoics evalated and summed at the given 
    coordinates.
    
    .. note::
        sphfit was designed to fit the LWA dipole response pattern as a function of
        azimuth and elevation.  Elevation angles are mapped to theta angles by adding
        pi/2 so that an elevation of 90 degrees corresponds to a theta of 180 degrees.
        To spherical harmonics in terms of spherical coordianates, subtract pi/2 from 
        the theta values before running.
    """
    
    if degrees:
        rAz = az*np.pi/180.0
        rAlt = alt*np.pi/180.0
    else:
        rAz = 1.0*az
        rAlt = 1.0*alt
    rAlt += np.pi/2
    
    nTerms = terms.size
    if real_only:
        lmax = int((np.sqrt(1+8*nTerms)-3)/2)

        t = 0
        out = np.zeros(az.shape, dtype=np.float32)
        for l in range(lmax+1):
            Ylm = sph_harm(0, l, rAz, rAlt)
            out += np.real(terms[t]*Ylm)
            t += 1
            for m in range(1, l+1):
                Ylm = sph_harm(m, l, rAz, rAlt)
                out += np.real(terms[t]*Ylm)
                out += np.real(terms[t]*Ylm.conj()/(-1)**m)
                t += 1
                
    else:
        lmax = int(np.sqrt(nTerms)-1)
        
        t = 0
        out = np.zeros(az.shape, dtype=np.complex64)
        for l in range(lmax+1):
            for m in range(-l, l+1):
                Ylm = sph_harm(m, l, rAz, rAlt)
                out += terms[t]*Ylm
                t += 1
                
    return out


def fit_spherical_harmonics(freq, p1, p2, t1, t2, verbose=False):
    """
    Script to represent the full gain pattern of a single 
    antenna at a single frequency in terms of spherical 
    harmonics.

    Parameters
    ----------
     * freq: NEC simulation frequency in MHz.
     * p1: NEC output file for parallel component of polarization 1.
     * p2: NEC output file for parallel component of polarization 2.
     * t1: NEC output file for transverse component of polarization 1.
     * t2: NEC output file for transverse component of polarization 2.
    
    Returns
    -------
    A .npz file containing the terms of the spherical harmonic fit for each 
    polarization.
    """
    
    #Compute the full pattern (parallel + transverse) for each polarization.
    extP = AntennaPattern(p1)
    extT = AntennaPattern(t1)
    ext1 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
    ext1 /= ext1.max()

    extP = AntennaPattern(p2)
    extT = AntennaPattern(t2)
    ext2 = (np.abs(extP.antenna_pat)**2 + np.abs(extT.antenna_pat)**2) / 2.0
    ext2 /= ext2.max()
    
    #Build the Az/El arrays.
    az = np.arange(0,360,1)
    el = np.arange(0,91,1)
    el,az = np.meshgrid(el,az)
    
    top = aipy.coord.azalt2top(np.array([[az*np.pi/180.0], [el*np.pi/180.0]]))
    theta, phi = aipy.coord.xyz2thphi(top)
    theta = theta.squeeze()
    phi = phi.squeeze()
    
    terms1 = sphfit(phi, theta-np.pi/2, ext1, lmax=12, degrees=False, real_only=True)
    terms2 = sphfit(phi, theta-np.pi/2, ext2, lmax=12, degrees=False, real_only=True)
    
    fit1 = sphval(terms1, phi, theta-np.pi/2, degrees=False, real_only=True)
    fit2 = sphval(terms2, phi, theta-np.pi/2, degrees=False, real_only=True)
    diff1 = ext1 - fit1
    diff2 = ext2 - fit2
    
    if verbose:
        print('Min, Mean, and Max differences between data and fit are:')
        print('Polarization 1: %.5f, %.5f, %.5f' % (diff1.min(), diff1.mean(), diff1.max()))
        print('Polarization 2: %.5f, %.5f, %.5f' % (diff2.min(), diff2.mean(), diff2.max()))
        print('')
    print('Fit coefficients saved to file: SphericalHarmonicsFit_%.1fMHz.npz' % freq)

    np.savez('SphericalHarmonicsFit_%.1fMHz.npz' % freq, l=12, realOnly=True,
             terms1=terms1, terms2=terms2)

def combine_harmonic_fits(*args):
    """
    Takes in spherical harmonic fits output by fit_spherical_harmonics and 
    represents them as a polynomial in frequency. Returns the fit coefficients

    Parameters
    ----------
     * args: Series of .npz files output by fit_spherical_harmonics for different 
           frequencies.
        
    Returns
    -------
    A .npz file containing all input spherical harmonic fit terms.
    """

    freqs, coeffs1, coeffs2 = [], [], []
    for f in args:
        freqs.append(float(f.split('_')[1].split('.')[0]))
        coeffs1.append( np.load(f)['terms1'] )
        coeffs2.append( np.load(f)['terms2'] )  

    freqs = np.array(freqs)
    coeffs1 = np.array(coeffs1)
    coeffs2 = np.array(coeffs2)
    
    coeffs1 = np.polyfit(freqs/1e3, coeffs1, deg=coeffs1.shape[0]-1)
    coeffs2 = np.polyfit(freqs/1e3, coeffs2, deg=coeffs2.shape[0]-1)
    
    np.savez('beam_coefficients.npz', coeffs1=coeffs1, coeffs2=coeffs2,
             lmax=12, deg=coeffs1.shape[0]-1)
