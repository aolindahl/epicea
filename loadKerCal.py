# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:32:11 2015

@author: marcus
"""

import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import interpolate
from scipy import ndimage
from scipy import special
from scipy import fftpack
from hankel import HankelTransform
import csv
import matplotlib.pyplot as plt
from sys import stdout


class kerCalibration:

#constructor that reads the reference csv file and saves the energy axis 
#and corresponding intensity, takes a filename
    def __init__(self, fileName):       
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            kerV = [];
            intensityV = []; 
            for row in reader:
                kerV.append(float(row[0]))
                intensityV.append(float(row[1]))
        self.kerRawV = kerV
        self.intensityRawV = intensityV

        #data extracted from plot is grainy
        #the fine features are best captured using the maximum of a range of values
        #moving average gives bad results
        #this upshifts the intensity by a value determined by the noise level
        #can roughly be corrected using an area without fine features
        samplesToAverageS = 100;
        newIndecesV = range(len(kerV)-samplesToAverageS+1)
        kerMeanV = np.zeros(len(kerV)-samplesToAverageS+1)
        intensityMaxV = np.zeros(len(kerV)-samplesToAverageS+1)
        intensityMeanV = np.zeros(len(kerV)-samplesToAverageS+1)
        for idxS in newIndecesV:
            kerMeanV[idxS] = kerV[idxS]
            intensityMaxV[idxS] = max(intensityV[idxS:idxS+samplesToAverageS])
            intensityMeanV[idxS] = sum(intensityV[idxS:idxS+samplesToAverageS])/float(samplesToAverageS)
            
        unJustifiedupshiftS = np.mean(intensityMaxV[int(len(intensityMaxV)*0.75):]-intensityMeanV[int(len(intensityMeanV)*0.75):])
        intensityMaxV = intensityMaxV-unJustifiedupshiftS
        #make sure nothing is <0
        intensityMaxV = intensityMaxV-np.min(intensityMaxV)
        
        #downsample and extend to zero and high momentum
        interpolationKerV = np.linspace(0., np.max(kerMeanV), num=3000, endpoint=True)
        #maxS = np.max(kerMeanV)*(1.+1.e-14)
        kerMeanV = np.concatenate(((0., np.min(kerMeanV)*(1.-1.e-14)), kerMeanV))
        intensityMaxV = np.concatenate(((0., 0.), intensityMaxV))
        interpolatedIntensityRadialDistanceC = interpolate.interp1d(kerMeanV, intensityMaxV)
        
        #make corrected values accessible  
        self.kerV = interpolationKerV;
        self.intensityKerV = interpolatedIntensityRadialDistanceC(interpolationKerV)/np.max(interpolatedIntensityRadialDistanceC(interpolationKerV))
        #calculate distribution versus absolute momentum of one ion
        #due to momentum conservation both ions should have the same energy
        #therefore we have to divide the energy axis by two and rescale due to p^2->p
        ionMassS = 14.*1.67262178e-27/9.10938291e-31
        evS = 1.
        self.momentumV = np.sqrt(self.kerV*evS*ionMassS)
        self.intensityAbsoluteMomentumV = self.intensityKerV*self.momentumV
        self.intensityAbsoluteMomentumV[np.isnan(self.intensityAbsoluteMomentumV)] = 0.
        self.intensityAbsoluteMomentumV = self.intensityAbsoluteMomentumV/np.max(self.intensityAbsoluteMomentumV)
        
        #calculate distribution against radial momentum(abel transform)
        self.radialEnergyV = self.kerV[0:-1]
        idxV = range(len(self.kerV)-1)
        intensityRadialEnergyV = np.zeros(len(self.kerV)-1)
        for idxS in idxV:
            prS = self.kerV[idxS]
            intensityRadialEnergyV[idxS] = integrate.trapz(2.*self.intensityKerV[idxS+1:]*self.kerV[idxS+1:] /
                                                    np.sqrt(self.kerV[idxS+1:]*self.kerV[idxS+1:]-prS*prS), self.kerV[idxS:-1])
        self.intensityRadialEnergyV = intensityRadialEnergyV/np.max(intensityRadialEnergyV)
        
        self.radialMomentumV = self.momentumV[0:-1]
        idxV = range(len(self.kerV)-1)
        intensityRadialMomentumV = np.zeros(len(self.kerV)-1)
        for idxS in idxV:
            prS = self.momentumV[idxS]
            intensityRadialMomentumV[idxS] = integrate.trapz(2.*self.intensityAbsoluteMomentumV[idxS+1:]*self.momentumV[idxS+1:] /
                                                    np.sqrt(self.momentumV[idxS+1:]*self.momentumV[idxS+1:]-prS*prS), self.momentumV[idxS:-1])
        self.intensityRadialMomentumV = intensityRadialMomentumV/np.max(intensityRadialMomentumV)

    def abelInvert(self, radialDistanceV, intensityRadialDistanceV, samplingS):  
        #save the position data
        intensityRadialDistanceV = (intensityRadialDistanceV/np.max(intensityRadialDistanceV))
        self.radialDistanceV = radialDistanceV
        self.intensityRadialDistanceV = intensityRadialDistanceV
        
        #interpolate on requested grid
        intenstityRadialDistanceInterpV =  interpolate.interp1d(radialDistanceV, intensityRadialDistanceV)               
        radialDistanceV = np.linspace(np.min(radialDistanceV), np.max(radialDistanceV), samplingS, endpoint=True)
        intensityRadialDistanceV = intenstityRadialDistanceInterpV(radialDistanceV)
        
        #fourier part
        idxV = range(len(radialDistanceV))
        intensityRadialDistanceFTV = np.zeros(len(radialDistanceV))
        for idxS in idxV:
            prS = radialDistanceV[idxS]  
            intensityRadialDistanceFTV[idxS] = np.trapz(intensityRadialDistanceV * np.cos(radialDistanceV*prS*(2*np.pi)),radialDistanceV)
        #hankel part
        idxV = range(len(radialDistanceV))
        intensityUnscaledMomentumV = np.zeros(len(radialDistanceV))
        for idxS in idxV:
            prS = radialDistanceV[idxS]
            intensityUnscaledMomentumV[idxS] = np.trapz(radialDistanceV * special.j0(prS*radialDistanceV*(2*np.pi)) * intensityRadialDistanceFTV, radialDistanceV)               
        
        intensityUnscaledMomentumV = intensityUnscaledMomentumV
        self.intensityUnscaledMomentumV = intensityUnscaledMomentumV[radialDistanceV>1]/np.max(intensityUnscaledMomentumV[radialDistanceV>1])
        self.radialDistanceV = radialDistanceV[radialDistanceV>1]
        

##################
###script start
    
#load comparison data
kC = kerCalibration('n2KerLundqvist.csv')
#############plot raw data
#plt.plot(kC.kerRawV, kC.intensityRawV)
############test abel inversion
if 1:
    plt.figure(0)
    kC.abelInvert(kC.radialEnergyV, kC.intensityRadialEnergyV, 3000)   
    plt.plot(kC.kerV, kC.intensityKerV, 
             kC.radialEnergyV, kC.intensityRadialEnergyV, 
             kC.radialDistanceV, kC.intensityUnscaledMomentumV)
    plt.xlabel('ker[eV]')
    plt.ylim([0,1.3])
    plt.legend(['reference(lundqvist)', 'abel transform', 'back transform'])

#plt.figure(2)
#plt.imshow(np.log(xy_img))
#du

plt.figure(1)
plt.plot(kC.momentumV/np.max(kC.momentumV), kC.intensityAbsoluteMomentumV, kC.radialMomentumV/np.max(kC.momentumV), kC.intensityRadialMomentumV)
plt.xlabel('abs momentum release[arb. u.]')
plt.ylim([0,1.3])
plt.legend(['reference(lundqvist)', 'abel transform'])

#if doing this, rescaling accoring to r*sin(theta) has to be done after the line wise transform(abel assumes cylindrical symmetry of which spherical is a special case)
#if we are dealing with spherical symmetry, theta is arbitrary and can be averaged for better statistics
intensityRadialDistanceV = rth_img[:,:].sum(axis=0)
radialDistanceV = r_axis_mm

#abelinvert takes x, y, resolution(higher=more), works better on equally spaced data
kC.abelInvert(radialDistanceV, ndimage.filters.gaussian_filter(intensityRadialDistanceV, 1), 3000)
plt.figure(2)
plt.plot(radialDistanceV/np.max(radialDistanceV), intensityRadialDistanceV/np.max(intensityRadialDistanceV))
plt.plot(kC.radialDistanceV/np.max(kC.radialDistanceV), (kC.intensityUnscaledMomentumV*kC.radialDistanceV)/np.max(kC.intensityUnscaledMomentumV*kC.radialDistanceV))
plt.xlabel('momentum release[arb. u.]')

plt.legend(['radial(direct data)', 'abel transform*r'])











