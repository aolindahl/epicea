# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:32:11 2015

@author: marcus
"""

import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
from scipy import special
import csv
import matplotlib.pyplot as plt
plt.ion()


class kerCalibration:
    """Clss to handle ion KER calibration"""

    def __init__(self, fileName):
        """Init from file name.

        Constructor that reads the reference csv file and saves the energy axis
        and corresponding intensity.

        - params
        filename of csv formated data file"""

        # Storage space for the raw data
        kerRawV = []
        intensityRawV = []

        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                kerRawV.append(float(row[0]))
                intensityRawV.append(float(row[1]))

        # Make sur the points are in order and make them avaliable
        ordering_indexes = np.argsort(kerRawV)
        self.kerRawV = np.array(kerRawV)[ordering_indexes]
        self.intensityRawV = np.array(intensityRawV)[ordering_indexes]

        # data extracted from plot is grainy
        # the fine features are best captured using the maximum of a range
        # of values, moving average gives bad results
        # this upshifts the intensity by a value determined by the noise level
        # can roughly be corrected using an area without fine features
        samplesToAverageS = 30
        n_averaged_samples = len(self.kerRawV) / samplesToAverageS
#        kerMeanV = np.zeros(n_averaged_samples)
        kerMeanV = np.linspace(self.kerRawV.min(), self.kerRawV.max(),
                               n_averaged_samples)
        de = kerMeanV[1] - kerMeanV[0]
        kerMean_edges = np.concatenate([kerMeanV - de/2,
                                        [kerMeanV[-1] + de/2]])
        intensityMaxV = np.zeros_like(kerMeanV)
        intensityMeanV = np.zeros_like(kerMeanV)
        for i_bin in range(n_averaged_samples):
            # Find all the samples in the current bin
            I = ((kerMean_edges[i_bin] <= self.kerRawV) &
                 (self.kerRawV < kerMean_edges[i_bin+1]))
            # If there is any dafta in the bin...
            if np.any(I):
                # ...update max...
                intensityMaxV[i_bin] = self.intensityRawV[I].max()
                # ...and mean...
                intensityMeanV[i_bin] = self.intensityRawV[I].mean()

        if 0:
            plt.figure('kerCalibration diagnostics')
            plt.clf()
            plt.plot(kerMeanV, intensityMaxV, label='max')
            plt.plot(kerMeanV, intensityMeanV, label='mean')
            plt.legend(loc='best')

#        unJustifiedupshiftS = np.mean(
#            intensityMaxV[int(len(intensityMaxV)*0.75):] -
#            intensityMeanV[int(len(intensityMeanV)*0.75):])
#        intensityMaxV = intensityMaxV-unJustifiedupshiftS

        # make sure nothing is <0
#        intensityMaxV = intensityMaxV - np.min(intensityMaxV)

        # downsample and extend to zero and high momentum
        interpolationKerV = np.linspace(0., np.max(kerMeanV),
                                        num=3000, endpoint=True)
        interpolatedIntensityRadialDistanceC = interpolate.interp1d(
            np.concatenate([[0, kerMeanV[0] * (1.0 - 1e-14)], kerMeanV]),
            np.concatenate([[0, 0], intensityMeanV]))

        # make corrected values accessible
        self.kerV = interpolationKerV
        self.intensityKerV = interpolatedIntensityRadialDistanceC(
            interpolationKerV)

        # calculate distribution versus absolute momentum of one ion
        # due to momentum conservation both ions should have the same energy
        # therefore we have to divide the energy axis by two and rescale due
        # to p^2->p
        ionMassS = 14.*1.67262178e-27/9.10938291e-31
        raw_momentum = np.sqrt(self.kerV * ionMassS)
        raw_momentum_intensity = self.intensityKerV * raw_momentum
        interp = interpolate.interp1d(raw_momentum, raw_momentum_intensity)
        self.momentumV = np.linspace(0, raw_momentum.max(), len(raw_momentum))
        self.intensityAbsoluteMomentumV = interp(self.momentumV)
        self.intensityAbsoluteMomentumV[
            np.isnan(self.intensityAbsoluteMomentumV)] = 0.
        self.intensityAbsoluteMomentumV /= (
            self.intensityAbsoluteMomentumV.max())


def abelTransform(r, I):
    # calculate distribution against radial momentum(abel transform)
    A = np.empty_like(I)
    print len(A), len(I)
    for i in range(len(A)-2):
        A[i] = integrate.trapz(2 * I[i+1:] * r[i+1:] /
                               np.sqrt(r[i+1:]*r[i+1:] - r[i]*r[i]), r[i+1:])
    A[-1] = 0
    return A


def abelInvert(R, A, samplingS):
    # save the position data
    A /= A.max()

    # interpolate on requested grid
    interp = interpolate.interp1d(R, A)
    R = np.linspace(np.min(R), np.max(R), samplingS)
    A = interp(R)

    # fourier part
    # Make the empty result vector
    ft_A = np.zeros_like(R)
    # Iterate over all the radial positions
    for i, r in enumerate(R):
        # Create the integrand
        integrand = A * np.cos(R * r * 2 * np.pi)
        # Perform the integration
        ft_A[i] = np.trapz(integrand, R)

    # hankel part
    I = np.zeros_like((R))
    for i in range(len(R)):
        prS = R[i]
        I[i] = np.trapz(
            R * special.j0(prS * R * (2*np.pi)) * ft_A, R)

    return I


##################
# script start
if __name__ == '__main__':
    # load comparison data
    kC = kerCalibration('n2KerLundqvist.csv')

    plt.figure('Reference data')
    plt.clf()
    # plot raw data
    if 1:
        plt.plot(kC.kerRawV, kC.intensityRawV, 'y', label='raw data')
        plt.legend(loc='best')
        plt.xlabel('kinetic energy [eV]')

    if 1:
#        kC.abelInvert(kC.radialEnergyV, kC.intensityRadialEnergyV, 3000)
        plt.plot(kC.kerV, kC.intensityKerV, label='moving average')
#        plt.plot(kC.radialEnergyV, kC.intensityRadialEnergyV,
#                 label='abel transform')
#        plt.plot(kC.radialDistanceV, kC.intensityUnscaledMomentumV,
#                 label='back transform')
        plt.xlabel('ker [eV]')
#        plt.ylim([0, 1.3])
        plt.legend(loc='best')

    # test abel inversion
    abel_momentum = abelTransform(kC.momentumV, kC.intensityAbsoluteMomentumV)
    abel_inverted = abelInvert(kC.momentumV, abel_momentum, 3000)
    plt.figure('Momentum - Abel validation')
    plt.clf()
    plt.plot(kC.momentumV,
             kC.intensityAbsoluteMomentumV,
             label='reference(lundqvist)')
    plt.plot(kC.momentumV,
             abel_momentum,
             label='abel transform')
    plt.plot(kC.momentumV, abel_inverted, label='back transform')
    plt.xlabel('abs momentum release[arb. u.]')
#    plt.ylim([0, 1.3])
    plt.legend()

    
    # if doing this, rescaling accoring to r*sin(theta) has to be done after
    # the line wise transform (abel assumes cylindrical symmetry of which
    # spherical is a special case)
    # if we are dealing with spherical symmetry, theta is arbitrary and can
    # be averaged for better statistics
    A = rth_img[:,:].sum(axis=0) / r_axis_mm**2
    radialDistanceV = r_axis_mm

    # abelinvert takes x, y, resolution(higher=more),
    # works better on equally spaced data
    kC.abelInvert(radialDistanceV,
                  ndimage.filters.gaussian_filter(intensityRadialDistanceV, 1),
                  3000)
    plt.figure('data')
    plt.clf()
    plt.plot(radialDistanceV / np.max(radialDistanceV),
             intensityRadialDistanceV / np.max(intensityRadialDistanceV))
    plt.plot(kC.radialDistanceV / np.max(kC.radialDistanceV),
             (kC.intensityUnscaledMomentumV * kC.radialDistanceV) /
             np.max(kC.intensityUnscaledMomentumV * kC.radialDistanceV))
    plt.xlabel('momentum release[arb. u.]')

    plt.legend(['radial(direct data)', 'abel transform*r'])
