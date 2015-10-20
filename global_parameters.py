# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:19:37 2015

@author: antlin
"""

import numpy as np

ION_VMI_OFFSET = {}
ION_VMI_OFFSET['430_373'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['412_373'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['430_366'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['412_366'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['430_357'] = np.array([-0.5, -0.1])
ION_VMI_OFFSET['412_357'] = np.array([-0.5, -0.1])
ION_VMI_OFFSET['560_500'] = np.array([1.8, -0.1])

ELECTRON_OFFSET = {}
ELECTRON_OFFSET['430_373'] = np.array([0, 0])
ELECTRON_OFFSET['412_373'] = np.array([0, 0])
ELECTRON_OFFSET['430_366'] = np.array([0, 0])
ELECTRON_OFFSET['412_366'] = np.array([0, 0])
ELECTRON_OFFSET['430_357'] = np.array([0, 0])
ELECTRON_OFFSET['412_357'] = np.array([0, 0])
ELECTRON_OFFSET['560_500'] = np.array([0, 0])

NO_N_TIME_SUM_RANGE_US = {}
NN_O_TIME_SUM_RANGE_US = {}
NO_N_TIME_SUM_RANGE_US['430_373'] = np.array([8.59, 8.63])
NN_O_TIME_SUM_RANGE_US['430_373'] = np.array([8.65, 8.70])
NO_N_TIME_SUM_RANGE_US['412_373'] = np.array([8.59, 8.63])
NN_O_TIME_SUM_RANGE_US['412_373'] = np.array([8.65, 8.70])
NO_N_TIME_SUM_RANGE_US['430_366'] = np.array([8.57, 8.61])
NN_O_TIME_SUM_RANGE_US['430_366'] = np.array([8.625, 8.675])
NO_N_TIME_SUM_RANGE_US['412_366'] = np.array([8.57, 8.61])
NN_O_TIME_SUM_RANGE_US['412_366'] = np.array([8.625, 8.675])
NO_N_TIME_SUM_RANGE_US['430_357'] = np.array([8.585, 8.625])
NN_O_TIME_SUM_RANGE_US['430_357'] = np.array([8.645, 8.695])
NO_N_TIME_SUM_RANGE_US['412_357'] = np.array([8.59, 8.63])
NN_O_TIME_SUM_RANGE_US['412_357'] = np.array([8.65, 8.70])
NO_N_TIME_SUM_RANGE_US['560_500'] = np.array([8.57, 8.61])
NN_O_TIME_SUM_RANGE_US['560_500'] = np.array([8.625, 8.675])

NO_N_TIME_DIFF_RANGE_US = {}
NN_O_TIME_DIFF_RANGE_US = {}
NO_N_TIME_DIFF_RANGE_US['430_373'] = np.array([-1.8, -1.27])
NN_O_TIME_DIFF_RANGE_US['430_373'] = np.array([-1.39, -0.82])
NO_N_TIME_DIFF_RANGE_US['412_373'] = np.array([-1.8, -1.27])
NN_O_TIME_DIFF_RANGE_US['412_373'] = np.array([-1.37, -0.86])
NO_N_TIME_DIFF_RANGE_US['430_366'] = np.array([-1.8, -1.27])
NN_O_TIME_DIFF_RANGE_US['430_366'] = np.array([-1.39, -0.82])
NO_N_TIME_DIFF_RANGE_US['412_366'] = np.array([-1.8, -1.27])
NN_O_TIME_DIFF_RANGE_US['412_366'] = np.array([-1.38, -0.84])
NO_N_TIME_DIFF_RANGE_US['430_357'] = np.array([-1.81, -1.27])
NN_O_TIME_DIFF_RANGE_US['430_357'] = np.array([-1.39, -0.82])
NO_N_TIME_DIFF_RANGE_US['412_357'] = np.array([-1.81, -1.27])
NN_O_TIME_DIFF_RANGE_US['412_357'] = np.array([-1.39, -0.82])
NO_N_TIME_DIFF_RANGE_US['560_500'] = np.array([-1.8, -1.27])
NN_O_TIME_DIFF_RANGE_US['560_500'] = np.array([-1.42, -0.82])

# Make an x axis
x_axis_mm = np.linspace(-23, 23, 2**8)

# Define polar coordinate axis vectors
r_axis_mm = np.linspace(0, 25, 2**9+1)[1::2]
th_axis_rad = np.linspace(0, 2*np.pi, 2**8+1)[1::2]
#th_axis_rad = np.linspace(2, 2.8, 2**7+1)[1::2]
