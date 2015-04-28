# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:06:29 2015

@author: Anton O Lindahl

This is a mudule made to treat conicedence data from the EPICEA setup at the
PLEIADES beamlina at the synchrotron SOLEIL.
"""

from _data_classes import DataSet, DataSetList
import _filter_functions as ff
from _testing import test_run as test
from _data_class_helper import center_histogram, center_histogram_2d
from _electron_calibration import PositionToEnergyCalibration as \
    ElectronEnergyCalibration
from _electron_calibration_helper import find_lines, poly_line
