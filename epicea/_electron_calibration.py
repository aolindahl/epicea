# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""

import numpy as np
import lmfit
import h5py
import time

from . import _electron_calibration_helper as _helper
from . progress import update_progress

_change_time = 1444808450

class PositionToEnergyCalibration(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._theta = None
        self._data = None
        self._energy_params_list = None
        self._error_params_list = None

    def _set_theta(self, theta):
        self._theta = theta
        self._n_theta = len(theta)
        self._d_theta = np.diff(theta).mean()

    def add_calibration_data(self, radii, angles, energy, errors,
                             relative_intensity):
        if len(radii) != len(angles):
            raise IndexError(
                '"radii" and "angles" do not have the same length')

        if self._theta is None:
            self._set_theta(angles)
            self._data = []
        elif (self._theta != angles).any():
            raise ValueError('The same angles must be used for all data.')

        self._data.append(np.empty((self._n_theta, 4)))
        self._data[-1][:, 0] = radii
        self._data[-1][:, 1] = energy
        self._data[-1][:, 2] = errors
        self._data[-1][:, 3] = relative_intensity

    def create_conversion(self, poly_order=1, verbose=False):
        # Maka some parameter lists, there will be one element for each angle
        self._energy_params_list = []
        self._error_params_list = []
#        self._weights_params_list = []
#        self._weight_interpolation_list = []

        # Easier acess to the data
        data = np.array(self._data)

#        self.intensity_scale_factors = 1.0 / np.mean(data[..., 3], axis=0)

        n_theta = len(self._theta)
        
        # Get some start parameters
        r_to_e_start_params = _helper.r_to_e_conversion_start_params()
        error_start_params = _helper.line_start_params([0]*(1+poly_order))
        # Iterate through all the angle bins
        for idx in range(n_theta):

            # Fit the energy function
            r_to_e_result = lmfit.minimize(_helper.r_to_e_conversion,
                                           r_to_e_start_params,
                                           args=(data[:, idx, 0],
                                                 data[:, idx, 1]))

            # Add a parameters object to the energy params list
            self._energy_params_list.append(r_to_e_result.params)
            
            # Fit the error function
            error_result = lmfit.minimize(_helper.poly_line,
                                          error_start_params,
                                          args=(data[:, idx, 0],
                                                data[:, idx, 2]))
            # Add parameters to the error listr
            self._error_params_list.append(error_result.params)

            update_progress(idx, n_theta, verbose=verbose)

        
        weights_start_params = _helper.line_start_params(
            [1] + [0]*(poly_order))
        a_sum = np.nansum(data[:, :, 3], axis=1)
        I = np.isfinite(a_sum)
        weights_result = lmfit.minimize(_helper.poly_line,
                                        weights_start_params,
                                        args=(data[I, :, 1].mean(1),
                                              1. / a_sum[I]))

        self._weights_params = weights_result.params

        self.conversion_time_stamp = time.time()
        self._data_sum = np.sum(self._data)
        self._energy_min = data[..., 1].min()
        self._energy_max = data[..., 1].max()

    def get_energies(self, radii, angles):
        if self._energy_params_list is None:
            return None, None

        if isinstance(radii, h5py.Dataset):
            radii = radii.value
        if isinstance(angles, h5py.Dataset):
            angles = angles.value

        energies = np.ones_like(radii) * np.nan
        errors = np.ones_like(radii) * np.nan
#        weights = np.ones_like(radii) * np.nan

        idx_list = np.round(
            (angles - self._theta[0]) / self._d_theta).astype(int)
#        idx_list[idx_list < 0] = 0
#        idx_list[idx_list >= self._n_theta] = self._n_theta - 1

        for i in range(len(self._theta)):
            theta_idx_mask = idx_list == i

            e_funk = (_helper.poly_line if
                      'r0' not in self._energy_params_list[i] else
                      _helper.r_to_e_conversion)
            energies[theta_idx_mask] = e_funk(
                self._energy_params_list[i], radii[theta_idx_mask])

#            print('\n')
#            print(self._error_params_list[i])
            errors[theta_idx_mask] = _helper.poly_line(
                self._error_params_list[i], radii[theta_idx_mask])

#         for idx, r in zip(idx_list, radii):
#            energies.append(poly_line(self._energy_params_list[idx], r))
#            errors.append(poly_line(self._error_params_list[idx], r))

#            weights[theta_idx_mask] = _helper.poly_line(
#                self._weights_params_list[i], radii[theta_idx_mask])
#            weights[theta_idx_mask] = self._weight_interpolation_list[i](
#                radii[theta_idx_mask])

        weights = _helper.poly_line(self._weights_params, energies)

        try:
            invalid_energy = ((energies < self._energy_min) |
                              (self._energy_max < energies))
        except RuntimeWarning as warn:
            print(warn.args)
            print(warn.with_traceback)

#        for data in [energies, errors, weights]:
#            data[invalid_energy] = np.nan

        return energies, errors, weights

    def get_data_copy(self):
        data = np.array(self._data)
#        data[..., 3] *= self.intensity_scale_factors
        return np.array(self._theta), data

    _PROP_NAME_LIST = ['value', 'stderr']

    def save_to_file(self, file_name):
        with h5py.File(file_name, 'w') as file_ref:
            b_type = h5py.special_dtype(vlen=bytes)         
            
            file_ref.create_dataset('theta', data=self._theta)
            file_ref.create_dataset('data', data=np.array(self._data))
#            file_ref.create_dataset('intensity_scale_factors',
#                                    data=self.intensity_scale_factors)

            file_ref.create_dataset('energy_min', data=self._energy_min)
            file_ref.create_dataset('energy_max', data=self._energy_max)

            file_ref.create_dataset('weight_params',
                                    data=self._weights_params.dumps())

            energy_dset = file_ref.create_dataset(
                'energy_params',
                (len(self._energy_params_list), ),
                dtype=b_type)
            error_dset = file_ref.create_dataset(
                'error_params',
                (len(self._error_params_list), ),
                dtype=b_type)
            for dset, params_list in zip([energy_dset, error_dset],
                                         [self._energy_params_list,
                                          self._error_params_list]):
                for i, params in enumerate(params_list):
                    dset[i] = params.dumps()

            file_ref.attrs['time_stamp'] = self.conversion_time_stamp
            file_ref.attrs['data_sum'] = self._data_sum

    def load_from_file(self, file_name):
        self.reset()

        with h5py.File(file_name, 'r') as file_ref:
            self._set_theta(file_ref['theta'][:])
            self._data = [d for d in file_ref['data'].value]
#            self.intensity_scale_factors = \
#                file_ref['intensity_scale_factors'].value
            self._energy_min = file_ref['energy_min'].value
            self._energy_max = file_ref['energy_max'].value
            self._energy_params_list = []
            self._error_params_list = []

            self._weights_params = lmfit.Parameters()
            self._weights_params.loads(file_ref['weight_params'].value)

            energy_dset = file_ref['energy_params']
            error_dset = file_ref['error_params']
            for dset, params_list in zip([energy_dset, error_dset],
                                         [self._energy_params_list,
                                          self._error_params_list]):
                for i in range(self._n_theta):
                    params_list.append(lmfit.Parameters())
                    params_list[-1].loads(dset[i].decode('utf-8'))
            

            if 'time_stamp' in file_ref.attrs.keys():
                self.conversion_time_stamp = file_ref.attrs['time_stamp']
                self._data_sum = file_ref.attrs['data_sum']
            else:
                self.conversion_time_stamp = 0
                self._data_sum = np.nan

    def create_or_load_conversion(self, file_name, compare_time_stmp,
                                  poly_order=2, verbose=False):

        compare_time_stmp = max(compare_time_stmp, _change_time)    

        try:
            with h5py.File(file_name, 'r') as file_ref:
                file_time_stamp = file_ref.attrs['time_stamp']
        except:
            file_time_stamp = 0

        try:
            with h5py.File(file_name, 'r') as file_ref:
                file_data_sum = file_ref.attrs['data_sum']
        except:
            file_data_sum = 0

        self._data_sum = np.sum(self._data)

        if ((file_time_stamp < compare_time_stmp) or (file_data_sum !=
                                                      self._data_sum)):
            if verbose:
                print('Calibration on file to old,',
                      'or data sum does not match. Make new one calibration.')
            self.create_conversion(poly_order=poly_order)
        else:
            if verbose:
                print('Loading old calibration since it is not that old :-)',
                      'and the data sum match.')
            self.load_from_file(file_name)
