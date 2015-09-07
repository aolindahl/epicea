# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""

import numpy as np
import lmfit
import h5py
import time

import _electron_calibration_helper as _helper
from progress import update_progress


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
        self._weights_params_list = []
        self._weight_interpolation_list = []

        # Easier acess to the data
        data = np.array(self._data)

        self.intensity_scale_factors = 1.0 / np.mean(data[..., 3], axis=0)

        n_theta = len(self._theta)
        # Iterate through all the angle bins
        for idx in range(n_theta):
            # Add a parameters object to the energy params list
            self._energy_params_list.append(
                _helper.line_start_params([325] + [0]*(poly_order)))

            # Fit a polynomial to the datapoints in energy
            # Thus filling the above inserted parameters with reall vallues
            lmfit.minimize(_helper.poly_line,
                           self._energy_params_list[idx],
                           args=(data[:, idx, 0], data[:, idx, 1],
                                 data[:, idx, 2]))

            # Add parameters to the error listr
            self._error_params_list.append(
                _helper.line_start_params([0]*(1+poly_order)))
            # and fit to populate them
            lmfit.minimize(_helper.poly_line,
                           self._error_params_list[idx],
                           args=(data[:, idx, 0], data[:, idx, 2]))

            # Add parameters to the wieghts list
            self._weights_params_list.append(
                _helper.line_start_params([1] + [0]*(poly_order)))
            # and populate the parameters through a fit
            lmfit.minimize(_helper.poly_line,
                           self._weights_params_list[idx],
                           args=(data[:, idx, 0],
                                 data[:, idx, 3] *
                                 self.intensity_scale_factors[idx],
                                 data[:, idx, 2]))
#            self._weight_interpolation_list.append(
#                interp1d(data[:, idx, 0], data[:, idx, 3],
#                         kind='slinear', bounds_error=False))

            update_progress(idx, n_theta, verbose=verbose)

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
        weights = np.ones_like(radii) * np.nan

        idx_list = np.round(
            (angles - self._theta[0]) / self._d_theta).astype(int)
#        idx_list[idx_list < 0] = 0
#        idx_list[idx_list >= self._n_theta] = self._n_theta - 1

        for i in range(len(self._theta)):
            theta_idx_mask = idx_list == i

            energies[theta_idx_mask] = _helper.poly_line(
                self._energy_params_list[i], radii[theta_idx_mask])

            errors[theta_idx_mask] = _helper.poly_line(
                self._error_params_list[i], radii[theta_idx_mask])

#         for idx, r in zip(idx_list, radii):
#            energies.append(poly_line(self._energy_params_list[idx], r))
#            errors.append(poly_line(self._error_params_list[idx], r))

            weights[theta_idx_mask] = _helper.poly_line(
                self._weights_params_list[i], radii[theta_idx_mask])
#            weights[theta_idx_mask] = self._weight_interpolation_list[i](
#                radii[theta_idx_mask])

        invalid_energy = ((energies < self._energy_min) |
                          (self._energy_max < energies))

        for data in [energies, errors, weights]:
            data[invalid_energy] = np.nan

        return energies, errors, weights

    def get_data_copy(self):
        data = np.array(self._data)
        data[..., 3] *= self.intensity_scale_factors
        return np.array(self._theta), data

    _PROP_NAME_LIST = ['value', 'stderr']

    def save_to_file(self, file_name):
        with h5py.File(file_name, 'w') as file_ref:
            file_ref.create_dataset('theta', data=self._theta)
            file_ref.create_dataset('data', data=np.array(self._data))
            file_ref.create_dataset('intensity_scale_factors',
                                    data=self.intensity_scale_factors)

            file_ref.create_dataset('energy_min', data=self._energy_min)
            file_ref.create_dataset('energy_max', data=self._energy_max)

            energy_group = file_ref.create_group('energy_params')
            error_group = file_ref.create_group('error_params')
            weight_group = file_ref.create_group('weigt_params')
            for group, params in zip([energy_group, error_group, weight_group],
                                     [self._energy_params_list,
                                      self._error_params_list,
                                      self._weights_params_list]):
                for a_i in params[0].keys():
                    a_group = group.create_group(a_i)
                    for prop_name in self._PROP_NAME_LIST:
                        a_group.create_dataset(
                            prop_name,
                            data=[getattr(par[a_i], prop_name) for
                                  par in params])
            file_ref.attrs['time_stamp'] = self.conversion_time_stamp
            file_ref.attrs['data_sum'] = self._data_sum

    def load_from_file(self, file_name):
        self.reset()

        with h5py.File(file_name, 'r') as file_ref:
            self._set_theta(file_ref['theta'][:])
            self._data = [d for d in file_ref['data'].value]
            self.intensity_scale_factors = \
                file_ref['intensity_scale_factors'].value
            self._energy_min = file_ref['energy_min'].value
            self._energy_max = file_ref['energy_max'].value
            self._energy_params_list = []
            self._error_params_list = []
            self._weights_params_list = []
            energy_group = file_ref['energy_params']
            error_group = file_ref['error_params']
            weight_group = file_ref['weigt_params']
            for group, params in zip([energy_group, error_group, weight_group],
                                     [self._energy_params_list,
                                      self._error_params_list,
                                      self._weights_params_list]):
                for i in range(self._n_theta):
                    params.append(lmfit.Parameters())
                    for a_i in group.keys():
                        a_group = group[a_i]
                        params[-1].add(a_i)
                        for prop_name in self._PROP_NAME_LIST:
                            setattr(params[-1][a_i], prop_name,
                                    a_group[prop_name][i])
            if 'time_stamp' in file_ref.attrs.keys():
                self.conversion_time_stamp = file_ref.attrs['time_stamp']
                self._data_sum = file_ref.attrs['data_sum']
            else:
                self.conversion_time_stamp = 0
                self._data_sum = np.nan

    def create_or_load_conversion(self, file_name, compare_time_stmp,
                                  poly_order=2, verbose=False):
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
                print 'Calibration on file to old,',
                print 'or data sum does not match. Make new one calibration.'
            self.create_conversion(poly_order=poly_order)
        else:
            if verbose:
                print 'Loading old calibration since it is not that old :-)',
                print 'and the data sum match.'
            self.load_from_file(file_name)
