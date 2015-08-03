# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""

import numpy as np
import lmfit
import h5py

import _electron_calibration_helper as _helper


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

    def create_conversion(self, poly_order=1):
        # Maka some parameter lists, there will be one element for each angle
        self._energy_params_list = []
        self._error_params_list = []
        self._weights_params_list = []

        # Easier acess to the data
        data = np.array(self._data)

        # Iterate through all the angle bins
        for idx in range(len(self._theta)):
            # Add a parameters object to the energy params list
            self._energy_params_list.append(
                _helper.line_start_params([0]*(1+poly_order)))

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
                _helper.line_start_params([0]*(1+poly_order)))
            # and populate the parameters through a fit
            lmfit.minimize(_helper.poly_line,
                           self._weights_params_list[idx],
                           args=(data[:, idx, 0], data[:, idx, 3]))

    def get_energies(self, radii, angles):
        if self._energy_params_list is None:
            return None, None

        if isinstance(radii, h5py.Dataset):
            radii = radii.value
        if isinstance(angles, h5py.Dataset):
            angles = angles.value

        energies = np.empty_like(radii)
        errors = np.empty_like(radii)
        weights = np.empty_like(radii)

        idx_list = np.round(
            (angles - self._theta[0]) / self._d_theta).astype(int)
        idx_list[idx_list < 0] = 0
        idx_list[idx_list >= self._n_theta] = self._n_theta - 1

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

        return energies, errors, weights

    def get_data_copy(self):
        return np.array(self._theta), np.array(self._data)

    _PROP_NAME_LIST = ['value', 'stderr']

    def save_to_file(self, file_name):
        with h5py.File(file_name, 'w') as file_ref:
            file_ref.create_dataset('theta', data=self._theta)

            energy_group = file_ref.create_group('energy_params')
            error_group = file_ref.create_group('error_params')
            for group, params in zip([energy_group, error_group],
                                     [self._energy_params_list,
                                      self._error_params_list]):
                for a_i in params[0].keys():
                    a_group = group.create_group(a_i)
                    for prop_name in self._PROP_NAME_LIST:
                        a_group.create_dataset(
                            prop_name,
                            data=[getattr(par[a_i], prop_name) for
                                  par in params])

    def load_from_file(self, file_name):
        self.reset()

        with h5py.File(file_name, 'r') as file_ref:
            self._set_theta(file_ref['theta'][:])
            self._energy_params_list = []
            self._error_params_list = []
            energy_group = file_ref['energy_params']
            error_group = file_ref['error_params']
            for group, params in zip([energy_group, error_group],
                                     [self._energy_params_list,
                                      self._error_params_list]):
                for i in range(self._n_theta):
                    params.append(lmfit.Parameters())
                    for a_i in group.keys():
                        a_group = group[a_i]
                        params[-1].add(a_i)
                        for prop_name in self._PROP_NAME_LIST:
                            setattr(params[-1][a_i], prop_name,
                                    a_group[prop_name][i])
