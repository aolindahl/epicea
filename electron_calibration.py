# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import lmfit
import h5py

import plt_func
if 'epicea' not in locals():
    import epicea_hdf5 as epicea


def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x-center)**2 / (2 * width**2))


def start_params(x=None, y=None, params_in=None):
    params = lmfit.Parameters()

    params.add('amplitude_1', 20, min=0)
    params.add('center_1', value=15)
    params.add('width_1', value=0.4, min=0)
    params.add('amp_ratio', value=1, min=0)
    params.add('amplitude_2', expr='amplitude_1 * amp_ratio')
    params.add('center_diff', value=1, min=0.5)
    params.add('center_2', expr='center_1 + center_diff')
    params.add('width_2', value=0.4, min=0)

    if (x is not None) and (y is not None):
        max_idx = np.argmax(y)
        y_max = y[max_idx]
        params['amplitude_1'].value = y_max
        params['center_1'].value = x[max_idx]
        for i in range(max_idx):
            if y[i] > float(y_max) / 20:
                params['center_1'].min = x[i]
                break
        for i in range(len(x)-1, max_idx-1, -1):
            if y[i] > float(y_max) / 20:
                params['center_diff'].max = x[i] - params['center_1'].min
                params['center_diff'].value = params['center_diff'].max / 2
                break

    if isinstance(params_in, lmfit.parameter.Parameters):
        for k in params_in:
            if k in params:
                params[k].value = params_in[k].value

    if isinstance(params_in, list):
        if np.all([isinstance(par, lmfit.parameter.Parameters) for
                  par in params_in]):
            for k in params:
                params[k].value = np.average(
                    [par[k].value for par in params_in if par[k].stderr != 0],
                    weights=[1./par[k].stderr for par in params_in
                             if par[k].stderr != 0])

    return params


def double_line_fit_model(params, x, data=None, eps_data=None):
    amplitude_1 = params['amplitude_1'].value
    center_1 = params['center_1'].value
    width_1 = params['width_1'].value
    amplitude_2 = params['amplitude_2'].value
    center_2 = params['center_2'].value
    width_2 = params['width_2'].value

    model = gaussian(x, amplitude_1, center_1, width_1) * x / center_1
    model += gaussian(x, amplitude_2, center_2, width_2) * x / center_2

    if data is None:
        return model
    if eps_data is None:
        return model - data
    return (model - data) / eps_data


def poly_line(params, x, y=None, err=None):
    mod = np.zeros_like(x)
    for i in range(len(params)):
        mod += params['a_{}'.format(i)] * x**i
    if y is None:
        return mod
    if err is None:
        return mod - y
    return (mod - y) / err


def line_start_params(a_list):
    if not isinstance(a_list, list):
        raise TypeError('Function linse_start_params expected a list' +
                        ' as input parameter, got {}.'.format(type(a_list)))

    params = lmfit.Parameters()
    for i, a in enumerate(a_list):
        params.add('a_{}'.format(i), value=a)

    return params


def find_kr_lines(rth_image, r_axis_mm, th_axis_rad,
                  return_line_params_list=False):
    # Fit to r projection to be used as initial parameters
    r_projection = rth_image.mean(axis=0)
    params_r_proj = start_params(x=r_axis_mm, y=r_projection)
    lmfit.minimize(double_line_fit_model, params_r_proj,
                   args=(r_axis_mm, r_projection))

    # Fit for each line based on the r projection fit
    line_params_list = []
    line_results_list = []
    for i in range(len(th_axis_rad)):
        line_params_list.append(start_params(
            x=r_axis_mm, y=rth_image[i, :]))
        line_results_list.append(
            lmfit.minimize(double_line_fit_model,
                           line_params_list[i],
                           args=(r_axis_mm, rth_image[i, :])))

#     Go trough once more
    num_to_average = len(th_axis_rad)/20
    if num_to_average == 0:
        num_to_average = 1
    for i in range(len(th_axis_rad)):
        selected_par_list = line_params_list[np.maximum(i-num_to_average, 0):
                                             i+num_to_average+1]
        if i < num_to_average:
            selected_par_list.extend(line_params_list[-num_to_average+i:])
        if num_to_average < len(line_params_list) - i:
            selected_par_list.extend(
                line_params_list[:num_to_average -
                                 len(line_params_list) + i + 1])

        line_params_list[i] = start_params(x=r_axis_mm, y=rth_image[i, :],
                                           params_in=selected_par_list)
        line_results_list[i] = lmfit.minimize(
            double_line_fit_model,
            line_params_list[i],
            args=(r_axis_mm, rth_image[i, :]))

    # Get the results in a nicer format
    r_1 = np.empty_like(line_params_list, dtype=float)
    r_2 = np.empty_like(r_1)
    w_1 = np.empty_like(r_1)
    w_2 = np.empty_like(r_1)
    red_chi2 = np.empty_like(r_1)
    for i in range(len(line_params_list)):
        r_1[i] = line_params_list[i]['center_1'].value
        w_1[i] = (line_params_list[i]['center_1'].stderr)
#                  line_results_list[i].redchi)
#        w_1[i] = line_params['width_1'].value
        r_2[i] = line_params_list[i]['center_2'].value
        w_2[i] = (line_params_list[i]['center_2'].stderr)
#                  line_results_list[i].redchi)
#        w_2[i] = line_params['width_2'].value
        red_chi2[i] = line_results_list[i].redchi

    w_1[w_1 <= 0] = np.inf
    w_2[w_2 <= 0] = np.inf
    if return_line_params_list:
        return r_1, w_1, r_2, w_2, red_chi2, line_params_list
    return r_1, w_1, r_2, w_2, red_chi2


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

    def add_calibration_data(self, radii, angles, energy, errors):
        if len(radii) != len(angles):
            raise IndexError(
                '"radii" and "angles" do not have the same length')

        if self._theta is None:
            self._set_theta(angles)
            self._data = []
        elif (self._theta != angles).any():
            raise ValueError('The same angles must be used for all data.')

        self._data.append(np.empty((self._n_theta, 3)))
        self._data[-1][:, 0] = radii
        self._data[-1][:, 1] = energy
        self._data[-1][:, 2] = errors

    def create_conversion(self, poly_order=1):
        self._energy_params_list = []
        self._error_params_list = []
        data = np.array(self._data)
        for idx in range(len(self._theta)):
            self._energy_params_list.append(
                line_start_params([0]*(1+poly_order)))
            lmfit.minimize(poly_line,
                           self._energy_params_list[idx],
                           args=(data[:, idx, 0], data[:, idx, 1],
                                 data[:, idx, 2]))
            self._error_params_list.append(
                line_start_params([0]*(1+poly_order)))
            lmfit.minimize(poly_line,
                           self._error_params_list[idx],
                           args=(data[:, idx, 0], data[:, idx, 2]))

    def get_energies(self, radii, angles):
        if self._energy_params_list is None:
            return None, None

        energies = np.empty_like(radii)
        errors = np.empty_like(radii)

        idx_list = np.round(
            (angles - self._theta[0]) / self._d_theta).astype(int)
        idx_list[idx_list < 0] = 0
        idx_list[idx_list >= self._n_theta] = self._n_theta - 1

        for i in range(len(self._theta)):
            theta_idx_mask = idx_list == i
            energies[theta_idx_mask] = poly_line(self._energy_params_list[i],
                                                 radii[theta_idx_mask])
            errors[theta_idx_mask] = poly_line(self._error_params_list[i],
                                               radii[theta_idx_mask])
#         for idx, r in zip(idx_list, radii):
#            energies.append(poly_line(self._energy_params_list[idx], r))
#            errors.append(poly_line(self._error_params_list[idx], r))

        return energies, errors

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

if __name__ == '__main__':
    h5_base_path = 'h5_data'
    raw_data_base_path = '../data/ExportedData'
    kr_binding_energies = np.array([93.788, 95.038])
    for path in [h5_base_path, raw_data_base_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    if 'calib_data_list' not in locals():
        calib_data_list = epicea.DataSetList()
        photon_energy_dict = {}

    verbose = True

    # Add the data sets to the list.
    # And store the corresponding photon energy values
    #calib_data_list.add_dataset(
    #    name='Kr_470_7_eV',
    #    h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_470_7_eV.h5'),
    #    raw_data_path=os.path.join(raw_data_base_path,
    #                               'CalibrationsForKE373eV/Kr_0023'),
    #    verbose=verbose)
    #calib_data_list.add_dataset(
    #    name='Kr_469_7_eV',
    #    h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_469_7_eV.h5'),
    #    raw_data_path=os.path.join(raw_data_base_path,
    #                               'CalibrationsForKE373eV/Kr_0024'),
    #    verbose=verbose)

    calib_data_list.add_dataset(
        name='Kr_468_7_eV',
        h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_468_7_eV.h5'),
        raw_data_path=os.path.join(raw_data_base_path,
                                   'CalibrationsForKE373eV/Kr_0022'),
        verbose=verbose)
    photon_energy_dict['Kr_468_7_eV'] = 469.0

    calib_data_list.add_dataset(
        name='Kr_466_7_eV',
        raw_data_path=os.path.join(raw_data_base_path,
                                   'CalibrationsForKE373eV/Kr_calib_00160021'),
        h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_466_7_eV.h5'),
        verbose=verbose)
    photon_energy_dict['Kr_466_7_eV'] = 467.0

    calib_data_list.add_dataset(
        name='Kr_464_7_eV',
        h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_464_7_eV.h5'),
        raw_data_path=os.path.join(raw_data_base_path,
                                   'CalibrationsForKE373eV/Kr_calib_00160020'),
        verbose=verbose)
    photon_energy_dict['Kr_464_7_eV'] = 465.0

    calib_data_list.add_dataset(
        name='Kr_462_7_eV',
        h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_462_7_eV.h5'),
        raw_data_path=os.path.join(raw_data_base_path,
                                   'CalibrationsForKE373eV/Kr_calib_00150019'),
        verbose=verbose)
    photon_energy_dict['Kr_462_7_eV'] = 463.0

    calib_data_list.add_dataset(
        name='Kr_461_7_eV',
        h5_path=os.path.join(h5_base_path, 'center_373_eV_Kr_461_7_eV.h5'),
        raw_data_path=os.path.join(raw_data_base_path,
                                   'CalibrationsForKE373eV/Kr_0025'),
        verbose=verbose)
    photon_energy_dict['Kr_461_7_eV'] = 462.0

    calib_data_list.sort(reverse=True)

    calibration = PositionToEnergyCalibration()

    # Need to be done only once
    #for c_data in calib_data_list:
    #    c_data.electrons.recalculate_polar_coordinates()

    print 'Plot all the raw spectra.'
    x_axis_mm = np.linspace(-20, 20, 512)
    plt_func.figure_wrapper('Kr e spectra')
    n_subplots = calib_data_list.len()
    n_rows = np.floor(np.sqrt(n_subplots))
    n_cols = np.ceil(float(n_subplots)/n_rows)
    for i, c_data in enumerate(calib_data_list):
        plt.subplot(n_rows, n_cols, i+1)
        plt_func.imshow_wrapper(c_data.get_e_xy_image(x_axis_mm), x_axis_mm)
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Position (mm)')

    r_axis_mm = np.linspace(0, 25, 257)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 257)[1::2]

    for i, c_data in enumerate(calib_data_list):
        print 'Make the theta-r spectrum for {}.'.format(c_data.name())
        plt_func.figure_wrapper(
            'Kr e spectra theta-r {}'.format(c_data.name()))
        e_rth_image = c_data.get_e_rth_image(r_axis_mm, th_axis_rad)
        plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
                                kw_args={'aspect': 'auto'})
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Angle (rad)')

        print 'Find the Kr 3d lines in {}.'.format(c_data.name())
        r_1, w_1, r_2, w_2, red_chi2 = find_kr_lines(e_rth_image,
                                                     r_axis_mm,
                                                     th_axis_rad)
        plt.errorbar(r_1, th_axis_rad, xerr=w_1, fmt='.r', capsize=0)
        plt.errorbar(r_2, th_axis_rad, xerr=w_2, fmt='.m', capsize=0)

    #    poly_order = 40
    #    l1_params = line_start_params([0]*poly_order)
    #    lmfit.minimize(poly_line, l1_params, args=(th_axis_rad, r_1, w_1))
    #    plt.plot(poly_line(l1_params, th_axis_rad,), th_axis_rad, 'k')
    #    l2_params = line_start_params([0]*poly_order)
    #    out = lmfit.minimize(poly_line, l2_params, args=(th_axis_rad, r_2, w_2))
    #    plt.plot(poly_line(l2_params, th_axis_rad), th_axis_rad, 'k')

        print 'Lineoiuts in waterfall plot.'
        plt_func.figure_wrapper('lineouts {}'.format(c_data.name()))
        for i, line in enumerate(e_rth_image):
            plt.plot(r_axis_mm, line + i*20, 'b')

        print 'Plot reduced chi^2.'
        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
        plt.plot(red_chi2, th_axis_rad)

        print 'Add data to the calibration object'
        calibration.add_calibration_data(
            r_1, th_axis_rad,
            photon_energy_dict[c_data.name()] - kr_binding_energies[0],
            red_chi2)
        calibration.add_calibration_data(
            r_2, th_axis_rad,
            photon_energy_dict[c_data.name()] - kr_binding_energies[1],
            red_chi2)

    print 'Create calibration'
    calibration.create_conversion(poly_order=2)

    print 'Check the calibration'
    plt_func.figure_wrapper('Calib data check')
    theta_list, data_list = calibration.get_data_copy()
    r_axis_for_calib_check_mm = np.linspace(data_list[:, :, 0].min() * 0.8,
                                            data_list[:, :, 0].max() * 1.2,
                                            256)
    for idx in range(len(theta_list)):
        plt.plot(data_list[:, idx,  0], data_list[:, idx, 1], '.b')
        plt.plot(r_axis_for_calib_check_mm,
                 poly_line(calibration._energy_params_list[idx],
                           r_axis_for_calib_check_mm), 'r')

    E_axis_eV = np.linspace(360, 380, 257)[1::2]
    E_all = []
    err_all = []
    theta_all = []
    for c_data in calib_data_list:
        print 'Get the calibrated energies.'
        E, err = calibration.get_energies(c_data.electrons.pos_r,
                                          c_data.electrons.pos_t)
        E_all.extend(E)
        err_all.extend(err)
        theta_all.extend(c_data.electrons.pos_t)

    fig = plt_func.figure_wrapper('Energy domain all calibration data')
    E_image = epicea.center_histogram_2d(E_all, theta_all,
                                         E_axis_eV, th_axis_rad)
    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
                            kw_args={'aspect': 'auto'})

    calibration.save_to_file('test_data/calib_373.h5')
