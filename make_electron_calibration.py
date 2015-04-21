# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""
import numpy as np
import matplotlib.pyplot as plt

import plt_func
import electron_calibration_data
if 'epicea' not in locals():
    import epicea


def make_calibration(setting, verbose=False):
    kr_binding_energies = np.array([93.788, 95.038])

    calib_data_list, photon_energy_dict = \
        electron_calibration_data.get_data_in_list(setting,
                                                   verbose=verbose)

    calib_data_list.sort(reverse=True)

    calibration = epicea.ElectronEnergyCalibration()

#     Need to be done only once
    for c_data in calib_data_list:
        c_data.electrons.recalculate_polar_coordinates()

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
        r_1, w_1, r_2, w_2, red_chi2 = epicea.find_kr_lines(
            e_rth_image, r_axis_mm, th_axis_rad)
        plt.errorbar(r_1, th_axis_rad, xerr=w_1, fmt='.r', capsize=0)
        plt.errorbar(r_2, th_axis_rad, xerr=w_2, fmt='.m', capsize=0)

#        poly_order = 40
#        l1_params = line_start_params([0]*poly_order)
#        lmfit.minimize(poly_line, l1_params, args=(th_axis_rad, r_1, w_1))
#        plt.plot(poly_line(l1_params, th_axis_rad,), th_axis_rad, 'k')
#        l2_params = line_start_params([0]*poly_order)
#        out = lmfit.minimize(poly_line, l2_params,
#                             args=(th_axis_rad, r_2, w_2))
#        plt.plot(poly_line(l2_params, th_axis_rad), th_axis_rad, 'k')

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
                 epicea.poly_line(calibration._energy_params_list[idx],
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

    plt_func.figure_wrapper('Energy domain all calibration data')
    E_image = epicea.center_histogram_2d(E_all, theta_all,
                                         E_axis_eV, th_axis_rad)
    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
                            kw_args={'aspect': 'auto'})

    calibration.save_to_file('h5_data/calib_{}.h5'.format(setting))


if __name__ == '__main__':
    for setting in ['373']:
        print 'Procesing calibration data for the {} eV setting.'.format(
            setting)
        make_calibration(setting, verbose=True)
        raw_input('Press enter to do next calibration.')
