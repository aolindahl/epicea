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
    """Make the electron calibration for a given center energy setting."""
    # The Kr binding energies are needed for Kr based calibrations
    kr_binding_energies = np.array([93.788, 95.038])
    n2_binidng_energy = np.array([409.9])

    # Get the list of data sets
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=verbose)
    except electron_calibration_data.CenterEnergyError as e:
        print e
        return
    # and make sure the list is sorted
    calib_data_list.sort(reverse=True)

    if 'Kr' in calib_data_list[0].name():
        gas = 'Kr'
        binding = kr_binding_energies
    elif 'N2' in calib_data_list[0].name():
        gas = 'N2'
        binding = n2_binidng_energy
    else:
        print 'Unknown gas.',
        print 'Dataset names must specify the used gas (N2 or Kr).'

    n_lines = len(binding)

    # Create an emptu calibration object
    calibration = epicea.ElectronEnergyCalibration()

    # Create a plot of all the spectra in the data list
    print 'Plot all the raw spectra.'
    # Make an x axis
    x_axis_mm = np.linspace(-23, 23, 512)
    # Create the figure
    plt_func.figure_wrapper('Calibration spectra {} eV'.format(setting))
    # One subplot for each data set...
    n_subplots = calib_data_list.len()
    # ... nicely set in collumns and rows
    n_rows = np.floor(np.sqrt(n_subplots))
    n_cols = np.ceil(float(n_subplots)/n_rows)
    # Iterate over the datasets
    for i, c_data in enumerate(calib_data_list):
        plt.subplot(n_rows, n_cols, i+1)  # Make the subplot
        # Show the electron figure
        plt_func.imshow_wrapper(c_data.get_e_xy_image(x_axis_mm), x_axis_mm)
        # Adjust the figure for good looks
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Position (mm)')

    plt.tight_layout()
    # Define polar coordinate axis vectors
    r_axis_mm = np.linspace(0, 25, 257)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 257)[1::2]

    # Iterate over the datasets and make images for each set.
    # Also find the lines in each of the spectra
    for i, c_data in enumerate(calib_data_list):
        print 'Make the theta-r spectrum for {}.'.format(c_data.name())
        # Make the figure
        plt_func.figure_wrapper(
            'e spectra theta-r {}'.format(c_data.name()))
        # Get the image fron the data in polar coordinates
        e_rth_image = c_data.get_e_rth_image(r_axis_mm, th_axis_rad)
        # Show the image
        plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
                                kw_args={'aspect': 'auto'})
        # Make the image look nice
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Angle (rad)')

        if gas == 'Kr':
            print 'Find the Kr 3d lines in {}.'.format(c_data.name())
        elif gas == 'N2':
            print 'Find the N_2 s1 line in {}.'.format(c_data.name())
        r, w, red_chi2 = epicea.find_lines(
            e_rth_image, r_axis_mm, th_axis_rad, n_lines=n_lines)
        fmts = ['.r', '.m', '.y']
        for line in range(n_lines):
            plt.errorbar(r[:, line], th_axis_rad, xerr=w[:, line],
                         fmt=fmts[line], capsize=1)
#        plt.errorbar(r_1, th_axis_rad, xerr=w_1, fmt='.r', capsize=0)
#        plt.errorbar(r_2, th_axis_rad, xerr=w_2, fmt='.m', capsize=0)

#        poly_order = 40
#        l1_params = line_start_params([0]*poly_order)
#        lmfit.minimize(poly_line, l1_params, args=(th_axis_rad, r_1, w_1))
#        plt.plot(poly_line(l1_params, th_axis_rad,), th_axis_rad, 'k')
#        l2_params = line_start_params([0]*poly_order)
#        out = lmfit.minimize(poly_line, l2_params,
#                             args=(th_axis_rad, r_2, w_2))
#        plt.plot(poly_line(l2_params, th_axis_rad), th_axis_rad, 'k')

#        print 'Lineoiuts in waterfall plot.'
#        plt_func.figure_wrapper('lineouts {}'.format(c_data.name()))
#        for i, line in enumerate(e_rth_image):
#            plt.plot(r_axis_mm, line + i*20, 'b')

        print 'Plot reduced chi^2.'
        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
        plt.plot(red_chi2, th_axis_rad)

        print 'Add data to the calibration object'
        for line in range(n_lines):
            calibration.add_calibration_data(
                r[:, line], th_axis_rad,
                c_data.photon_energy() - binding[line],
                w[:, line])

    print 'Create calibration'
    calibration.create_conversion(poly_order=1)

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

    E_axis_eV = np.linspace(setting-30, setting+20, 2**8+1)[1::2]
    E_all = []
    err_all = []
    theta_all = []
    for c_data in calib_data_list:
        print 'Get the calibrated energies, {} eV.'.format(
            c_data.photon_energy())
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
    for setting in [500]:
        print 'Procesing calibration data for the {} eV setting.'.format(
            setting)
        make_calibration(setting, verbose=True)
        raw_input('Press enter to do next calibration.')
