# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

import plt_func
import electron_calibration_data
if 'epicea' not in locals():
    import epicea


def make_calibration(setting, verbose=False):
    """Make the electron calibration for a given center energy setting."""
    # The Kr binding energies are needed for Kr based calibrations
    kr_binding_energies = np.array([93.788, 95.038])
    kr_multiplicity = np.array([11., 9.])
    n2_binidng_energy = np.array([409.9])
    n2_multiplicity = np.array([1.])
    calibration_file_name = 'h5_data/calib_{}.h5'.format(setting)

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
        multiplicity = kr_multiplicity
    elif 'N2' in calib_data_list[0].name():
        gas = 'N2'
        binding = n2_binidng_energy
        multiplicity = n2_multiplicity
    else:
        print 'Unknown gas.',
        print 'Dataset names must specify the used gas (N2 or Kr).'

    n_lines = len(binding)

    # Create an empty calibration object
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
        plt_func.imshow_wrapper(c_data.get_e_xy_image(x_axis_mm)[0], x_axis_mm)
        # Adjust the figure for good looks
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Position (mm)')

    plt.tight_layout()
    # Define polar coordinate axis vectors
    r_axis_mm = np.linspace(0, 25, 2**8+1)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 2**8+1)[1::2]

    # Keep track of the latest time stamp of the lines
    latest_lines = 0

    # Iterate over the datasets and make images for each set.
    # Also find the lines in each of the spectra
    for i, c_data in enumerate(calib_data_list):
        print 'Make the theta-r spectrum for {}.'.format(c_data.name())
        stdout.flush()
        # Make the figure
        plt_func.figure_wrapper(
            'e spectra theta-r {}'.format(c_data.name()))
        # Get the image fron the data in polar coordinates
        e_rth_image, e_rth_image_time_stamp = \
            c_data.get_e_rth_image(r_axis_mm, th_axis_rad)
        # Show the image
        plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
                                kw_args={'aspect': 'auto'})
        # Make the image look nice
        plt_func.tick_fontsize()
        plt_func.title_wrapper(c_data.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Angle (rad)')
        plt_func.colorbar_wrapper()

        if gas == 'Kr':
            print 'Find the Kr 3d lines in {}.'.format(c_data.name())
            stdout.flush()
        elif gas == 'N2':
            print 'Find the N_2 s1 line in {}.'.format(c_data.name())
            stdout.flush()

        # Find the lines in the image
        # Define some information about the data
        data_name = 'electron_lines_r_th'
        filter_sum_string = 'no_filter'
        match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}

        # Look if they are already stored
        lines_data, lines_time_stamp = c_data.load_derived_data(
            data_name, filter_sum_string,
            compare_time_stamp=e_rth_image_time_stamp,
            verbose=verbose,
            match_data_dict=match_data_dict)

        # Check if there is any data
        if lines_data.size == 0:
            # if not make it
            r, w, a, red_chi2 = epicea.find_lines(
                e_rth_image, r_axis_mm, th_axis_rad, n_lines=n_lines)
            # put the data in a single array
            lines_data = np.concatenate([r.T,
                                         w.T,
                                         a.T,
                                         red_chi2.reshape(1, -1)])
            # and save it
            lines_time_stamp = c_data.store_derived_data(lines_data,
                                                         data_name,
                                                         filter_sum_string,
                                                         match_data_dict,
                                                         verbose=verbose)

        # Unpack the lines data
        r = lines_data[:n_lines, :].T
        w = lines_data[n_lines: 2*n_lines, :].T
        a = lines_data[2*n_lines: 3*n_lines, :].T
        red_chi2 = lines_data[3*n_lines, :].flatten()

        # Keep the latest time stamp
        latest_lines = max(lines_time_stamp, latest_lines)

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
        stdout.flush()
        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
        plt.plot(red_chi2, th_axis_rad)

        print 'Add data to the calibration object'
        stdout.flush()
        for line in range(n_lines):
            calibration.add_calibration_data(
                r[:, line], th_axis_rad,
                c_data.photon_energy() - binding[line],
                w[:, line],
                a[:, line] / multiplicity[line])

    print 'Create calibration'
    stdout.flush()
    calibration.create_conversion(poly_order=2)

    print 'Check the calibration'
    stdout.flush()
    plt_func.figure_wrapper('Calib data check')
    theta_list, data_list = calibration.get_data_copy()
    r_min = data_list[:, :, 0].min()
    r_max = data_list[:, :, 0].max()
    r_axis_for_calib_check_mm = np.linspace(r_min + (r_min-r_max) * 0.1,
                                            r_max + (r_max-r_min) * 0.1,
                                            256)
    for idx in range(len(theta_list)):
        plt.subplot(121)
        plt.plot(data_list[:, idx,  0], data_list[:, idx, 1], '.b')
        plt.plot(r_axis_for_calib_check_mm,
                 epicea.poly_line(calibration._energy_params_list[idx],
                                  r_axis_for_calib_check_mm), 'r')
        plt.subplot(122)
        plt.plot(data_list[:, idx,  0], data_list[:, idx, 3], '.b')
        plt.plot(r_axis_for_calib_check_mm,
                 epicea.poly_line(calibration._weights_params_list[idx],
                                  r_axis_for_calib_check_mm), 'r')

#    E_axis_eV = np.linspace(setting-30, setting+20, 2**8+1)[1::2]
#    E_all = []
#    err_all = []
#    theta_all = []
#    for c_data in calib_data_list:
#        print 'Get the calibrated energies, {} eV.'.format(
#            c_data.photon_energy())
#        stdout.flush()
#        E, err, weigths = calibration.get_energies(c_data.electrons.pos_r,
#                                                   c_data.electrons.pos_t)
#
#        E_all.extend(E)
#        err_all.extend(err)
#        theta_all.extend(c_data.electrons.pos_t)
#
#    plt_func.figure_wrapper('Energy domain all calibration data')
#    E_image = epicea.center_histogram_2d(E_all, theta_all,
#                                         E_axis_eV, th_axis_rad)
#    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
#                            kw_args={'aspect': 'auto'})

#    plt_func.figure_wrapper('Amplitude data')

    calibration.save_to_file(calibration_file_name)


if __name__ == '__main__':
#    for setting in [500]:
#    for setting in [373, 366, 357, 500]:
    for setting in [373]:
        print 'Procesing calibration data for the {} eV setting.'.format(
            setting)
        stdout.flush()
        make_calibration(setting, verbose=True)
#        raw_input('Press enter to do next calibration.')
#        plt.waitforbuttonpress(100)
