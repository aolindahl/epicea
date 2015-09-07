# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
    """
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
import lmfit

import plt_func
import electron_calibration_data
import determine_n2_background as background
import epicea


n_voigt_with_bg_start_params = \
    epicea.electron_calibration_helper.n_voigt_with_bg_start_params
n_voigt_with_bg_model = \
    epicea.electron_calibration_helper.n_voigt_with_bg_model


#def make_calibration(setting, verbose=False):
#    """Make the electron calibration for a given center energy setting."""
#    # The Kr binding energies are needed for Kr based calibrations
#    kr_binding_energies = np.array([93.788, 95.038])
#    kr_multiplicity = np.array([11., 9.])
#    n2_binidng_energy = np.array([409.9])
#    n2_multiplicity = np.array([1.])
#    calibration_file_name = 'h5_data/calib_{}.h5'.format(setting)
#
#    # Get the list of data sets
#    try:
#        calib_data_list = electron_calibration_data.get_data_in_list(
#            setting, verbose=verbose)
#    except electron_calibration_data.CenterEnergyError as e:
#        print e
#        return
#    # and make sure the list is sorted
#    calib_data_list.sort(reverse=True)
#
#    if 'Kr' in calib_data_list[0].name():
#        gas = 'Kr'
#        binding = kr_binding_energies
#        multiplicity = kr_multiplicity
#    elif 'N2' in calib_data_list[0].name():
#        gas = 'N2'
#        binding = n2_binidng_energy
#        multiplicity = n2_multiplicity
#    else:
#        print 'Unknown gas.',
#        print 'Dataset names must specify the used gas (N2 or Kr).'
#
#    n_lines = len(binding)
#
#    # Create an empty calibration object
#    calibration = epicea.ElectronEnergyCalibration()
#
#    # Create a plot of all the spectra in the data list
#    print 'Plot all the raw spectra.'
#    # Make an x axis
#    x_axis_mm = np.linspace(-23, 23, 512)
#    # Create the figure
#    plt_func.figure_wrapper('Calibration spectra {} eV'.format(setting))
#    # One subplot for each data set...
#    n_subplots = calib_data_list.len()
#    # ... nicely set in collumns and rows
#    n_rows = np.floor(np.sqrt(n_subplots))
#    n_cols = np.ceil(float(n_subplots)/n_rows)
#    # Iterate over the datasets
#    for i, c_data in enumerate(calib_data_list):
#        plt.subplot(n_rows, n_cols, i+1)  # Make the subplot
#        # Show the electron figure
#        plt_func.imshow_wrapper(c_data.get_e_xy_image(x_axis_mm)[0], x_axis_mm)
#        # Adjust the figure for good looks
#        plt_func.tick_fontsize()
#        plt_func.title_wrapper(c_data.name())
#        plt_func.xlabel_wrapper('Position (mm)')
#        plt_func.ylabel_wrapper('Position (mm)')
#
#    plt.tight_layout()
#    # Define polar coordinate axis vectors
#    r_axis_mm = np.linspace(0, 25, 2**8+1)[1::2]
#    th_axis_rad = np.linspace(0, 2*np.pi, 2**8+1)[1::2]
#
#    # Keep track of the latest time stamp of the lines
#    latest_lines = 0
#
#    # Iterate over the datasets and make images for each set.
#    # Also find the lines in each of the spectra
#    for i, c_data in enumerate(calib_data_list):
#        print 'Make the theta-r spectrum for {}.'.format(c_data.name())
#        stdout.flush()
#        # Make the figure
#        plt_func.figure_wrapper(
#            'e spectra theta-r {}'.format(c_data.name()))
#        # Get the image fron the data in polar coordinates
#        e_rth_image, e_rth_image_time_stamp = \
#            c_data.get_e_rth_image(r_axis_mm, th_axis_rad)
#        # Show the image
#        plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
#                                kw_args={'aspect': 'auto'})
#        # Make the image look nice
#        plt_func.tick_fontsize()
#        plt_func.title_wrapper(c_data.name())
#        plt_func.xlabel_wrapper('Position (mm)')
#        plt_func.ylabel_wrapper('Angle (rad)')
#        plt_func.colorbar_wrapper()
#
#        if gas == 'Kr':
#            print 'Find the Kr 3d lines in {}.'.format(c_data.name())
#            stdout.flush()
#        elif gas == 'N2':
#            print 'Find the N_2 s1 line in {}.'.format(c_data.name())
#            stdout.flush()
#
#        # Find the lines in the image
#        # Define some information about the data
#        data_name = 'electron_lines_r_th'
#        filter_sum_string = 'no_filter'
#        match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}
#
#        # Look if they are already stored
#        lines_data, lines_time_stamp = c_data.load_derived_data(
#            data_name, filter_sum_string,
#            compare_time_stamp=e_rth_image_time_stamp,
#            verbose=verbose,
#            match_data_dict=match_data_dict)
#
#        # Check if there is any data
#        if lines_data.size == 0:
#            # if not make it
#            r, w, a, red_chi2 = epicea.find_lines(
#                e_rth_image, r_axis_mm, th_axis_rad, n_lines=n_lines)
#            # put the data in a single array
#            lines_data = np.concatenate([r.T,
#                                         w.T,
#                                         a.T,
#                                         red_chi2.reshape(1, -1)])
#            # and save it
#            lines_time_stamp = c_data.store_derived_data(lines_data,
#                                                         data_name,
#                                                         filter_sum_string,
#                                                         match_data_dict,
#                                                         verbose=verbose)
#
#        # Unpack the lines data
#        r = lines_data[:n_lines, :].T
#        w = lines_data[n_lines: 2*n_lines, :].T
#        a = lines_data[2*n_lines: 3*n_lines, :].T
#        red_chi2 = lines_data[3*n_lines, :].flatten()
#
#        # Keep the latest time stamp
#        latest_lines = max(lines_time_stamp, latest_lines)
#
#        fmts = ['.r', '.m', '.y']
#        for line in range(n_lines):
#            plt.errorbar(r[:, line], th_axis_rad, xerr=w[:, line],
#                         fmt=fmts[line], capsize=1)
##        plt.errorbar(r_1, th_axis_rad, xerr=w_1, fmt='.r', capsize=0)
##        plt.errorbar(r_2, th_axis_rad, xerr=w_2, fmt='.m', capsize=0)
#
##        poly_order = 40
##        l1_params = line_start_params([0]*poly_order)
##        lmfit.minimize(poly_line, l1_params, args=(th_axis_rad, r_1, w_1))
##        plt.plot(poly_line(l1_params, th_axis_rad,), th_axis_rad, 'k')
##        l2_params = line_start_params([0]*poly_order)
##        out = lmfit.minimize(poly_line, l2_params,
##                             args=(th_axis_rad, r_2, w_2))
##        plt.plot(poly_line(l2_params, th_axis_rad), th_axis_rad, 'k')
#
##        print 'Lineoiuts in waterfall plot.'
##        plt_func.figure_wrapper('lineouts {}'.format(c_data.name()))
##        for i, line in enumerate(e_rth_image):
##            plt.plot(r_axis_mm, line + i*20, 'b')
#
#        print 'Plot reduced chi^2.'
#        stdout.flush()
#        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
#        plt.plot(red_chi2, th_axis_rad)
#
#        print 'Add data to the calibration object'
#        stdout.flush()
#        for line in range(n_lines):
#            calibration.add_calibration_data(
#                r[:, line], th_axis_rad,
#                c_data.photon_energy() - binding[line],
#                w[:, line],
#                a[:, line] / multiplicity[line])
#
#    print 'Create calibration'
#    stdout.flush()
#    calibration.create_conversion(poly_order=2)
#
#    print 'Check the calibration'
#    stdout.flush()
#    plt_func.figure_wrapper('Calib data check')
#    theta_list, data_list = calibration.get_data_copy()
#    r_min = data_list[:, :, 0].min()
#    r_max = data_list[:, :, 0].max()
#    r_axis_for_calib_check_mm = np.linspace(r_min + (r_min-r_max) * 0.1,
#                                            r_max + (r_max-r_min) * 0.1,
#                                            256)
#    for idx in range(len(theta_list)):
#        plt.subplot(121)
#        plt.plot(data_list[:, idx,  0], data_list[:, idx, 1], '.b')
#        plt.plot(r_axis_for_calib_check_mm,
#                 epicea.poly_line(calibration._energy_params_list[idx],
#                                  r_axis_for_calib_check_mm), 'r')
#        plt.subplot(122)
#        plt.plot(data_list[:, idx,  0], data_list[:, idx, 3], '.b')
#        plt.plot(r_axis_for_calib_check_mm,
#                 epicea.poly_line(calibration._weights_params_list[idx],
#                                  r_axis_for_calib_check_mm), 'r')
#
##    E_axis_eV = np.linspace(setting-30, setting+20, 2**8+1)[1::2]
##    E_all = []
##    err_all = []
##    theta_all = []
##    for c_data in calib_data_list:
##        print 'Get the calibrated energies, {} eV.'.format(
##            c_data.photon_energy())
##        stdout.flush()
##        E, err, weigths = calibration.get_energies(c_data.electrons.pos_r,
##                                                   c_data.electrons.pos_t)
##
##        E_all.extend(E)
##        err_all.extend(err)
##        theta_all.extend(c_data.electrons.pos_t)
##
##    plt_func.figure_wrapper('Energy domain all calibration data')
##    E_image = epicea.center_histogram_2d(E_all, theta_all,
##                                         E_axis_eV, th_axis_rad)
##    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
##                            kw_args={'aspect': 'auto'})
#
##    plt_func.figure_wrapper('Amplitude data')
#
#    calibration.save_to_file(calibration_file_name)


def make_calibration(setting, plot=True, verbose=False):
    """Make the electron calibration for a given center energy setting."""
    # The Kr binding energies are needed for Kr based calibrations
    kr_binding_energies = np.array([93.788, 95.038])
    kr_multiplicity = np.array([4., 6.])
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

    # Make an x axis
    x_axis_mm = np.linspace(-23, 23, 2**8)
    # Define polar coordinate axis vectors
    r_axis_mm = np.linspace(0, 25, 2**7+1)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 2**9+1)[1::2]
    # Get the theta limits
    th_limits = epicea.limits_from_centers(th_axis_rad)

    if plot:
        # Create a plot of all the spectra in the data list
        print 'Plot all the raw spectra.'
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
            plt_func.imshow_wrapper(c_data.get_e_xy_image(x_axis_mm)[0],
                                    x_axis_mm)
            # Adjust the figure for good looks
            plt_func.tick_fontsize()
            plt_func.title_wrapper(c_data.name())
            plt_func.xlabel_wrapper('Position (mm)')
            plt_func.ylabel_wrapper('Position (mm)')

        plt.tight_layout()

    # Keep track of the latest time stamp of the lines
    latest_lines = 0
    rth_fig = {}
    proj_ax = {}
    original_ax = {}
    straight_img = {}
    radial_factors = {}

    # Make some empty dictionaries

    # Iterate over the datasets and make images for each set.
    # Also find the lines in each of the spectra
    for i, c_data in enumerate(calib_data_list):
        d_name = c_data.name()
        #######################################
        # Raw theta vs. r spectrum
        print 'Make the theta-r spectrum for {}.'.format(c_data.name())
        stdout.flush()
        # Get the image fron the data in polar coordinates
        e_rth_image, e_rth_image_time_stamp = \
            c_data.get_e_rth_image(r_axis_mm, th_axis_rad)

        if plot:
            if verbose:
                print 'Plot the raw theta r image of', c_data.name()
            # Make the figure
            rth_fig[d_name] = plt_func.figure_wrapper(
                'e spectra theta-r {}'.format(c_data.name()))
            original_ax[d_name] = rth_fig[d_name].add_subplot(131)
            # Show the image
            plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
                                    kw_args={'aspect': 'auto'})
            # Make the image look nice
            plt_func.tick_fontsize()
            plt_func.title_wrapper('original ' + c_data.name())
            plt_func.xlabel_wrapper('Position (mm)')
            plt_func.ylabel_wrapper('Angle (rad)')
            plt_func.colorbar_wrapper()

            proj_ax[d_name] = rth_fig[d_name].add_subplot(
                133, sharex=original_ax[d_name])
            proj_ax[d_name].plot(r_axis_mm, e_rth_image.sum(axis=0))

        #######################################
        # radial factors

        # Get the corrected image with some defined properties
        straight_img_name = 'e_rth_image_straight'
        radial_factors_name = 'radial_factors'
        filter_sum_string = 'no_filter'
        match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}

        # First check the radial factors
        radial_factors[d_name], radial_factors_time_stamp = \
            c_data.load_derived_data(
            radial_factors_name, filter_sum_string,
            compare_time_stamp=max(e_rth_image_time_stamp, 1539364016),
            match_data_dict=match_data_dict,
            verbose=verbose)

        if radial_factors[d_name].size == 0:
            # Do one thing for the Kr data
            if gas in ['Kr']:
                # Get the centers from the original image
                centers = ((e_rth_image * r_axis_mm).sum(axis=1) /
                           e_rth_image.sum(axis=1))

                # Find the center of the first line in the original
                low_radius_centers = np.empty_like(centers)
                for i_th in range(len(th_axis_rad)):
                    y = e_rth_image[i_th, :]
                    i_min = r_axis_mm.searchsorted(centers[i_th])
                    while y[i_min] > y[i_min - 1]:
                        i_min -= 1
                    while y[i_min] > y[i_min+1]:
                        i_min += 1
                    I_low_radius = (((centers[i_th] - 3) <= r_axis_mm) &
                                    (r_axis_mm <= centers[i_th]))
                    low_radius_centers[i_th] = (
                        (e_rth_image[i_th, I_low_radius] *
                         r_axis_mm[I_low_radius]).sum() /
                        e_rth_image[i_th, I_low_radius].sum())

                # Make radial scaling factors
                radial_factors[d_name] = (low_radius_centers.mean() /
                                          low_radius_centers)

            elif gas in ['N2']:
                # Do something else for the N2 data
                # Find the maximum position in the image for each line
                max_at_r_vec = np.argmax(e_rth_image, axis=1)
                max_at_r_start = int(np.nanmedian(max_at_r_vec) -
                                     max((5, np.nanstd(max_at_r_vec) / 2)))
                max_at_r_vec = (max_at_r_start +
                                np.argmax(e_rth_image[:,
                                                      max_at_r_start:
                                                          max_at_r_start+10],
                                          axis=1))
                max_radius = np.empty_like(max_at_r_vec, dtype=float)
                for i_th, max_at_r in enumerate(max_at_r_vec):
                    max_val_half = e_rth_image[i_th, max_at_r] / 2
                    r_low = max_at_r - 1
                    while (e_rth_image[i_th, r_low] > max_val_half):
                        r_low -= 1
                    r_high = max_at_r + 2
                    while (e_rth_image[i_th, r_high] > max_val_half):
                        r_high += 1
                    sl = slice(r_low, r_high)

                    params = epicea.electron_calibration_helper.start_params(
                        r_axis_mm[sl], e_rth_image[i_th, sl], n_lines=1)
#                    params.add('skew_1', 0)

                    lmfit.minimize(
                        epicea.electron_calibration_helper.n_line_fit_model,
                        params, args=(r_axis_mm[sl], e_rth_image[i_th, sl]))

                    max_radius[i_th] = params['center_1'].value

#                    max_radius[i_th] = (
#                        (e_rth_image[i_th, sl] *
#                         r_axis_mm[sl]).sum() /
#                        e_rth_image[i_th, sl].sum())

                radial_factors[d_name] = max_radius.mean() / max_radius
#                print radial_factors[d_name]
#                plt.sca(proj_ax[d_name])
#                plt_func.legend_wrapper()
#                radial_factors[d_name] = max_at_r_vec.mean() / max_at_r_vec
            # and save them
            radial_factors_time_stamp = c_data.store_derived_data(
                radial_factors[d_name], radial_factors_name, filter_sum_string,
                match_data_dict=match_data_dict, verbose=verbose)

        #######################################
        # Straight image

        # With the radial factors done, now look for the corrected image
        match_data_dict['radial_factors'] = radial_factors[d_name]
        straight_img[d_name], straight_img_time_stamp = \
            c_data.load_derived_data(
                straight_img_name, filter_sum_string,
                compare_time_stamp=max((radial_factors_time_stamp,
                                        1438681958)),
                verbose=verbose,
                match_data_dict=match_data_dict)

        # If there is no data it needs to be made
        if straight_img[d_name].size == 0:
            # Get and adjust the coordinates
            th = c_data.electrons.pos_t.value
            r = c_data.electrons.pos_r.value
            for i_th in range(len(th_axis_rad)):
                I = (th_limits[i_th] <= th) & (th < th_limits[i_th+1])
                r[I] *= radial_factors[d_name][i_th]

            straight_img[d_name] = epicea.center_histogram_2d(
                r, th, r_axis_mm, th_axis_rad)

            # Save the straight image
            straight_img_time_stamp = c_data.store_derived_data(
                 straight_img[d_name], straight_img_name, filter_sum_string,
                 match_data_dict=match_data_dict, verbose=verbose)

    # Make sure the straight image is avaliable for all the calibration data
    # sets before making the background.
    # This is achieved by exiting the loop and starting a neew one.
    for i, c_data in enumerate(calib_data_list):
        d_name = c_data.name()
        #######################################
        # N_2 background
        if gas == 'N2':
            if not background.check_bg_valid(setting=setting, verbose=verbose):
                background.make_new_bg(setting=setting, plot=plot,
                                       verbose=False)

            n2_bg = background.load_background(setting) * c_data.data_scaling

        else:
            n2_bg = None

        kws = {'bg': n2_bg}

        ########
        # do some plotting
        if plot:
            if verbose:
                print 'Plot the straight image of', c_data.name()
            straight_ax = rth_fig[d_name].add_subplot(
                132, sharex=original_ax[d_name], sharey=original_ax[d_name])
            img_axis = plt_func.imshow_wrapper(
                straight_img[d_name], r_axis_mm, th_axis_rad,
                ax=straight_ax, kw_args={'aspect': 'auto'})
            plt_func.tick_fontsize()
            plt_func.title_wrapper('straight ' + c_data.name())
            plt_func.xlabel_wrapper('Position (mm)')
            plt_func.ylabel_wrapper('Angle (rad)')
            plt_func.colorbar_wrapper(mappable=img_axis)

            # Get the projection
            r_projection = straight_img[d_name].sum(axis=0)
            # parameters for the two line fit
            line_type = {'line_type': 'voigt'}
            if gas == 'Kr':
                params_r_proj = \
                    epicea.electron_calibration_helper.start_params(
                        x=r_axis_mm, y=r_projection,
                        n_lines=n_lines,
                        **line_type)
                # do the fit
                res = lmfit.minimize(
                    epicea.electron_calibration_helper.n_line_fit_model,
                    params_r_proj, args=(r_axis_mm, r_projection),
                    kws=line_type)
            elif gas == 'N2':
                # get start parameters
                params_r_proj = n_voigt_with_bg_start_params(
                        r_axis_mm, r_projection,
                        n_lines=(2 if ((setting == 500) and
                                       (c_data.photon_energy() > 905)) else 1),
                        bg=True)
                if setting == 500:
                    params_r_proj['bg_factor'].value = 1
                elif 'bg_factor' in params_r_proj:
                    params_r_proj['bg_factor'].vary = False
                # do the fit
                res = lmfit.minimize(
                    n_voigt_with_bg_model,
                    params_r_proj, args=(r_axis_mm, r_projection),
                    kws=kws)

            if verbose:
                print 'Projection fit report.'
                print res.message
                lmfit.report_fit(res)
#                lmfit.report_errors(params_r_proj)
            proj_ax[d_name].plot(r_axis_mm, r_projection)
            if gas == 'Kr':
                proj_ax[d_name].plot(
                    r_axis_mm,
                    epicea.electron_calibration_helper.n_line_fit_model(
                        params_r_proj, r_axis_mm, **line_type),
                    'r--')
            else:
                proj_ax[d_name].plot(
                    r_axis_mm, n_voigt_with_bg_model(params_r_proj,
                                                     r_axis_mm,
                                                     bg=n2_bg),
                    'm--')
            plt.sca(proj_ax[d_name])
            plt_func.tick_fontsize()
            plt_func.title_wrapper('straight ' + c_data.name())
            plt_func.xlabel_wrapper('Position (mm)')

        #######################################
        # Find lines

        if verbose:
            if gas == 'Kr':
                print 'Find the Kr 3d lines in {}.'.format(c_data.name())
            elif gas == 'N2':
                print 'Find the N_2 s1 line in {}.'.format(c_data.name())
            stdout.flush()

        # Find the lines in the image
        # Define some information about the data
        lines_name = 'electron_lines_r_th'

        # Look if they are already stored
        lines_data, lines_time_stamp = c_data.load_derived_data(
            lines_name, filter_sum_string,
            compare_time_stamp=max(straight_img_time_stamp, 1439294979),
            verbose=verbose,
            match_data_dict=match_data_dict)

        # Check if there is any data
        if lines_data.size == 0:
            # if not make it
            r, w, a, red_chi2 = epicea.find_lines(
                straight_img[d_name], r_axis_mm, th_axis_rad,
                n_lines_fit=(1 if ((gas == 'N2') and
                                   ((setting != 500) or
                                    (c_data.photon_energy() < 905)))
                             else 2),
                n_lines_store=n_lines,
                bg=(n2_bg if gas == 'N2' else None),
                gas=gas,
                verbose=verbose)
            # put the data in a single array
            lines_data = np.concatenate([r.T,
                                         w.T,
                                         a.T,
                                         red_chi2.reshape(1, -1)])
            # and save it
            lines_time_stamp = c_data.store_derived_data(lines_data,
                                                         lines_name,
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

        # Plot the lines in the straight plot
        if plot:
            fmts = ['.r', '.m', '.y']
            for line in range(n_lines):
                straight_ax.errorbar(r[:, line], th_axis_rad, xerr=w[:, line],
                                     fmt=fmts[line], capsize=1)

        # Adjust the line parameters according to the radial factors
        r = (r.T / radial_factors[d_name]).T
        w = (w.T / radial_factors[d_name]).T
        # And the amplitude according to the data scaling
        a /= (c_data.data_scaling * multiplicity)
#        a /= (multiplicity)
#        a /= (c_data.data_scaling)

        # Plot the lines in the original plot
        if plot:
            for line in range(n_lines):
                original_ax[d_name].errorbar(r[:, line], th_axis_rad,
                                             xerr=w[:, line],
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

#        print 'Plot reduced chi^2.'
#        stdout.flush()
#        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
#        plt.plot(red_chi2, th_axis_rad)

        print 'Add data to the calibration object'
        stdout.flush()
        for line in range(n_lines):
            calibration.add_calibration_data(
                r[:, line], th_axis_rad,
                c_data.photon_energy() - binding[line],
                w[:, line],
                1.0 / a[:, line])

    print 'Create calibration'
    stdout.flush()
#    calibration.create_conversion(poly_order=2)
    calibration.create_or_load_conversion(calibration_file_name,
                                          compare_time_stmp=max(latest_lines,
                                                                1439381383),
                                          poly_order=2,
                                          verbose=verbose)

    print 'Check the calibration'
    stdout.flush()
    plt_func.figure_wrapper('Calib data check')
    theta_list, data_list = calibration.get_data_copy()
    r_min = data_list[:, :, 0].min()
    r_max = data_list[:, :, 0].max()
    r_axis_for_calib_check_mm = np.linspace(r_min + (r_min-r_max) * 0.0,
                                            r_max + (r_max-r_min) * 0.0,
                                            256)
    for idx in range(0, len(theta_list),
                     int(np.ceil(float(len(theta_list))/20))):
        plt.subplot(121)
        plt.plot(data_list[:, idx,  0], data_list[:, idx, 1], '.:')
        plt.plot(r_axis_for_calib_check_mm,
                 epicea.poly_line(calibration._energy_params_list[idx],
                                  r_axis_for_calib_check_mm), 'r')
        plt.subplot(122)
        shift = 0.1*idx
        plt.errorbar(data_list[:, idx,  0],
                     data_list[:, idx, 3] + shift,
                     data_list[:, idx,  2], fmt='.:')
        plt.plot(r_axis_for_calib_check_mm,
                 epicea.poly_line(calibration._weights_params_list[idx],
                                  r_axis_for_calib_check_mm) + shift, 'r')

    E_axis_eV = np.linspace(setting-30, setting+20, 2**8+1)[1::2]
    E_all = []
    err_all = []
    theta_all = []
    for c_data in calib_data_list:
        print 'Get the calibrated energies, {} eV.'.format(
            c_data.photon_energy())
        stdout.flush()
        c_data.calculate_electron_energy(calibration, verbose=verbose)
#        E, err, weigths = calibration.get_energies(c_data.electrons.pos_r,
#                                                   c_data.electrons.pos_t)

        E_all.extend(c_data.electrons.energy.value)
        err_all.extend(c_data.electrons.energy_uncertainty.value)
        theta_all.extend(c_data.electrons.pos_t.value)

    plt_func.figure_wrapper('Energy domain all calibration data')
    E_image = epicea.center_histogram_2d(E_all, theta_all,
                                         E_axis_eV, th_axis_rad)
    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
                            kw_args={'aspect': 'auto'})

    calibration.save_to_file(calibration_file_name)


if __name__ == '__main__':
    plt.ion()
#    for setting in [373, 366, 357, 500]:
    for setting in [373]:
#    for setting in [366]:
#    for setting in [357]:
#    for setting in [500]:
        print 'Procesing calibration data for the {} eV setting.'.format(
            setting)
        stdout.flush()
        make_calibration(setting, plot=True, verbose=True)
