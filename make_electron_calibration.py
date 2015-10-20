# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
    """
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import peakutils

import plt_func
import electron_calibration_data
import determine_n2_background as background
import epicea

from global_parameters import (x_axis_mm, r_axis_mm, th_axis_rad)


n_voigt_with_bg_start_params = \
    epicea.electron_calibration_helper.n_voigt_with_bg_start_params
n_voigt_with_bg_model = \
    epicea.electron_calibration_helper.n_voigt_with_bg_model
n_voight_with_bg_500_eV_start_params = \
    epicea.electron_calibration_helper.n_voight_with_bg_500_eV_start_params
skewed_gauss_for_500_eV = \
    epicea.electron_calibration_helper.skewed_gauss_for_500_eV
skewed_gauss_for_500_eV_start_params = \
    epicea.electron_calibration_helper.skewed_gauss_for_500_eV_start_params

def load_data_for_calibration(setting, *, verbose=False):
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=False)
    except electron_calibration_data.CenterEnergyError as e:
        print(e)
        return
    # and make sure the list is sorted
    calib_data_list.sort(reverse=True)

    remove = {}
#    remove[373] = [462, 463, 465, 467.0, 469, 471.0, 470.0]
    remove[373] = [471.0, 470.0]
#    remove[373] = [471.0, 470.0, 467, 463]
#    remove[373] = [462, 463, 465, 469, 471.0, 470.0]
#    remove[373] = []
    
    remove[366] = [453.3, 463.3]
    
    remove[357] = [749.0]
    
#    remove[500] = [891.0, 893.0, 896.0, 900.0, 906.0, 910.0, 912.0, 914, 916]
    remove[500] = []
#    remove[500] = [893.0, 896.0, 900.0, 906.0, 910.0, 912.0, 914]
#    remove[500] = [893.0, 896.0, 900.0, 910.0, 912.0, 914]

    i = 0    
    while i < len(calib_data_list):
        if calib_data_list[i].photon_energy() in remove[setting]:
            calib_data_list.pop(i)
        else:
            i += 1
            continue

    return calib_data_list


def get_r_theta_image(dataset, r_axis_mm, th_axis_rad, gas, *,
                       plot=False, verbose=False, fig=None):

    data_name = dataset.name()
    if verbose:
        print('Get the theta-r spectrum for {}.'.format(data_name),
              flush=True)
    # Get the image fron the data in polar coordinates
    e_rth_image, e_rth_image_time_stamp = \
        dataset.get_e_rth_image(r_axis_mm, th_axis_rad, verbose=False)

    if plot:
        if verbose:
            print('Plot the raw theta r image of', data_name)
        if fig is None:
            # Make the figure
            rth_fig = plt_func.figure_wrapper(
                'e spectra theta-r {}'.format(data_name))
        else:
            rth_fig = fig
        ax1 = rth_fig.add_subplot(131)
        # Show the image
        plt_func.imshow_wrapper(e_rth_image, r_axis_mm, th_axis_rad,
                                kw_args={'aspect': 'auto'})
        # Make the image look nice
        plt_func.tick_fontsize()
        plt_func.title_wrapper('original ' + dataset.name())
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Angle (rad)')
#        plt_func.colorbar_wrapper()

        rth_fig.add_subplot(132, sharex=ax1, sharey=ax1)
        ax3 = rth_fig.add_subplot(133, sharex=ax1)
        ax3.plot(r_axis_mm, e_rth_image.sum(axis=0))
    else:
        rth_fig = None

    return e_rth_image, e_rth_image_time_stamp, rth_fig


def get_radial_factors(data, r_axis_mm, th_axis_rad,
                       compare_time_stamp, gas, *,
                       e_rth_image=None,
                       verbose=False, plot=False, fig=None):

    if e_rth_image is None:
        e_rth_image, e_rth_image_time_stamp, fig = get_r_theta_image(
            data, r_axis_mm, th_axis_rad, gas,
            verbose=verbose, plot=plot, fig=fig)
        compare_time_stamp = max(compare_time_stamp, e_rth_image_time_stamp)

    compare_time_stamp = max(compare_time_stamp, 1443779760)

    # Get the radial factors with some defined properties
    radial_factors_name = 'radial_factors'
    filter_sum_string = 'no_filter'
    match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}

    # First check the radial factors
    radial_factors, radial_factors_time_stamp = \
        data.load_derived_data(
        radial_factors_name, filter_sum_string,
        compare_time_stamp=compare_time_stamp,
        match_data_dict=match_data_dict,
        verbose=verbose)

    if radial_factors.size > 0:
        if verbose:
            print('Use old radial factors', flush=True)
    else:
        if verbose:
            print('Compute new radial factors.', flush=True)
        # Do one thing for the Kr data
        if gas in ['Kr']:
            # Get the centers from the original image
#            centers = ((e_rth_image * r_axis_mm).sum(axis=1) /
#                       e_rth_image.sum(axis=1))
#
#            # Find the center of the first line in the original
#            low_radius_centers = np.empty_like(centers)
#            for i_th in range(len(th_axis_rad)):
#                y = e_rth_image[i_th, :]
#                i_min = r_axis_mm.searchsorted(centers[i_th])
#                while y[i_min] >= y[i_min - 1]:
#                    i_min -= 1
#                while y[i_min] < y[i_min - 1]:
#                    i_min -= 1
##                I_low_radius = (((centers[i_th] - 3) <= r_axis_mm) &
##                                (r_axis_mm <= centers[i_th]))
#                I_low_radius = (((r_axis_mm[i_min] - 3) <= r_axis_mm) &
#                                (r_axis_mm <= (r_axis_mm[i_min] + 3)))
#                low_radius_centers[i_th] = (
#                    (e_rth_image[i_th, I_low_radius] *
#                     r_axis_mm[I_low_radius]).sum() /
#                    e_rth_image[i_th, I_low_radius].sum())
#
#            # Make radial scaling factors
#            radial_factors = low_radius_centers.mean() / low_radius_centers

    
            low_radius_centers = np.empty_like(th_axis_rad)
            for i_th in range(len(th_axis_rad)):
                peakind = peakutils.indexes(e_rth_image[i_th, :],
                                            thres=0.1, min_dist=3)
#                print('{}: {}, {}'.format(i_th, peakind,
#                      r_axis_mm[peakind[0]]))
                I_low_radius = slice(peakind[0]-2, peakind[0]+3)
                low_radius_centers[i_th] = (
                    (e_rth_image[i_th, I_low_radius] *
                     r_axis_mm[I_low_radius]).sum() /
                    e_rth_image[i_th, I_low_radius].sum())

            # Make radial scaling factors
            radial_factors = low_radius_centers.mean() / low_radius_centers
                

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

                result = lmfit.minimize(
                    epicea.electron_calibration_helper.n_line_fit_model,
                    params, args=(r_axis_mm[sl], e_rth_image[i_th, sl]))

                max_radius[i_th] = result.params['center_1'].value

#                    max_radius[i_th] = (
#                        (e_rth_image[i_th, sl] *
#                         r_axis_mm[sl]).sum() /
#                        e_rth_image[i_th, sl].sum())

            radial_factors = max_radius.mean() / max_radius
#                print(radial_factors)
#                plt.sca(proj_ax)
#                plt_func.legend_wrapper()
#                radial_factors = max_at_r_vec.mean() / max_at_r_vec
        # and save them
        radial_factors_time_stamp = data.store_derived_data(
            radial_factors, radial_factors_name, filter_sum_string,
            match_data_dict=match_data_dict, verbose=verbose)

    return radial_factors, radial_factors_time_stamp, fig


def get_straight_image(data, r_axis_mm, th_axis_rad,
                       compare_time_stamp, gas, *,
                       verbose=False, plot=False, fig=None,
                       radial_factors=None):

    d_name = data.name()
    
    if radial_factors is None:
        radial_factors, r_f_time_stamp, fig = get_radial_factors(
            data, r_axis_mm, th_axis_rad, 0, gas,
            verbose=verbose, plot=plot)
        compare_time_stamp = max(compare_time_stamp, r_f_time_stamp)

    compare_time_stamp = max(compare_time_stamp, 144178446)

    th_limits = epicea.limits_from_centers(th_axis_rad)

    # Get the corrected image with some defined properties
    straight_img_name = 'e_rth_image_straight'
    filter_sum_string = 'no_filter'
    match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}
    
    # With the radial factors done, now look for the corrected image
    match_data_dict['radial_factors'] = radial_factors
    straight_img, straight_img_time_stamp = \
        data.load_derived_data(
            straight_img_name, filter_sum_string,
            compare_time_stamp=compare_time_stamp,
            verbose=verbose,
            match_data_dict=match_data_dict)

    # If there is no data it needs to be made
    if straight_img.size > 0:
        if verbose:
            print('Use old straight image.', flush=True)
    else:
        if verbose:
            print('Make new straight image.', flush=True)
        # Get and adjust the coordinates
        th = data.electrons.pos_t.value
        r = data.electrons.pos_r.value
        for i_th in range(len(th_axis_rad)):
            try:
                I = (th_limits[i_th] <= th) & (th < th_limits[i_th+1])
            except RuntimeWarning as W:
                print(W.args)
                print(W.with_traceback)
                print('I.sum() =', I.sum(), 'i_th =', i_th, th_limits[i_th])
            r[I] *= radial_factors[i_th]

        straight_img = epicea.center_histogram_2d(r, th,
                                                  r_axis_mm, th_axis_rad)

        # Save the straight image
        straight_img_time_stamp = data.store_derived_data(
             straight_img, straight_img_name, filter_sum_string,
             match_data_dict=match_data_dict, verbose=verbose)

    if (fig is not None) & (plot == True):
        if verbose:
            print('Plot the straight image of', d_name, flush=True)
        plt.sca(fig.axes[1])
        img_axis = plt_func.imshow_wrapper(
            straight_img, r_axis_mm, th_axis_rad,
            kw_args={'aspect': 'auto'})
        plt_func.tick_fontsize()
        plt_func.title_wrapper('straight ' + d_name)
        plt_func.xlabel_wrapper('Position (mm)')
        plt_func.ylabel_wrapper('Angle (rad)')
        plt_func.colorbar_wrapper(mappable=img_axis)

    return straight_img, straight_img_time_stamp, fig


def plot_projections(fig, data, straight_img, r_axis_mm, th_axis_rad, gas,
                     n_lines, setting, *, verbose=False, **kws):
    # Get the projection
    r_projection = straight_img.sum(axis=0)
    
    params, fit_funk, kws, method = \
        epicea.electron_calibration_helper.get_params_and_funktion(
            setting, gas, r_axis_mm, r_projection,
            bg=(kws['bg'] if 'bg' in kws else None))
    # parameters for the two line fit
#    line_type = {'line_type': 'voigt'}
#    if gas == 'Kr':
#        params_r_proj = \
#            epicea.electron_calibration_helper.start_params(
#                x=r_axis_mm, y=r_projection,
#                n_lines=n_lines,
#                **line_type)
#        fit_funk = epicea.electron_calibration_helper.n_line_fit_model
#    elif gas == 'N2':
#        # get start parameters
#        if setting == 500:
#            params_r_proj = skewed_gauss_for_500_eV_start_params(
#                r_axis_mm, r_projection, bg=True)
##            params_r_proj['bg_factor'].value = 1
#            fit_funk = skewed_gauss_for_500_eV
#        else:
#            params_r_proj = n_voigt_with_bg_start_params(
#                    r_axis_mm, r_projection,
#                    n_lines=1, bg=True)
#            if 'bg_factor' in params_r_proj:
#                params_r_proj['bg_factor'].vary = False
#            fit_funk = n_voigt_with_bg_model

    # do the fit
    leastsq_kws = {}
    res = lmfit.minimize(fit_funk, params,
                         args=(r_axis_mm, r_projection),
                         kws=kws, method=method,
                         **leastsq_kws)

#        ci = lmfit.conf_interval(res)

    if verbose:
        print('\nProjection fit report for {}.'.format(data.name()))
        # Not sure what is in the MinimizerResult, wrap stuff in try statements
        try:
            print(res.message)
        except:
            pass
        try:
            print(res.ier)
        except:
            pass
        try:
            print(res.lmdif_message)
        except:
            pass
        lmfit.report_fit(res)
#        lmfit.ci_report(ci)
#                lmfit.report_errors(params_r_proj)
    ax = fig.axes[2]
    ax.plot(r_axis_mm, r_projection)
#    if gas == 'Kr':
#        ax.plot(
#            r_axis_mm,
#            epicea.electron_calibration_helper.n_line_fit_model(
#                params_r_proj, r_axis_mm, **line_type),
#            'r--')
#    elif setting == 500:
#        proj_fit = skewed_gauss_for_500_eV(params_r_proj, r_axis_mm, **kws)
#        ax.plot(
#            r_axis_mm, n_voigt_with_bg_model(params_r_proj,
#                                             r_axis_mm,
#                                             **kws),
#            'm--')
    ax.plot(r_axis_mm, fit_funk(res.params, r_axis_mm, **kws), '--r')
    plt.sca(ax)
    plt_func.tick_fontsize()
    plt_func.title_wrapper('straight ' + data.name())
    plt_func.xlabel_wrapper('Position (mm)')


def get_lines(data, r_axis_mm, th_axis_rad, n_lines, setting,
              compare_time_stamp, gas, *,
              verbose=False, plot=False, fig=None,
              straight_img=None, radial_factors=None, n2_bg=None,
              calibration=None):

    if straight_img is None:
        straigh_img, si_time_stamp, fig = get_straight_image(
            data, r_axis_mm, th_axis_rad, compare_time_stamp,
            gas, verbose=verbose, plot=plot, fig=fig)
        compare_time_stamp = max(compare_time_stamp, si_time_stamp)

    if radial_factors is None:
        radial_factors, rf_time_stamp, fig = get_radial_factors(
            data, r_axis_mm, th_axis_rad, compare_time_stamp, gas,
            verbose=verbose, plot=plot, fig=fig)
        compare_time_stamp = max(compare_time_stamp, rf_time_stamp)

    data_name = data.name()
    compare_time_stamp = max(compare_time_stamp, 1444390800)
 
    if gas == 'Kr':
        binding = np.array([93.788, 95.038])
        multiplicity = np.array([7., 5.])
        if verbose:
            print('Find the Kr 3d lines in {}.'.format(data_name), flush=True)
    elif gas == 'N2':
        binding = np.array([409.9])
        multiplicity = np.array([1.])
        if verbose:
            print('Find the N_2 s1 line in {}.'.format(data_name), flush=True)
    else:
        print('Unknown gas.',
              'Dataset names must specify the used gas (N2 or Kr).')

    # Find the lines in the image
    # Define some information about the data
    lines_name = 'electron_lines_r_th'
    filter_sum_string = 'no_filter'
    match_data_dict = {'r_axis_mm': r_axis_mm, 'th_axis_rad': th_axis_rad}

    # Look if they are already stored
    lines_data, lines_time_stamp = data.load_derived_data(
        lines_name, filter_sum_string,
        compare_time_stamp=compare_time_stamp,
        verbose=True,
        match_data_dict=match_data_dict)

    # Check if there is any data
    if lines_data.size == 0:
        if verbose:
            print('Make new lines.', flush=True)
        # if not make it
        r, w, a, red_chi2 = epicea.find_lines(
            straight_img, r_axis_mm, th_axis_rad,
            setting,
            n_lines_fit=(1 if ((gas == 'N2') and
                               ((setting != 500) or
                                (data.photon_energy() < 905)))
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
        lines_time_stamp = data.store_derived_data(lines_data,
                                                   lines_name,
                                                   filter_sum_string,
                                                   match_data_dict,
                                                   verbose=verbose)
    else:
       if verbose:
           print('Use old lines.', flush=True)

    # Unpack the lines data
    r = lines_data[:n_lines, :].T
    w = lines_data[n_lines: 2*n_lines, :].T
    a = lines_data[2*n_lines: 3*n_lines, :].T
    red_chi2 = lines_data[3*n_lines, :].flatten()


    # Plot the lines in the straight plot
    if plot:
        fmts = ['.r', '.m', '.y']
        for line in range(n_lines):
            fig.axes[1].errorbar(r[:, line], th_axis_rad, xerr=w[:, line],
                                 fmt=fmts[line], capsize=1)

    # Adjust the line parameters according to the radial factors
    r = (r.T / radial_factors).T
    w = (w.T / radial_factors).T
    # And the amplitude according to the data scaling
#    a /= (data.data_scaling * multiplicity)
#    a /= (multiplicity)
    a /= (data.data_scaling)

    # Plot the lines in the original plot
    if plot:
        for line in range(n_lines):
            fig.axes[0].errorbar(r[:, line], th_axis_rad,
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

#        print('Lineoiuts in waterfall plot.'
#        plt_func.figure_wrapper('lineouts {}'.format(c_data.name()))
#        for i, line in enumerate(e_rth_image):
#            plt.plot(r_axis_mm, line + i*20, 'b')

#        print('Plot reduced chi^2.'
#        stdout.flush()
#        plt_func.figure_wrapper('reduced chi^2 {}'.format(c_data.name()))
#        plt.plot(red_chi2, th_axis_rad)

    if calibration is not None:
        print('Add data to the calibration object', flush=True)
        for line in range(n_lines):
            calibration.add_calibration_data(
                r[:, line], th_axis_rad,
                data.photon_energy() - binding[line],
                w[:, line],
                a[:, line] if line in ([0] if n_lines == 2 else [0]) else
                    np.nan * np.ones_like(a[:, line]))

    return r, w, a, red_chi2, lines_time_stamp


def make_calibration(setting, plot=True, verbose=False):
    """Make the electron calibration for a given center energy setting."""
    # The Kr binding energies are needed for Kr based calibrations
    calibration_file_name = 'h5_data/calib_{}.h5'.format(setting)

    # Get the list of data sets
    calib_data_list = load_data_for_calibration(setting, verbose=verbose)

    if 'Kr' in calib_data_list[0].name():
        gas = 'Kr'
        n_lines = 2
    elif 'N2' in calib_data_list[0].name():
        gas = 'N2'
        n_lines = 1
    else:
        print('Unknown gas.',
              'Dataset names must specify the used gas (N2 or Kr).')

    # Create an empty calibration object
    calibration = epicea.ElectronEnergyCalibration()

#    # Make an x axis
#    x_axis_mm = np.linspace(-23, 23, 2**8)
#    # Define polar coordinate axis vectors
#    r_axis_mm = np.linspace(0, 25, 2**7+1)[1::2]
#    th_axis_rad = np.linspace(0, 2*np.pi, 2**9+1)[1::2]

    if plot:
        # Create a plot of all the spectra in the data list
        if verbose:
            print('Plot all the raw spectra.')
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

#        plt.tight_layout()

    # Keep track of the latest time stamp of the lines
    latest_lines = 0
    rth_fig = {}
    straight_img = {}
    straight_img_time_stamp = {}
    radial_factors = {}

    # Iterate over the datasets and make images for each set.
    # Also find the lines in each of the spectra
    for i, c_data in enumerate(calib_data_list):
        d_name = c_data.name()
        #######################################
        # Raw theta vs. r spectrum
        e_rth_image, e_rth_image_time_stamp, rth_fig[d_name] = \
            get_r_theta_image(c_data, r_axis_mm, th_axis_rad, gas,
                               plot=True, verbose=verbose, fig=None)

        #######################################
        # radial factors
        radial_factors[d_name], radial_factors_time_stamp, _ = \
            get_radial_factors(c_data, r_axis_mm, th_axis_rad,
                               e_rth_image_time_stamp, gas,
                               e_rth_image=e_rth_image,
                               verbose=verbose)

        #######################################
        # Straight image
        straight_img[d_name], straight_img_time_stamp[d_name], _= \
            get_straight_image(c_data, r_axis_mm, th_axis_rad,
                               radial_factors_time_stamp, gas,
                               radial_factors=radial_factors[d_name],
                               verbose=verbose, plot=plot,
                               fig=rth_fig[d_name])

    # Make sure the straight image is avaliable for all the calibration data
    # sets before making the background.
    # This is achieved by exiting the loop and starting a neew one.
    for i, c_data in enumerate(calib_data_list):
        d_name = c_data.name()
        #######################################
        # N_2 background
        if gas == 'N2':
            if not background.check_bg_valid(setting, len(r_axis_mm),
                                             verbose=verbose):
                if verbose:
                    print('Make new N2 background.', flush=True)
                background.make_new_bg(setting=setting, plot=plot,
                                       verbose=verbose)
            elif verbose:
                print('Use old N2 background.')

            n2_bg = (background.load_background(setting, len(r_axis_mm))
                     * c_data.data_scaling)
            kws = {'bg': n2_bg}

        else:
            n2_bg = None
            kws = {}


        ########
        # do some plotting of the straight image projection and a fit to it
        if plot:
            plot_projections(rth_fig[d_name], c_data, straight_img[d_name],
                             r_axis_mm, th_axis_rad,
                             gas, n_lines, setting,
                             verbose=verbose,
                             **kws)

        #######################################
        # Find lines
        r, w, a, red_chi2, lines_time_stamp = get_lines(
            c_data, r_axis_mm, th_axis_rad, n_lines, setting,
            max(straight_img_time_stamp.values()), gas,
            verbose=verbose, plot=plot, fig=rth_fig[d_name],
            straight_img=straight_img[d_name],
            radial_factors=radial_factors[d_name],
            n2_bg=n2_bg, calibration=calibration)

        
        # Keep the latest time stamp
        latest_lines = max(lines_time_stamp, latest_lines)

    print('Create calibration', flush=True)
#    calibration.create_conversion(poly_order=2)
    calibration.create_or_load_conversion(calibration_file_name,
                                          compare_time_stmp=max(latest_lines,
                                                                1444746720),
                                          poly_order=2,
                                          verbose=verbose)

    print('Check the calibration', flush=True)
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
#        plt.plot(r_axis_for_calib_check_mm,
#                 epicea.poly_line(calibration._energy_params_list[idx],
#                                  r_axis_for_calib_check_mm), 'r')
        plt.plot(r_axis_for_calib_check_mm,
                 epicea.electron_calibration_helper.r_to_e_conversion(
                     calibration._energy_params_list[idx],
                     r_axis_for_calib_check_mm), 'r')
        plt.subplot(122)
        shift = 0.1*idx
        plt.plot(data_list[:, idx,  0],
                     1./data_list[:, idx, 3] + shift,
                     '.:')
#        plt.plot(r_axis_for_calib_check_mm,
#                 epicea.poly_line(calibration._weights_params_list[idx],
#                                  r_axis_for_calib_check_mm) + shift, 'r')

    plt.plot(data_list[...,  0].mean(1), 1./data_list[..., 3].mean(1),
             'k', lw=2)

    a, b = ((338, 368) if setting == 357 else
            (359, 368) if setting == 366 else
            (366, 376) if setting == 373 else
            (480, 508))
    E_axis_eV = np.linspace(a, b, 2**9+1)[1::2]
    E_all = []
    err_all = []
    theta_all = []
    weight_all = []
    for c_data in calib_data_list:
        print('Get the calibrated energies, {} eV.'.format(
            c_data.photon_energy()), flush=True)
        c_data.calculate_electron_energy(calibration, verbose=verbose)
#        E, err, weigths = calibration.get_energies(c_data.electrons.pos_r,
#                                                   c_data.electrons.pos_t)

        E_all.extend(c_data.electrons.energy.value)
        err_all.extend(c_data.electrons.energy_uncertainty.value)
        theta_all.extend(c_data.electrons.pos_t.value)
        weight_all.extend(c_data.electrons.spectral_weight.value)

    plt_func.figure_wrapper('Energy domain all calibration data')
    ax = plt.subplot(211)
    E_image = epicea.center_histogram_2d(E_all, theta_all,
                                         E_axis_eV, th_axis_rad,
                                         weights=weight_all)
    plt_func.imshow_wrapper(E_image, E_axis_eV, th_axis_rad,
                            kw_args={'aspect': 'auto'})

    plt.subplot(212, sharex=ax)
    plt.plot(E_axis_eV, E_image.sum(0))
    plt.grid(True)

    calibration.save_to_file(calibration_file_name)


if __name__ == '__main__':
    plt.ion()
    calibration = epicea.ElectronEnergyCalibration()
#    for setting in [373, 366, 357, 500]:
    for setting in [373]:
#    for setting in [366]:
#    for setting in [357]:
#    for setting in [500]:
        print('Procesing calibration data for the {} eV setting.'.format(
              setting), flush=True)
        make_calibration(setting, plot=True, verbose=True)

        calibration.load_from_file('h5_data/calib_{}.h5'.format(setting))
