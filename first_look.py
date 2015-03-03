# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:03:56 2015

@author: Anton O. Lindahl
"""

import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
# plt.tight_layout()

import epicea_hdf5 as epicea

ION_VMI_X_OFFSET = 1.8
ION_VMI_Y_OFFSET = -0.1
ION_VMI_OFFSET = {}
ION_VMI_OFFSET['430_high'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['412_high'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['430_mid'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['412_mid'] = np.array([1.8, -0.1])
ION_VMI_OFFSET['430_low'] = np.array([-0.5, -0.1])
ION_VMI_OFFSET['412_low'] = np.array([-0.5, -0.1])
ION_VMI_OFFSET['560'] = np.array([1.8, -0.1])

ELECTRON_X_OFFSET = -0.2
ELECTRON_Y_OFFSET = -0.1
ELECTRON_OFFSET = {}
ELECTRON_OFFSET['430_high'] = np.array([-0.2, -0.1])
ELECTRON_OFFSET['412_high'] = np.array([-0.2, 0.1])
ELECTRON_OFFSET['430_mid'] = np.array([1., -0.1])
ELECTRON_OFFSET['412_mid'] = np.array([-0.2, 0.1])
ELECTRON_OFFSET['430_low'] = np.array([-0.2, -0.2])
ELECTRON_OFFSET['412_low'] = np.array([-0.2, -0.1])
ELECTRON_OFFSET['560'] = np.array([-0.2, -0.2])

NN_O_TIME_SUM_RANGE_US = {}
NO_N_TIME_SUM_RANGE_US = {}
NN_O_TIME_SUM_RANGE_US['430_high'] = np.array([8.59, 8.63])
NO_N_TIME_SUM_RANGE_US['430_high'] = np.array([8.65, 8.70])
NN_O_TIME_SUM_RANGE_US['412_high'] = np.array([8.59, 8.63])
NO_N_TIME_SUM_RANGE_US['412_high'] = np.array([8.65, 8.70])
NN_O_TIME_SUM_RANGE_US['430_mid'] = np.array([8.57, 8.61])
NO_N_TIME_SUM_RANGE_US['430_mid'] = np.array([8.625, 8.675])
NN_O_TIME_SUM_RANGE_US['412_mid'] = np.array([8.57, 8.61])
NO_N_TIME_SUM_RANGE_US['412_mid'] = np.array([8.625, 8.675])
NN_O_TIME_SUM_RANGE_US['430_low'] = np.array([8.585, 8.625])
NO_N_TIME_SUM_RANGE_US['430_low'] = np.array([8.645, 8.695])
NN_O_TIME_SUM_RANGE_US['412_low'] = np.array([8.59, 8.63])
NO_N_TIME_SUM_RANGE_US['412_low'] = np.array([8.65, 8.70])
NN_O_TIME_SUM_RANGE_US['560'] = np.array([8.57, 8.61])
NO_N_TIME_SUM_RANGE_US['560'] = np.array([8.625, 8.675])

ANGLE_CUT = 2.7
# %%

FONTSIZE = 10


def figure_wrapper(name):
    fig_handle = plt.figure(name, figsize=(11, 8))
    plt.clf()
    plt.suptitle(name)
    return fig_handle


def text_wrapper(function, text):
    function(text, fontsize=FONTSIZE)


def xlabel_wrapper(text):
    text_wrapper(plt.xlabel, text)


def ylabel_wrapper(text):
    text_wrapper(plt.ylabel, text)


def legend_wrapper(loc='best'):
    plt.legend(loc=loc, fontsize=FONTSIZE)


def title_wrapper(text):
    plt.title(text, fontsize=FONTSIZE)


def colorbar_wrapper(label=None):
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONTSIZE)
    if label is not None:
        cbar.set_label(label)


def tick_fontsize(axis=None):
    if axis is None:
        axis = plt.gca()
    for xyaxis in ['xaxis', 'yaxis']:
        for tick in getattr(axis, xyaxis).get_major_ticks():
            tick.label.set_fontsize(FONTSIZE)


def bar_wrapper(x, y, color=None, label=None, verbose=False):
    if verbose:
        print 'In bar_wrapper()'
        print 'x.shape =', x.shape
        print 'y.shape =', y.shape
    width = np.diff(x).mean(dtype=float)
    plt.bar(x - width/2, y, width=width, linewidth=0, color=color,
            label=label)


def imshow_wrapper(img, x_centers, y_centers=None, kw_args={}):
    x_step = np.diff(x_centers).mean(dtype=float)
    x_min = x_centers.min() - x_step/2
    x_max = x_centers.max() + x_step/2
    if y_centers is None:
        y_min, y_max = x_min, x_max
    else:
        y_step = np.diff(y_centers).mean(dtype=float)
        y_min = y_centers.min() - y_step/2
        y_max = y_centers.max() + y_step/2

    plt.imshow(img, extent=(x_min, x_max, y_min, y_max), origin='lower',
               interpolation='none', **kw_args)


def savefig_wrapper():
    fig_path = 'first_look_figures'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig = plt.gcf()
    fig_name = fig.canvas.get_window_title()
    fig_name = fig_name.replace(' ', '_')
    file_name = '.'.join([fig_name, 'pdf'])
    plt.savefig(os.path.join(fig_path, file_name))
# %%


def make_filters(data, data_name):
    # Set up some event filters
    data.get_filter('two_ions_events', epicea.ff_num_ion_events,
                    {'min_ions': 2, 'max_ions': 2},
                    verbose=verbose)
    data.get_filter('e_start_events', epicea.ff_e_start_events)
    data.get_filter('rand_start_events', epicea.ff_invert,
                    {'filter_name': 'e_start_events'},
                    verbose=verbose)
    data.get_filter('two_ion_e_start_events', epicea.ff_combine,
                    {'filter_name_list':
                        ['two_ions_events', 'e_start_events']},
                    verbose=verbose)
    data.get_filter('two_ion_rand_start_events', epicea.ff_combine,
                    {'filter_name_list':
                        ['two_ions_events', 'rand_start_events']},
                    verbose=verbose)
    data.get_filter('NN_O_events',
                    epicea.ff_two_ions_time_sum_events,
                    {'t_sum_min_us': NN_O_TIME_SUM_RANGE_US[data_name][0],
                     't_sum_max_us': NN_O_TIME_SUM_RANGE_US[data_name][1]},
                    verbose=verbose)
    data.get_filter('NO_N_events',
                    epicea.ff_two_ions_time_sum_events,
                    {'t_sum_min_us': NO_N_TIME_SUM_RANGE_US[data_name][0],
                     't_sum_max_us': NO_N_TIME_SUM_RANGE_US[data_name][1]},
                    verbose=verbose)

    # Ion filters
    data.get_filter('e_start_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'e_start_events'})
    data.get_filter('rand_start_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'rand_start_events'})
    data.get_filter('two_ions_events_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'two_ions_events'},
                    verbose=verbose)
    data.get_filter('two_ion_e_start_events_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'two_ion_e_start_events'},
                    verbose=verbose)
    data.get_filter('two_ion_rand_start_events_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'two_ion_rand_start_events'},
                    verbose=verbose)
    data.get_filter('NN_O_events_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'NN_O_events'},
                    verbose=verbose)

    data.get_filter('NO_N_events_ions',
                    epicea.ff_events_filtered_ions,
                    {'events_filter_name': 'NO_N_events'},
                    verbose=verbose)

    # Electron filters
    data.get_filter('has_position_electrons',
                    epicea.ff_has_position_particles,
                    {'particles': 'electrons'},
                    verbose=verbose)
    data.get_filter('NN_O_events_electrons',
                    epicea.ff_events_filtered_electrons,
                    {'events_filter_name': 'NN_O_events'},
                    verbose=verbose)
    data.get_filter('NO_N_events_electrons',
                    epicea.ff_events_filtered_electrons,
                    {'events_filter_name': 'NO_N_events'},
                    verbose=verbose)
# %%


def plot_ion_tof(data, data_name='', verbose=False):
    """Plot some ions tof spectra."""
    if verbose:
        print '\nPlotting tof spectra'
    figure_wrapper('Ion TOF {}'.format(data_name))
    plt.subplot(211)

    t_axis_us = np.arange(2., 6.5, 0.01)
    t_axis_ns = t_axis_us * 1e6
    if verbose:
        print 'Get tof specturm for all ions.'
    i_tof_all = data.get_i_tof_spectrum(t_axis_ns)
    plt.plot(t_axis_us, i_tof_all, 'k', label='All ions')

    if verbose:
        print 'Get tof specturm for e start ions.'
    i_tof_e_start = data.get_i_tof_spectrum(
        t_axis_ns,
        data.get_filter('e_start_ions'))
    if verbose:
        print 'Get tof specturm for rand start ions.'
    i_tof_random_start = data.get_i_tof_spectrum(
        t_axis_ns, data.get_filter('rand_start_ions'))
    if verbose:
        print 'Plot e start and rand start tof spectra.'
    plt.plot(t_axis_us, i_tof_e_start, 'b', label='electron start ions')
    plt.plot(t_axis_us, i_tof_random_start, 'g', label='random start ions')

    xlabel_wrapper('flight time (us)')
    ylabel_wrapper('number of ions per bin')
    legend_wrapper()
    tick_fontsize()

    plt.subplot(212)
    i_tof_NN_O = data.get_i_tof_spectrum(
        t_axis_ns, data.get_filter('NN_O_events_ions'))
    i_tof_NO_N = data.get_i_tof_spectrum(
        t_axis_ns, data.get_filter('NO_N_events_ions'))

    if verbose:
        print 'Finding random start rescaling factor.'
    rescale_region = slice(t_axis_us.searchsorted(5.5),
                           t_axis_us.searchsorted(6.5))
    random_start_scaling = (i_tof_e_start[rescale_region].sum(dtype=float) /
                            i_tof_random_start[rescale_region].sum())

    plt.plot(t_axis_us, i_tof_e_start, 'b', label='electron start ions')
    plt.plot(t_axis_us, i_tof_random_start * random_start_scaling,
             'g', label='random start ions, rescaled')
    plt.plot(t_axis_us,
             i_tof_e_start - i_tof_random_start*random_start_scaling,
             'r', label='pure coincidence ions')
    plt.plot(t_axis_us, i_tof_NN_O * 5, 'y', label='NN+ O+ ions x 5')
    plt.plot(t_axis_us, i_tof_NO_N * 5, 'm', label='NO+ N+ ions x 5')

    xlabel_wrapper('flight time (us)')
    ylabel_wrapper('number of ions per bin')
    legend_wrapper()
    tick_fontsize()

    savefig_wrapper()
# %%


def plot_ion_image(data, data_name, verbose=False):
    """Plot some ion images."""
    if verbose:
        print '\nPlot some ion image data.'
    x_axis_mm = np.arange(-40, 40, 0.1)
    if verbose:
        print 'Get full image.'
    i_image = data.get_i_xy_image(x_axis_mm)
    if verbose:
        print 'Get e start image.'
    i_image_e_start = data.get_i_xy_image(
        x_axis_mm, ions_filter=data.get_filter('e_start_ions'))
    if verbose:
        print 'Get rand start iage.'
    i_image_random_start = data.get_i_xy_image(
        x_axis_mm, ions_filter=~data.get_filter('e_start_ions'))

    figure_wrapper('Ion image {}'.format(data_name))
    plt.subplot(331)
    imshow_wrapper(i_image, x_axis_mm)
    title_wrapper('all ions')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    colorbar_wrapper()
    tick_fontsize()

    plt.subplot(332)
    imshow_wrapper(i_image_e_start, x_axis_mm)
    title_wrapper('electron start')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    colorbar_wrapper()
    tick_fontsize()

    plt.subplot(334)
    imshow_wrapper(i_image_random_start, x_axis_mm)
    title_wrapper('random start')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    colorbar_wrapper()
    tick_fontsize()

    # Make a slice selection around the center of the image
    selection = slice(x_axis_mm.searchsorted(-5),
                      x_axis_mm.searchsorted(5))
    # Project along x and y
    x_projection = i_image_e_start[selection, :].sum(axis=0)
    y_projection = i_image_e_start[:, selection].sum(axis=1)
    plt.subplot(638)
    plt.plot(x_axis_mm, x_projection, label='normal')
    plt.plot(x_axis_mm[::-1], x_projection, label='reverse')
    title_wrapper('x slice')
    xlabel_wrapper('Position (mm)')
    legend_wrapper()
    tick_fontsize()
    plt.subplot(6, 3, 11)
    plt.plot(x_axis_mm, y_projection, label='nomral')
    plt.plot(x_axis_mm[::-1], y_projection, label='reversed')
    title_wrapper('y slice')
    xlabel_wrapper('Position (mm)')
    legend_wrapper()
    tick_fontsize()

    r_axis_mm = np.linspace(0, 40, 201)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 513)[1::2]

    rt_image = data.get_i_rth_image(r_axis_mm=r_axis_mm,
                                    th_axis_rad=th_axis_rad)
    NN_O_ions = data.get_filter('NN_O_events_ions')
    NO_N_ions = data.get_filter('NO_N_events_ions')
    NN_O_rth_image = data.get_i_rth_image(r_axis_mm, th_axis_rad,
                                          ions_filter=NN_O_ions)
    NO_N_rth_image = data.get_i_rth_image(r_axis_mm, th_axis_rad,
                                          ions_filter=NO_N_ions)
    NN_O_r_proj = NN_O_rth_image.sum(axis=0)
    NO_N_r_proj = NO_N_rth_image.sum(axis=0)

    plt.subplot(333)
    imshow_wrapper(rt_image, r_axis_mm, th_axis_rad, {'aspect': 'auto'})
    tick_fontsize()
    title_wrapper('electron start polar')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Angle (rad)')

    plt.subplot(336)
    r_projection = rt_image.sum(axis=0)
    plt.plot(r_axis_mm, r_projection, label='e start')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Number of counts')
    legend_wrapper()
    tick_fontsize()

    plt.subplot(337)
    plt.plot(r_axis_mm, NN_O_r_proj, 'y', label='NN+ O+ ions')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Number of counts')
    legend_wrapper()
    tick_fontsize()

    plt.subplot(338)
    r_projection = rt_image.sum(axis=0)
    plt.plot(r_axis_mm, NO_N_r_proj, 'm', label='NO+ N+ ions')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Number of counts')
    legend_wrapper()
    tick_fontsize()

    savefig_wrapper()
# %%


def calc_angle_diffs_hist(angles, th_axis_rad):
    # Fold the angles
    angle_diffs = np.abs(np.diff(angles)[::2])
    large_angle_diffs = angle_diffs > np.pi
    angle_diffs[large_angle_diffs] = 2*np.pi - angle_diffs[large_angle_diffs]
    hist = epicea.center_histogram(angle_diffs[np.isfinite(angle_diffs)],
                                   th_axis_rad)

    return hist, angle_diffs


def calc_radius_fraction_hist(r, r_frac_axis):
    r_pairs = r.reshape(-1, 2)
    r_pairs.sort(axis=1)
    r_frac = r_pairs[:, 1] / r_pairs[:, 0]
    valid = np.isfinite(r_frac)
    return epicea.center_histogram(r_frac[valid], r_frac_axis)


def plot_two_ion_corelations(data, data_name, verbose=False):

    if verbose:
        print 'In plot_two_ion_corelations.'
    # Get some ions filters
    double_ions = data.get_filter('two_ions_events_ions')
    double_ions_e_start = data.get_filter('two_ion_e_start_events_ions')
    double_ions_rand_start = data.get_filter('two_ion_rand_start_events_ions')
    NN_O_events_ions = data.get_filter('NN_O_events_ions')
    NO_N_events_ions = data.get_filter('NO_N_events_ions')

    # Plot distribution of nomber of ions
    num_ions = data.events.num_i.value

    num_ions_axis = np.arange(5)
    hist_ions_all = epicea.center_histogram(num_ions, num_ions_axis)
    hist_ions_e_start = epicea.center_histogram(
        num_ions[data.get_filter('e_start_events')], num_ions_axis)
    hist_ions_rand_start = epicea.center_histogram(
        num_ions[data.get_filter('rand_start_events')], num_ions_axis)

    figure_wrapper('Two-ion correlations {}'.format(data_name))
    plt.subplot(221)
    plt.plot(num_ions_axis, hist_ions_all, 'o-', label='All events')
    plt.plot(num_ions_axis, hist_ions_rand_start, 'o-', label='random start')
    plt.plot(num_ions_axis, hist_ions_e_start, 'o-', label='e start')
    legend_wrapper()
    title_wrapper('Ion number distributions.')
    xlabel_wrapper('Number of ions per event')
    ylabel_wrapper('Number of events')
    tick_fontsize()

    # Look at some two ions, e start event correlations
    # Angle information
    # figure_wrapper('two ion angles {}'.format(data_name))
    plt.subplot(222)
    if verbose:
        print 'Get angles.'
    th_e_start = data.ions.pos_t[double_ions_e_start]
    th_NN_O = data.ions.pos_t[NN_O_events_ions]
    th_NO_N = data.ions.pos_t[NO_N_events_ions]

    th_axis_rad = np.linspace(0, np.pi, 513)[1::2]
    th_hist_e_start, th_diff_e_start = calc_angle_diffs_hist(th_e_start,
                                                             th_axis_rad)
    th_hist_NN_O, _ = calc_angle_diffs_hist(th_NN_O, th_axis_rad)
    th_hist_NO_N, _ = calc_angle_diffs_hist(th_NO_N, th_axis_rad)

    bar_wrapper(th_axis_rad, th_hist_e_start, label='e start ion pairs')
    sl = slice(th_axis_rad.searchsorted(ANGLE_CUT), None)
    bar_wrapper(th_axis_rad[sl], th_hist_e_start[sl], 'r', label='selection')
    bar_wrapper(th_axis_rad, th_hist_NN_O, 'y', label='NN+ O+ ion pairs')
    bar_wrapper(th_axis_rad, th_hist_NO_N, 'm', label='NO+ N+ ion pairs')
#    plt.hist(angle_diff[np.isfinite(angle_diff)],
#             bins=np.linspace(-0.1, np.pi, 256))
    xlabel_wrapper('Angle difference between tow ions (rad)')
    ylabel_wrapper('Number of ion pairs')
    title_wrapper('Two ion angles.')
    legend_wrapper()
    tick_fontsize()

    if verbose:
        print '{} events with two ions and electron start identified.'.format(
            data.get_filter('two_ion_e_start_events').sum())
        print '{} valid angle diffs found.'.format(th_hist_e_start.sum())

    # Radial information
    # figure_wrapper('radii {}'.format(data_name))
    plt.subplot(223)
    if verbose:
        print 'Get radii.'
    r_frac_axis = np.linspace(1., 3.5, 257)[1::2]
    r_e_start = data.ions.pos_r[double_ions_e_start]
    r_NN_O = data.ions.pos_r[NN_O_events_ions]
    r_NO_N = data.ions.pos_r[NO_N_events_ions]

    r_frac_e_start_hist = calc_radius_fraction_hist(r_e_start, r_frac_axis)
    r_frac_selection_hist = calc_radius_fraction_hist(
        r_e_start.reshape(-1, 2)[th_diff_e_start > ANGLE_CUT, :], r_frac_axis)
    r_frac_NN_O_hist = calc_radius_fraction_hist(r_NN_O, r_frac_axis)
    r_frac_NO_N_hist = calc_radius_fraction_hist(r_NO_N, r_frac_axis)

    bar_wrapper(r_frac_axis, r_frac_e_start_hist, label='e start')
    bar_wrapper(r_frac_axis, r_frac_selection_hist, 'r', label='selection')
    bar_wrapper(r_frac_axis, r_frac_NN_O_hist, 'y', label='NN+ O+ ions')
    bar_wrapper(r_frac_axis, r_frac_NO_N_hist, 'm', label='NO+ N+ ions')
    # plt.hist(r_frac, bins=np.linspace(0.9, 3., 256))
    xlabel_wrapper('Radial quotient for two ions r1/r2')
    ylabel_wrapper('Number of ion pairs')
    title_wrapper('Radius quotient')
    legend_wrapper()
    tick_fontsize()
    savefig_wrapper()

    ###########################
    # TOF-TOF correlation plot
    figure_wrapper('tof-tof hist {}'.format(data_name))
    t_axis_us = np.linspace(3.3, 5.3, 512)
    t_axis = t_axis_us * 1e6

#    names = ['e start', 'all', 'random start']
#    for i, ions_filter in enumerate([double_ions_e_start,
#                                     double_ions,
#                                     double_ions_rand_start]):
#        plt.subplot(2, 2, i+1)
    names = ['electrons']
    for i, ions_filter in enumerate([double_ions_e_start]):
        i_tof = data.ions.tof_falling_edge[ions_filter]
        i_tof = i_tof.reshape(-1, 2)
        i_tof.sort(axis=1)
        i_tof = i_tof[np.isfinite(i_tof).sum(1) == 2, :]

        tof_tof_hist = epicea.center_histogram_2d(i_tof[:, 0], i_tof[:, 1],
                                                  t_axis)
        tof_tof_sym_hist = tof_tof_hist + tof_tof_hist.T
#        imshow_wrapper(tof_tof_sym_hist, t_axis_us)
        imshow_wrapper(np.log(tof_tof_sym_hist+1), t_axis_us)
        plt.plot(t_axis_us, NN_O_TIME_SUM_RANGE_US[data_name][0] - t_axis_us,
                 'y', label='NN+ O+ selection')
        plt.plot(t_axis_us, NN_O_TIME_SUM_RANGE_US[data_name][1] - t_axis_us,
                 'y')
        plt.plot(t_axis_us, NO_N_TIME_SUM_RANGE_US[data_name][0] - t_axis_us,
                 'm', label='NO+ N+ selection')
        plt.plot(t_axis_us, NO_N_TIME_SUM_RANGE_US[data_name][1] - t_axis_us,
                 'm')
        title_wrapper(names[i])
        plt.axis([t_axis_us.min(), t_axis_us.max(),
                  t_axis_us.min(), t_axis_us.max()])
        ylabel_wrapper('Time of flight (us)')
        xlabel_wrapper('Time of flight (us)')
        legend_wrapper(loc='center right')
        colorbar_wrapper('log(counts + 1)')
        tick_fontsize()

#        if i == 0:
#            tof_tof_hist_e_start = tof_tof_sym_hist.copy()
#        if i == 2:
#            tof_tof_hist_random = tof_tof_sym_hist.copy()
#
#    sl = slice(t_axis.searchsorted(5.5e6))
#    factor = (tof_tof_hist_e_start[sl, :].sum() /
#              tof_tof_hist_random[sl, :].sum())
#    pure = tof_tof_hist_e_start - tof_tof_hist_random * factor
#
#    plt.subplot(224)
#    imshow_wrapper(pure, t_axis_us)
#    tick_fontsize()
    savefig_wrapper()
# %%


def plot_e_spec(data, data_name, verbose=False):
    """First view of the electron spectra."""

    if verbose:
        print 'Plotting electron spectra for {}.'.format(data_name)
    figure_wrapper('Electron data {}'.format(data_name))

    valid_pos = data.get_filter('has_position_electrons')

    NN_O_events_electrons = data.get_filter('NN_O_events_electrons')

    NO_N_events_electrons = data.get_filter('NO_N_events_electrons')

    if verbose:
        print 'Valid positions for {} electrons.'.format(valid_pos.sum())

    x_axis_mm = np.linspace(-23, 23, 512)
    r_axis_mm = np.linspace(0, 23, 513)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 513)[1::2]
    xy_center_slice = slice(x_axis_mm.searchsorted(-3),
                            x_axis_mm.searchsorted(3, side='right'))

    e_all_image_xy = data.get_e_xy_image(x_axis_mm)
    e_all_x_slice = e_all_image_xy[xy_center_slice, :].sum(axis=0)
    e_all_y_slice = e_all_image_xy[:, xy_center_slice].sum(axis=1)
    e_all_image_rth = data.get_e_rth_image(r_axis_mm, th_axis_rad)
    e_all_radial_dist = e_all_image_rth.sum(axis=0)

#    e_NN_O_image_xy = data.get_e_xy_image(
#        x_axis_mm, electrons_filter=NN_O_events_electrons)
#    e_NO_N_image_xy = data.get_e_xy_image(
#        x_axis_mm, electrons_filter=NO_N_events_electrons)

    e_NN_O_image_rth = data.get_e_rth_image(
        r_axis_mm, th_axis_rad, electrons_filter=NN_O_events_electrons)
    e_NN_O_radial_dist = e_NN_O_image_rth.sum(axis=0)
    e_NO_N_image_rth = data.get_e_rth_image(
        r_axis_mm, th_axis_rad, electrons_filter=NO_N_events_electrons)
    e_NO_N_radial_dist = e_NO_N_image_rth.sum(axis=0)

    plt.subplot(231)
    imshow_wrapper(e_all_image_xy, x_axis_mm)
    title_wrapper('2D image')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    tick_fontsize()

    plt.subplot(4, 3, 7)
    plt.plot(x_axis_mm, e_all_x_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_all_x_slice, label='flipped')
    title_wrapper('x slice')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of ions')
    tick_fontsize()
    legend_wrapper()

    plt.subplot(4, 3, 10)
    plt.plot(x_axis_mm, e_all_y_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_all_y_slice, label='flipped')
    title_wrapper('y slice')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of ions')
    tick_fontsize()
    legend_wrapper()

    plt.subplot(232)
    imshow_wrapper(e_all_image_rth, r_axis_mm, th_axis_rad, {'aspect': 'auto'})
    title_wrapper('2D image')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Angle (rad)')
    tick_fontsize()

    plt.subplot(235)
    plt.plot(r_axis_mm, e_all_radial_dist)
    title_wrapper('Radial distribution')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of electrons')
    tick_fontsize()

#    plt.subplot(233)
#    imshow_wrapper(e_NN_O_image_xy, x_axis_mm)
#    title_wrapper('NN+ O+ events')
#    xlabel_wrapper('Position (mm)')
#    ylabel_wrapper('Position (mm)')
#    tick_fontsize()
#
#    plt.subplot(236)
#    imshow_wrapper(e_NO_N_image_xy, x_axis_mm)
#    title_wrapper('NO+ N+ events')
#    xlabel_wrapper('Position (mm)')
#    ylabel_wrapper('Position (mm)')
#    tick_fontsize()
#
#    plt.subplot(233)
#    imshow_wrapper(e_NN_O_image_rth, r_axis_mm, th_axis_rad)
#    title_wrapper('NN+ O+ events')
#    xlabel_wrapper('Position (mm)')
#    ylabel_wrapper('Angle (rad)')
#    tick_fontsize()
#
#    plt.subplot(236)
#    imshow_wrapper(e_NO_N_image_rth, r_axis_mm, th_axis_rad)
#    title_wrapper('NO+ N+ events')
#    xlabel_wrapper('Position (mm)')
#    ylabel_wrapper('Angle (rad)')
#    tick_fontsize()

    plt.subplot(233)
    plt.plot(r_axis_mm, e_NN_O_radial_dist, 'y')
    title_wrapper('NN+ O+ events')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of electrons')
    tick_fontsize()

    plt.subplot(236)
    plt.plot(r_axis_mm, e_NO_N_radial_dist, 'm')
    title_wrapper('NO+ N+ events')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of electrons')
    tick_fontsize()

    savefig_wrapper()
# %%

if __name__ == '__main__':
    # %%

    dataset_info = {}
#    dataset_info['430_high'] = {
#        'data_path': '../data/ExportedData/N2O_0029_KE373_hv430eV/',
#        'h5_path': 'h5_data/N20_430_high.h5'}
#    dataset_info['412_high'] = {
#        'data_path': '../data/ExportedData/N2O_0031_KE373_hv412eV/',
#        'h5_path': 'h5_data/N20_412_high.h5'}
    dataset_info['430_mid'] = {
        'data_path': '../data/ExportedData/N2O_366PE_430eV_0014/',
        'h5_path': 'h5_data/N20_430_mid.h5'}
#    dataset_info['412_mid'] = {
#        'data_path': '../data/ExportedData/N2O_366PE_4119eV_combined/',
#        'h5_path': 'h5_data/N20_412_mid.h5'}
#    dataset_info['430_low'] = {
#        'data_path': '../data/ExportedData/N2O_KE357_hv430p9_0047/',
#        'h5_path': 'h5_data/N20_430_low.h5'}
#    dataset_info['412_low'] = {
#        'data_path': '../data/ExportedData/N2O_KE357_hv412p9_0049/',
#        'h5_path': 'h5_data/N20_412_low.h5'}
#    dataset_info['560'] = {
#        'data_path': '../data/ExportedData/N2O_500PE_560eV_0017/',
#        'h5_path': 'h5_data/N20_560.h5'}

    verbose = True
    # %%

    if 'data' not in locals():
        data = {}
        for data_name, data_info in dataset_info.iteritems():
            if data_name not in data:
                if verbose:
                    print '\n##############################'
                    print 'Load data set {} -> {}'.format(
                        data_name, data_info)
                data[data_name] = epicea.DataSet(verbose=verbose, **data_info)
                data[data_name].ions.correct_center(
                    ION_VMI_OFFSET[data_name][0],
                    ION_VMI_OFFSET[data_name][1])
                data[data_name].electrons.correct_center(
                    ELECTRON_OFFSET[data_name][0],
                    ELECTRON_OFFSET[data_name][1])
                make_filters(data[data_name], data_name)
    # %%

#    for data_name, dataset in data.iteritems():
#        plot_ion_tof(dataset, data_name, verbose=verbose)
    # %%

#    for data_name, dataset in data.iteritems():
#        plot_ion_image(dataset, data_name, verbose=verbose)
    # %%

#    for data_name, dataset in data.iteritems():
#        plot_two_ion_corelations(dataset, data_name, verbose=verbose)
    # %%

    for data_name, dataset in data.iteritems():
        plot_e_spec(dataset, data_name, verbose=verbose)
