# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 12:03:56 2015

@author: Anton O. Lindahl
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

import plt_func
import global_parameters as glob
if 'epicea' not in locals():
    import epicea

plt.ion()

ANGLE_CUT = 2.7

# %%


def make_filters(data, verbose=False):
    # Set up some event filters
    data.get_filter('two_ions_events', epicea.ff.num_ion_events,
                    {'min_ions': 2, 'max_ions': 2},
                    verbose=verbose)
    data.get_filter('e_start_events', epicea.ff.e_start_events,
                    verbose=verbose)
    data.get_filter('rand_start_events', epicea.ff.invert,
                    {'filter_name': 'e_start_events'},
                    verbose=verbose)
    data.get_filter('two_ion_e_start_events', epicea.ff.combine,
                    {'filter_name_list':
                        ['two_ions_events', 'e_start_events']},
                    verbose=verbose)
    data.get_filter('two_ion_rand_start_events', epicea.ff.combine,
                    {'filter_name_list':
                        ['two_ions_events', 'rand_start_events']},
                    verbose=verbose)
    data.get_filter('NN_O_events',
                    epicea.ff.two_ions_time_sum_events,
                    {'t_sum_min_us':
                        glob.NN_O_TIME_SUM_RANGE_US[data.name()][0],
                     't_sum_max_us':
                         glob.NN_O_TIME_SUM_RANGE_US[data.name()][1]},
                    verbose=verbose)
    data.get_filter('NO_N_events',
                    epicea.ff.two_ions_time_sum_events,
                    {'t_sum_min_us':
                        glob.NO_N_TIME_SUM_RANGE_US[data.name()][0],
                     't_sum_max_us':
                         glob.NO_N_TIME_SUM_RANGE_US[data.name()][1]},
                    verbose=verbose)

    # Ion filters
    data.get_filter('e_start_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'e_start_events'},
                    verbose=verbose)
    data.get_filter('rand_start_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'rand_start_events'},
                    verbose=verbose)
    data.get_filter('two_ions_events_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'two_ions_events'},
                    verbose=verbose)
    data.get_filter('two_ion_e_start_events_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'two_ion_e_start_events'},
                    verbose=verbose)
    data.get_filter('two_ion_rand_start_events_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'two_ion_rand_start_events'},
                    verbose=verbose)
    data.get_filter('NN_O_events_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'NN_O_events'},
                    verbose=verbose)

    data.get_filter('NO_N_events_ions',
                    epicea.ff.events_filtered_ions,
                    {'events_filter_name': 'NO_N_events'},
                    verbose=verbose)

    # Electron filters
    data.get_filter('has_position_electrons',
                    epicea.ff.has_position_particles,
                    {'particles': 'electrons'},
                    verbose=verbose)
    data.get_filter('NN_O_events_electrons',
                    epicea.ff.events_filtered_electrons,
                    {'events_filter_name': 'NN_O_events'},
                    verbose=verbose)
    data.get_filter('NO_N_events_electrons',
                    epicea.ff.events_filtered_electrons,
                    {'events_filter_name': 'NO_N_events'},
                    verbose=verbose)
# %%


def plot_ion_tof(data, verbose=False):
    """Plot some ions tof spectra."""
    if verbose:
        print '\nPlotting tof spectra'
    plt_func.figure_wrapper('Ion TOF {}'.format(data.name()))
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

    plt_func.xlabel_wrapper('flight time (us)')
    plt_func.ylabel_wrapper('number of ions per bin')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

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

    plt_func.xlabel_wrapper('flight time (us)')
    plt_func.ylabel_wrapper('number of ions per bin')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

    plt_func.savefig_wrapper()
# %%


def plot_ion_image(data, verbose=False):
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

    plt_func.figure_wrapper('Ion image {}'.format(data.name()))
    plt.subplot(331)
    plt_func.imshow_wrapper(i_image, x_axis_mm)
    plt_func.title_wrapper('all ions')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Position (mm)')
    plt_func.colorbar_wrapper()
    plt_func.tick_fontsize()

    plt.subplot(332)
    plt_func.imshow_wrapper(i_image_e_start, x_axis_mm)
    plt_func.title_wrapper('electron start')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Position (mm)')
    plt_func.colorbar_wrapper()
    plt_func.tick_fontsize()

    plt.subplot(334)
    plt_func.imshow_wrapper(i_image_random_start, x_axis_mm)
    plt_func.title_wrapper('random start')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Position (mm)')
    plt_func.colorbar_wrapper()
    plt_func.tick_fontsize()

    # Make a slice selection around the center of the image
    selection = slice(x_axis_mm.searchsorted(-5),
                      x_axis_mm.searchsorted(5))
    # Project along x and y
    x_projection = i_image_e_start[selection, :].sum(axis=0)
    y_projection = i_image_e_start[:, selection].sum(axis=1)
    plt.subplot(638)
    plt.plot(x_axis_mm, x_projection, label='normal')
    plt.plot(x_axis_mm[::-1], x_projection, label='reverse')
    plt_func.title_wrapper('x slice')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()
    plt.subplot(6, 3, 11)
    plt.plot(x_axis_mm, y_projection, label='nomral')
    plt.plot(x_axis_mm[::-1], y_projection, label='reversed')
    plt_func.title_wrapper('y slice')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

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
    plt_func.imshow_wrapper(rt_image, r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'})
    plt_func.tick_fontsize()
    plt_func.title_wrapper('electron start polar')
    plt_func.xlabel_wrapper('Radius (mm)')
    plt_func.ylabel_wrapper('Angle (rad)')

    plt.subplot(336)
    r_projection = rt_image.sum(axis=0)
    plt.plot(r_axis_mm, r_projection, label='e start')
    plt_func.xlabel_wrapper('Radius (mm)')
    plt_func.ylabel_wrapper('Number of counts')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

    plt.subplot(337)
    plt.plot(r_axis_mm, NN_O_r_proj, 'y', label='NN+ O+ ions')
    plt_func.xlabel_wrapper('Radius (mm)')
    plt_func.ylabel_wrapper('Number of counts')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

    plt.subplot(338)
    r_projection = rt_image.sum(axis=0)
    plt.plot(r_axis_mm, NO_N_r_proj, 'm', label='NO+ N+ ions')
    plt_func.xlabel_wrapper('Radius (mm)')
    plt_func.ylabel_wrapper('Number of counts')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

    plt_func.savefig_wrapper()
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


def plot_two_ion_corelations(data, verbose=False):
    if verbose:
        print 'In plot_two_ion_corelations.'
    # Get some ions filters
    data.get_filter('two_ions_events_ions')
    double_ions_e_start = data.get_filter('two_ion_e_start_events_ions')
    data.get_filter('two_ion_rand_start_events_ions')
    NN_O_events_ions = data.get_filter('NN_O_events_ions', verbose=verbose)
    NO_N_events_ions = data.get_filter('NO_N_events_ions', verbose=verbose)

    # Plot distribution of nomber of ions
    num_ions = data.events.num_i.value

    num_ions_axis = np.arange(5)
    hist_ions_all = epicea.center_histogram(num_ions, num_ions_axis)
    hist_ions_e_start = epicea.center_histogram(
        num_ions[data.get_filter('e_start_events')], num_ions_axis)
    hist_ions_rand_start = epicea.center_histogram(
        num_ions[data.get_filter('rand_start_events')], num_ions_axis)

    plt_func.figure_wrapper('Two-ion correlations {}'.format(data.name()))
    plt.subplot(221)
    plt.plot(num_ions_axis, hist_ions_all, 'o-', label='All events')
    plt.plot(num_ions_axis, hist_ions_rand_start, 'o-', label='random start')
    plt.plot(num_ions_axis, hist_ions_e_start, 'o-', label='e start')
    plt_func.legend_wrapper()
    plt_func.title_wrapper('Ion number distributions.')
    plt_func.xlabel_wrapper('Number of ions per event')
    plt_func.ylabel_wrapper('Number of events')
    plt_func.tick_fontsize()

    # Look at some two ions, e start event correlations
    # Angle information
    # plt_func.figure_wrapper('two ion angles {}'.format(data.name()))
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

    plt_func.bar_wrapper(th_axis_rad, th_hist_e_start,
                         label='e start ion pairs')
    sl = slice(th_axis_rad.searchsorted(ANGLE_CUT), None)
    plt_func.bar_wrapper(th_axis_rad[sl], th_hist_e_start[sl], 'r',
                         label='selection')
    plt_func.bar_wrapper(th_axis_rad, th_hist_NN_O, 'y',
                         label='NN+ O+ ion pairs')
    plt_func.bar_wrapper(th_axis_rad, th_hist_NO_N, 'm',
                         label='NO+ N+ ion pairs')
#    plt.hist(angle_diff[np.isfinite(angle_diff)],
#             bins=np.linspace(-0.1, np.pi, 256))
    plt_func.xlabel_wrapper('Angle difference between tow ions (rad)')
    plt_func.ylabel_wrapper('Number of ion pairs')
    plt_func.title_wrapper('Two ion angles.')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()

    if verbose:
        print '{} events with two ions and electron start identified.'.format(
            data.get_filter('two_ion_e_start_events').sum())
        print '{} valid angle diffs found.'.format(th_hist_e_start.sum())

    # Radial information
    # plt_func.figure_wrapper('radii {}'.format(data.name()))
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

    plt_func.bar_wrapper(r_frac_axis, r_frac_e_start_hist, label='e start')
    plt_func.bar_wrapper(r_frac_axis, r_frac_selection_hist, 'r',
                         label='selection')
    plt_func.bar_wrapper(r_frac_axis, r_frac_NN_O_hist, 'y',
                         label='NN+ O+ ions')
    plt_func.bar_wrapper(r_frac_axis, r_frac_NO_N_hist, 'm',
                         label='NO+ N+ ions')
    # plt.hist(r_frac, bins=np.linspace(0.9, 3., 256))
    plt_func.xlabel_wrapper('Radial quotient for two ions r1/r2')
    plt_func.ylabel_wrapper('Number of ion pairs')
    plt_func.title_wrapper('Radius quotient')
    plt_func.legend_wrapper()
    plt_func.tick_fontsize()
    plt_func.savefig_wrapper()

    ###########################
    # TOF-TOF correlation plot
    plt_func.figure_wrapper('tof-tof hist {}'.format(data.name()))
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
#        plt_func.imshow_wrapper(tof_tof_sym_hist, t_axis_us)
        plt_func.imshow_wrapper(np.log(tof_tof_sym_hist+1), t_axis_us)
        plt.plot(t_axis_us,
                 glob.NN_O_TIME_SUM_RANGE_US[data.name()][0] - t_axis_us,
                 'y', label='NN+ O+ selection')
        plt.plot(t_axis_us,
                 glob.NN_O_TIME_SUM_RANGE_US[data.name()][1] - t_axis_us,
                 'y')
        plt.plot(t_axis_us,
                 glob.NO_N_TIME_SUM_RANGE_US[data.name()][0] - t_axis_us,
                 'm', label='NO+ N+ selection')
        plt.plot(t_axis_us,
                 glob.NO_N_TIME_SUM_RANGE_US[data.name()][1] - t_axis_us,
                 'm')
        plt_func.title_wrapper(names[i])
        plt.axis([t_axis_us.min(), t_axis_us.max(),
                  t_axis_us.min(), t_axis_us.max()])
        plt_func.ylabel_wrapper('Time of flight (us)')
        plt_func.xlabel_wrapper('Time of flight (us)')
        plt_func.legend_wrapper(loc='center right')
        plt_func.colorbar_wrapper('log(counts + 1)')
        plt_func.tick_fontsize()

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
#    plt_func.imshow_wrapper(pure, t_axis_us)
#    plt_func.tick_fontsize()
    plt_func.savefig_wrapper()
# %%


def plot_e_spec(data, verbose=False):
    """First view of the electron spectra."""

    if verbose:
        print 'Plotting electron spectra for {}.'.format(data.name())
    plt_func.figure_wrapper('Electron data {}'.format(data.name()))

    valid_pos = data.get_filter('has_position_electrons',
                                verbose=verbose)

    NN_O_events_electrons = data.get_filter('NN_O_events_electrons',
                                            verbose=verbose)

    NO_N_events_electrons = data.get_filter('NO_N_events_electrons',
                                            verbose=verbose)

    if verbose:
        print 'Valid positions for {} electrons.'.format(valid_pos.sum())

    x_axis_mm = np.linspace(-23, 23, 512)
    r_axis_mm = np.linspace(0, 23, 513)[1::2]
    th_axis_rad = np.linspace(0, 2*np.pi, 513)[1::2]
    xy_center_slice = slice(x_axis_mm.searchsorted(-3),
                            x_axis_mm.searchsorted(3, side='right'))

#    e_all_image_xy = data.get_e_xy_image(x_axis_mm, verbose=verbose)
    e_all_image_xy, _ = data.get_e_xy_image(
        x_axis_mm, verbose=verbose,
        electrons_filter=data.electrons.event_id.value > 0.9 *
        data.events.len()
        )
    e_all_x_slice = e_all_image_xy[xy_center_slice, :].sum(axis=0)
    e_all_y_slice = e_all_image_xy[:, xy_center_slice].sum(axis=1)
#    e_all_image_rth = data.get_e_rth_image(r_axis_mm, th_axis_rad,
#                                           verbose=verbose)
    e_all_image_rth, _ = data.get_e_rth_image(
        r_axis_mm, th_axis_rad, verbose=verbose,
        electrons_filter=data.electrons.event_id.value > 0.9 *
        data.events.len()
        )
    e_all_radial_dist = e_all_image_rth.sum(axis=0)

#    e_NN_O_image_xy = data.get_e_xy_image(
#        x_axis_mm, electrons_filter=NN_O_events_electrons)
#    e_NO_N_image_xy = data.get_e_xy_image(
#        x_axis_mm, electrons_filter=NO_N_events_electrons)

    e_NN_O_image_rth, _ = data.get_e_rth_image(
        r_axis_mm, th_axis_rad, electrons_filter=NN_O_events_electrons,
        verbose=verbose)
    e_NN_O_radial_dist = e_NN_O_image_rth.sum(axis=0)
    e_NO_N_image_rth, _ = data.get_e_rth_image(
        r_axis_mm, th_axis_rad, electrons_filter=NO_N_events_electrons,
        verbose=verbose)
    e_NO_N_radial_dist = e_NO_N_image_rth.sum(axis=0)

    plt.subplot(231)
    plt_func.imshow_wrapper(e_all_image_xy, x_axis_mm)
    plt_func.title_wrapper('2D image')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Position (mm)')
    plt_func.tick_fontsize()

    plt.subplot(4, 3, 7)
    plt.plot(x_axis_mm, e_all_x_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_all_x_slice, label='flipped')
    plt_func.title_wrapper('x slice')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Number of ions')
    plt_func.tick_fontsize()
    plt_func.legend_wrapper()

    plt.subplot(4, 3, 10)
    plt.plot(x_axis_mm, e_all_y_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_all_y_slice, label='flipped')
    plt_func.title_wrapper('y slice')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Number of ions')
    plt_func.tick_fontsize()
    plt_func.legend_wrapper()

    plt.subplot(232)
    plt_func.imshow_wrapper(e_all_image_rth, r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'})
    plt_func.title_wrapper('2D image')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Angle (rad)')
    plt_func.tick_fontsize()

    plt.subplot(235)
    plt.plot(r_axis_mm, e_all_radial_dist)
    plt_func.title_wrapper('Radial distribution')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Number of electrons')
    plt_func.tick_fontsize()

#    plt.subplot(233)
#    plt_func.imshow_wrapper(e_NN_O_image_xy, x_axis_mm)
#    plt_func.title_wrapper('NN+ O+ events')
#    plt_func.xlabel_wrapper('Position (mm)')
#    plt_func.ylabel_wrapper('Position (mm)')
#    plt_func.tick_fontsize()
#
#    plt.subplot(236)
#    plt_func.imshow_wrapper(e_NO_N_image_xy, x_axis_mm)
#    plt_func.title_wrapper('NO+ N+ events')
#    plt_func.xlabel_wrapper('Position (mm)')
#    plt_func.ylabel_wrapper('Position (mm)')
#    plt_func.tick_fontsize()
#
#    plt.subplot(233)
#    plt_func.imshow_wrapper(e_NN_O_image_rth, r_axis_mm, th_axis_rad)
#    plt_func.title_wrapper('NN+ O+ events')
#    plt_func.xlabel_wrapper('Position (mm)')
#    plt_func.ylabel_wrapper('Angle (rad)')
#    plt_func.tick_fontsize()
#
#    plt.subplot(236)
#    plt_func.imshow_wrapper(e_NO_N_image_rth, r_axis_mm, th_axis_rad)
#    plt_func.title_wrapper('NO+ N+ events')
#    plt_func.xlabel_wrapper('Position (mm)')
#    plt_func.ylabel_wrapper('Angle (rad)')
#    plt_func.tick_fontsize()

    plt.subplot(233)
    plt.plot(r_axis_mm, e_NN_O_radial_dist, 'y')
    plt_func.title_wrapper(r'N$_2$$^+$ + O$^+$ events')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Number of electrons')
    plt_func.tick_fontsize()

    plt.subplot(236)
    plt.plot(r_axis_mm, e_NO_N_radial_dist, 'm')
    plt_func.title_wrapper('NO$^+$ + N$^+$ events')
    plt_func.xlabel_wrapper('Position (mm)')
    plt_func.ylabel_wrapper('Number of electrons')
    plt_func.tick_fontsize()

    plt.tight_layout()
    plt_func.savefig_wrapper()
# %%

if __name__ == '__main__':
    # %%

    verbose = True
    if 'data_list' not in locals():
        data_list = epicea.DataSetList()

    raw_data_base_path = '../data/ExportedData'

    # data_info list: [photon_energy, center_energy, data_path]
    data_info = [
#        [430, 373, 'N2O_0029_KE373_hv430eV'],
        [412, 373, 'N2O_0031_KE373_hv412eV'],
#        [430, 366, 'N2O_366PE_430eV_0014'],
#        [412, 366, 'N2O_366PE_4119eV_combined'],
#        [430, 357, 'N2O_KE357_hv430p9_0047'],
#        [412, 357, 'N2O_KE357_hv412p9_0049'],
#        [560, 500, 'N2O_500PE_560eV_0017']
        ]

    for photon_energy, center_energy, data_path in data_info:
        name = '{}_{}'.format(photon_energy,
                              center_energy).replace('.', '_')
        h5_name = 'h5_data/N2O_{}.h5'.format(name)
        data_list.add_dataset(
            name=name,
            h5_path=h5_name,
            raw_data_path=os.path.join(raw_data_base_path, data_path),
            photon_energy=photon_energy,
            electron_center_energy=center_energy,
            verbose=verbose)
# %%

    calibration_373 = epicea.ElectronEnergyCalibration()
    calibration_373.load_from_file('test_data/calib_373.h5')
    for data in data_list:
        if '373' in data.name():
            data.calculate_electron_energy(calibration_373)
# %%

    for data in data_list:
        data.ions.correct_center(
            glob.ION_VMI_OFFSET[data.name()][0],
            glob.ION_VMI_OFFSET[data.name()][1])
        data.electrons.correct_center(
            glob.ELECTRON_OFFSET[data.name()][0],
            glob.ELECTRON_OFFSET[data.name()][1])
#        data.electrons.recalculate_polar_coordinates()
# %%

    for data in data_list:
        make_filters(data, verbose=True)
    # %%

#    for data in data_list:
#        plot_ion_tof(data, verbose=verbose)
    # %%

#    for data in data_list:
#        plot_ion_image(data, verbose=verbose)
    # %%

#    for data in data_list:
#        plot_two_ion_corelations(data, verbose=verbose)
    # %%

    for data in data_list:
        plot_e_spec(data, verbose=False)
