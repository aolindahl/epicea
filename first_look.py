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
ELECTRON_X_OFFSET = -0.2
ELECTRON_Y_OFFSET = -0.1

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


def legend_wrapper():
    plt.legend(loc='best', fontsize=FONTSIZE)


def title_wrapper(text):
    plt.title(text, fontsize=FONTSIZE)


def colorbar_wrapper():
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=FONTSIZE)


def tick_fontsize(axis=None):
    if axis is None:
        axis = plt.gca()
    for xyaxis in ['xaxis', 'yaxis']:
        for tick in getattr(axis, xyaxis).get_major_ticks():
            tick.label.set_fontsize(FONTSIZE)


def bar_wrapper(x, y, color=None):
    width = np.diff(x).mean(dtype=float)
    plt.bar(x - width/2, y, width=width, linewidth=0, color=color)


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

    data.get_filter('e_start_events', epicea.ff_e_start_events)
    if verbose:
        print 'Get tof specturm for e start ions.'
    i_tof_e_start = data.get_i_tof_spectrum(
        t_axis_ns,
        data.get_filter('e_start_ions',
                        epicea.ff_events_filtered_ions,
                        {'events_filter_name': 'e_start_events'}))
    if verbose:
        print 'Get tof specturm for rand start ions.'
    i_tof_random_start = data.get_i_tof_spectrum(
        t_axis_ns, ~data.get_filter('e_start_ions'))
    if verbose:
        print 'Plot e start and rand start tof spectra.'
    plt.plot(t_axis_us, i_tof_e_start, 'b', label='electron start ions')
    plt.plot(t_axis_us, i_tof_random_start, 'g', label='random start ions')

    xlabel_wrapper('flight time (us)')
    ylabel_wrapper('number of ions per bin')
    legend_wrapper()
    tick_fontsize()

    plt.subplot(212)
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
        print 'Setup e start filter.'
    data.get_filter('e_start_events', epicea.ff_e_start_events)
    x_axis_mm = np.arange(-40, 40, 0.1)
    if verbose:
        print 'Get full image.'
    i_image = data.get_i_image(x_axis_mm)
    if verbose:
        print 'Get e start image.'
    i_image_e_start = data.get_i_image(
        x_axis_mm, ions_filter=data.get_filter('e_start_ions',
                                               epicea.ff_events_filtered_ions,
                                               {'events_filter_name':
                                                   'e_start_events'}))
    if verbose:
        print 'Get rand start iage.'
    i_image_random_start = data.get_i_image(
        x_axis_mm, ions_filter=~data.get_filter('e_start_ions'))

    figure_wrapper('Ion image {}'.format(data_name))
    plt.subplot(231)
    imshow_wrapper(i_image, x_axis_mm)
    title_wrapper('all ions')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    colorbar_wrapper()
    tick_fontsize()

    plt.subplot(232)
    imshow_wrapper(i_image_e_start, x_axis_mm)
    title_wrapper('electron start')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    colorbar_wrapper()
    tick_fontsize()

    plt.subplot(234)
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
    plt.subplot(438)
    plt.plot(x_axis_mm, x_projection, label='normal')
    plt.plot(x_axis_mm[::-1], x_projection, label='reverse')
    title_wrapper('x slice')
    xlabel_wrapper('Position (mm)')
    legend_wrapper()
    tick_fontsize()
    plt.subplot(4, 3, 11)
    plt.plot(x_axis_mm, y_projection, label='nomral')
    plt.plot(x_axis_mm[::-1], y_projection, label='reversed')
    title_wrapper('y slice')
    xlabel_wrapper('Position (mm)')
    legend_wrapper()
    tick_fontsize()

    r_axis_mm = np.linspace(0, 40, 201)[1::2]
    t_axis_rad = np.linspace(0, 2*np.pi, 513)[1::2]

    rt_image = data.get_i_rt_image(r_axis_mm=r_axis_mm,
                                   t_axis_rad=t_axis_rad)

    plt.subplot(233)
    imshow_wrapper(rt_image, r_axis_mm, t_axis_rad, {'aspect': 'auto'})
    tick_fontsize()
    title_wrapper('electron start polar')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Angle (rad)')

    plt.subplot(236)
    r_projection = rt_image.sum(axis=0)
    plt.plot(r_axis_mm, r_projection)
    title_wrapper('e start r projection')
    xlabel_wrapper('Radius (mm)')
    ylabel_wrapper('Number of counts')
    tick_fontsize()

    savefig_wrapper()
# %%


def plot_two_ion_corelations(data, data_name, verbose=False):

    if verbose:
        print 'In plot_two_ion_corelations.'
    # Set up some event filters
    data.get_filter('two_ion_events', epicea.ff_num_ion_events,
                    {'min_ions': 2, 'max_ions': 2},
                    verbose=verbose)
    data.get_filter('e_start_events', epicea.ff_e_start_events)
    data.get_filter('rand_start_events', epicea.ff_invert,
                    {'filter_name': 'e_start_events'},
                    verbose=verbose)

    data.get_filter('two_ion_e_start_events', epicea.ff_combine,
                    {'filter_name_list': ['two_ion_events', 'e_start_events']},
                    verbose=verbose)
    data.get_filter('two_ion_rand_start_events', epicea.ff_combine,
                    {'filter_name_list': ['two_ion_events',
                                          'rand_start_events']},
                    verbose=verbose)

    # And some ion filters based on the event filters
    double_ions = data.get_filter('two_ion_events_ions',
                                  epicea.ff_events_filtered_ions,
                                  {'events_filter_name': 'two_ion_events'},
                                  verbose=verbose)
    double_ions_e_start = data.get_filter('two_ion_e_start_events_ions',
                                          epicea.ff_events_filtered_ions,
                                          {'events_filter_name':
                                              'two_ion_e_start_events'},
                                          verbose=verbose)
    double_ions_rand_start = data.get_filter('two_ion_rand_start_events_ions',
                                             epicea.ff_events_filtered_ions,
                                             {'events_filter_name':
                                                 'two_ion_rand_start_events'},
                                             verbose=verbose)

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
    t = data.ions.pos_t[double_ions_e_start]
    # Fold the angles
    angle_diff = np.abs(np.diff(t)[::2])
    large_angle = angle_diff > np.pi
    angle_diff[large_angle] = 2*np.pi - angle_diff[large_angle]
    t_axis_rad = np.linspace(0, np.pi, 513)[1::2]
    rad_hist = epicea.center_histogram(angle_diff[np.isfinite(angle_diff)],
                                       t_axis_rad)
    bar_wrapper(t_axis_rad, rad_hist)
    sl = slice(t_axis_rad.searchsorted(ANGLE_CUT), None)
    bar_wrapper(t_axis_rad[sl], rad_hist[sl], 'r')
#    plt.hist(angle_diff[np.isfinite(angle_diff)],
#             bins=np.linspace(-0.1, np.pi, 256))
    xlabel_wrapper('Angle difference between tow ions (rad)')
    ylabel_wrapper('Number of ion pairs')
    title_wrapper('Two ion angles.')
    tick_fontsize()

    if verbose:
        print '{} events with two ions and electron start identified.'.format(
            data.get_filter('two_ion_e_start_events').sum())
        print '{} valid angle diffs found.'.format(
            np.isfinite(angle_diff).sum())

    # Radial information
    # figure_wrapper('radii {}'.format(data_name))
    plt.subplot(223)
    if verbose:
        print 'Get radii.'
    r = data.ions.pos_r[double_ions_e_start]
    r_pairs = r.reshape(-1, 2)
    r_pairs.sort(axis=1)
    r_frac = r_pairs[:, 1] / r_pairs[:, 0]
    valid = np.isfinite(r_frac)
    valid_and_cut = valid & (angle_diff > ANGLE_CUT)
    r_frac_all = r_frac[valid]
    r_frac_cut = r_frac[valid_and_cut]
    r_frac_ax = np.linspace(1., 3.5, 257)[1::2]
    r_frac_all_hist = epicea.center_histogram(r_frac_all, r_frac_ax)
    r_frac_cut_hist = epicea.center_histogram(r_frac_cut, r_frac_ax)
    bar_wrapper(r_frac_ax, r_frac_all_hist)
    bar_wrapper(r_frac_ax, r_frac_cut_hist, 'r')
    # plt.hist(r_frac, bins=np.linspace(0.9, 3., 256))
    xlabel_wrapper('Radial quotient for two ions r1/r2')
    ylabel_wrapper('Number of ion pairs')
    title_wrapper('Radius quotient')
    tick_fontsize()
    savefig_wrapper()

    figure_wrapper('tof-tof hist {}'.format(data_name))
    t_axis_us = np.linspace(3.3, 5.3, 512)
    t_axis = t_axis_us * 1e6

    names = ['e start', 'all', 'random start']
    for i, ions_filter in enumerate([double_ions_e_start,
                                     double_ions,
                                     double_ions_rand_start]):
        plt.subplot(2, 2, i+1)
        i_tof = data.ions.tof_falling_edge[ions_filter]
        i_tof = i_tof.reshape(-1, 2)
        i_tof.sort(axis=1)
        i_tof = i_tof[np.isfinite(i_tof).sum(1) == 2, :]

        tof_tof_hist = epicea.center_histogram_2d(i_tof[:, 0], i_tof[:, 1],
                                                  t_axis)
        tof_tof_sym_hist = tof_tof_hist + tof_tof_hist.T
        imshow_wrapper(tof_tof_sym_hist, t_axis_us)
        title_wrapper(names[i])
        ylabel_wrapper('Time of flight (us)')
        xlabel_wrapper('Time of flight (us)')
        tick_fontsize()

        if i == 0:
            tof_tof_hist_e_start = tof_tof_sym_hist.copy()
        if i == 2:
            tof_tof_hist_random = tof_tof_sym_hist.copy()

    sl = slice(t_axis.searchsorted(5.5e6))
    factor = (tof_tof_hist_e_start[sl, :].sum() /
              tof_tof_hist_random[sl, :].sum())
    pure = tof_tof_hist_e_start - tof_tof_hist_random * factor

    plt.subplot(224)
    imshow_wrapper(pure, t_axis_us)
    tick_fontsize()
    savefig_wrapper()
# %%


def plot_e_spec(data, data_name, verbose=False):
    """First view of the electron spectra."""

    if verbose:
        print 'Plotting electron spectra for {}.'.format(data_name)
    figure_wrapper('Electron data {}'.format(data_name))

    valid_pos = data.get_filter('has_position_electrons',
                                epicea.ff_has_position_particles,
                                {'particles': 'electrons',
                                 'verbose': verbose})
    if verbose:
        print 'Valid positions for {} electrons.'.format(valid_pos.sum())

    x = data.electrons.pos_x[valid_pos]
    y = data.electrons.pos_y[valid_pos]
    r = data.electrons.pos_r[valid_pos]
    t = data.electrons.pos_t[valid_pos]
    x_axis_mm = np.linspace(-23, 23, 512)
    r_axis_mm = np.linspace(0, 23, 513)[1::2]
    t_axis_rad = np.linspace(0, 2*np.pi, 513)[1::2]
    xy_center_slice = slice(x_axis_mm.searchsorted(-3),
                            x_axis_mm.searchsorted(3, side='right'))

    e_image_xy = epicea.center_histogram_2d(x, y, x_axis_mm)
    e_x_slice = e_image_xy[xy_center_slice, :].sum(axis=0)
    e_y_slice = e_image_xy[:, xy_center_slice].sum(axis=1)
    e_image_rt = epicea.center_histogram_2d(r, t, r_axis_mm, t_axis_rad)
    e_radial_dist = e_image_rt.sum(axis=0)

    plt.subplot(221)
    imshow_wrapper(e_image_xy, x_axis_mm)
    title_wrapper('2D image')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Position (mm)')
    tick_fontsize()

    plt.subplot(425)
    plt.plot(x_axis_mm, e_x_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_x_slice, label='flipped')
    title_wrapper('x slice')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of ions')
    tick_fontsize()
    legend_wrapper()

    plt.subplot(427)
    plt.plot(x_axis_mm, e_y_slice, label='normal')
    plt.plot(x_axis_mm[::-1], e_y_slice, label='flipped')
    title_wrapper('y slice')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of ions')
    tick_fontsize()
    legend_wrapper()

    plt.subplot(222)
    imshow_wrapper(e_image_rt, r_axis_mm, t_axis_rad, {'aspect': 'auto'})
    title_wrapper('2D image')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Angle (rad)')
    tick_fontsize()

    plt.subplot(224)
    plt.plot(r_axis_mm, e_radial_dist)
    title_wrapper('Radial distribution')
    xlabel_wrapper('Position (mm)')
    ylabel_wrapper('Number of electrons')
    tick_fontsize()

    savefig_wrapper()
# %%

if __name__ == '__main__':
    # %%

    dataset_info = {}
    dataset_info['430_high'] = {
        'data_path': '../data/ExportedData/N2O_0029_KE373_hv430eV/',
        'h5_path': 'h5_data/N20_430_high.h5'
        }
    dataset_info['412_high'] = {
        'data_path': '../data/ExportedData/N2O_0031_KE373_hv412eV/',
        'h5_path': 'h5_data/N20_412_high.h5'
        }
    dataset_info['430_mid'] = {
        'data_path': '../data/ExportedData/N2O_366PE_430eV_0014/',
        'h5_path': 'h5_data/N20_430_mid.h5'
        }
    dataset_info['412_mid'] = {
        'data_path': '../data/ExportedData/N2O_366PE_4119eV_combined/',
        'h5_path': 'h5_data/N20_412_mid.h5'
        }
    dataset_info['430_low'] = {
        'data_path': '../data/ExportedData/N2O_KE357_hv430p9_0047/',
        'h5_path': 'h5_data/N20_430_low.h5'
        }
    dataset_info['412_low'] = {
        'data_path': '../data/ExportedData/N2O_KE357_hv412p9_0049/',
        'h5_path': 'h5_data/N20_412_low.h5'
        }
    dataset_info['560'] = {
        'data_path': '../data/ExportedData/N2O_500PE_560eV_0017/',
        'h5_path': 'h5_data/N20_560.h5'
        }

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
                data[data_name].ions.correct_center(ION_VMI_X_OFFSET,
                                                    ION_VMI_Y_OFFSET)
                data[data_name].electrons.correct_center(ELECTRON_X_OFFSET,
                                                         ELECTRON_Y_OFFSET)
    # %%

    for data_name, dataset in data.iteritems():
        plot_ion_tof(dataset, data_name, verbose=verbose)
    # %%

    for data_name, dataset in data.iteritems():
        plot_ion_image(dataset, data_name, verbose=verbose)
    # %%

    for data_name, dataset in data.iteritems():
        plot_two_ion_corelations(dataset, data_name, verbose=verbose)
    # %%

    for data_name, dataset in data.iteritems():
        plot_e_spec(dataset, data_name, verbose=verbose)
