# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:49:40 2015

@author: antlin
"""

import numpy as np
import h5py
import time

import electron_calibration_data
import plt_func


_change_time = 1439204598


def _file_name(setting):
    return 'h5_data/center_{}_bg.h5'.format(setting)


def make_new_bg(setting, plot=False, verbose=False):
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=False)
    except electron_calibration_data.CenterEnergyError as e:
        print e
        raise

#    d = calib_data_list[0]
#    for dset_name, dset in d._derived_data[
#            'e_rth_image_straight/no_filter'].items():
#        print dset_name, dset.keys(), dset['data'].shape

    if plot:
        proj_fig = plt_func.figure_wrapper('proj')
        proj_ax = proj_fig.add_subplot(111)
        dev_fig = plt_func.figure_wrapper('deviations')
        dev_ax = dev_fig.add_subplot(111)
        number_fig = plt_func.figure_wrapper('number')
        number_ax = number_fig.add_subplot(111)

    proj_list = []

    for d in calib_data_list:
        no_filter = d._derived_data['e_rth_image_straight/no_filter']
        time_stamp = 0
        for dset_tmp in no_filter.values():
            if (('time_stamp' in dset_tmp.attrs) and
                    (dset_tmp.attrs['time_stamp'] > time_stamp)):
                time_stamp = dset_tmp.attrs['time_stamp']
                dset = dset_tmp

        proj = dset['data'].value.sum(axis=0) / d.data_scaling
        r_axis = dset['r_axis_mm'].value
        proj_list.append(proj)
        if plot:
            for ax in [proj_ax, dev_ax]:
                ax.plot(r_axis, proj, label=d.name())

    if plot:
        proj_array_0 = np.array(proj_list)
        for remove_i in range(7):
            m = np.nanmean(proj_array_0, axis=0)
            d = np.nanstd(proj_array_0, axis=0)
            n = np.isfinite(proj_array_0).sum(axis=0)
            dev_ax.errorbar(r_axis, m, d, label=str(remove_i),
                            fmt='.', capsize=0)
            number_ax.plot(r_axis, n, label=str(remove_i))
            i_r_list = np.where((d*4 > m) & (n > 3))[0]
            if len(i_r_list) == 0:
                break
            for i_r in i_r_list:
                maxdev = np.nanargmax(np.abs(proj_array_0[:, i_r] - m[i_r]))
                proj_array_0[maxdev, i_r] = np.nan
        plt_func.legend_wrapper(ax=number_ax)
        plt_func.legend_wrapper(ax=dev_ax)

#    proj_array = np.array(proj_list)
#    for i_r in range(len(r_axis)):
#        for i in range(2):
#            proj_array[np.nanargmax(proj_array[:, i_r]), i_r] = np.nan

    proj_mean = np.nanmean(proj_array_0, axis=0)

    if plot:
        proj_ax.plot(r_axis, proj_mean, linewidth=2, label='bg')
        plt_func.legend_wrapper()

    with h5py.File(_file_name(setting), 'w') as h5:
        dset = h5.create_dataset('background', data=proj_mean)
        dset.attrs['time_stamp'] = time.time()


def check_bg_valid(setting, verbose=False):
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=False)
    except electron_calibration_data.CenterEnergyError as e:
        print e
        return

    try:
        with h5py.File(_file_name(setting), 'r') as h5:
            bg_time = h5['background'].attrs['time_stamp']
    except:
        bg_time = 0

    if bg_time < _change_time:
        return False

    for d in calib_data_list:
        no_filter = d._derived_data['e_rth_image_straight/no_filter']
        for dset in no_filter.values():
            if (('time_stamp' in dset.attrs) and
                    (dset.attrs['time_stamp'] > bg_time)):
                return False

    return True


def load_background(setting):
    with h5py.File(_file_name(setting), 'r') as h5:
        return h5['background'].value

if __name__ == '__main__':
    setting = 357
    setting = 500
    setting = 373

    verbose = True
    plot = True

    if check_bg_valid(setting, verbose=verbose):
        if verbose:
            print 'bg is valid.'
    elif verbose:
        print 'bg is NOT valid.'

    make_new_bg(setting, plot=plot, verbose=verbose)
