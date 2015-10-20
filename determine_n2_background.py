# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:49:40 2015

@author: antlin
"""

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

import electron_calibration_data
import plt_func

#import global_parameters as glob
from global_parameters import (r_axis_mm, th_axis_rad)
import make_electron_calibration


_change_time = 144180272


def _file_name(setting):
    return 'h5_data/center_{}_bg.h5'.format(setting)


def make_new_bg(setting, *, plot=False, verbose=False):
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=False)
    except electron_calibration_data.CenterEnergyError as e:
        print(e)
        raise

    calib_data_list.sort()

    if plot:
        proj_fig = plt_func.figure_wrapper('proj')
        proj_ax = proj_fig.add_subplot(111)
        dev_fig = plt_func.figure_wrapper('deviations')
        dev_ax = dev_fig.add_subplot(111)
        number_fig = plt_func.figure_wrapper('number')
        number_ax = number_fig.add_subplot(111)

    proj_list = []

    for d in calib_data_list:
        straight_image, s_i_time_stamp, fig = \
            make_electron_calibration.get_straight_image(
                d, r_axis_mm, th_axis_rad, 0, 'N2',
                verbose=True, plot=True)
#        straight_image, s_i_time_stamp = d.load_derived_data(
#            'e_rth_image_straight', 'no_filter',
#            match_data_dict={'r_axis_mm': r_axis_mm,
#                             'th_axis_rad': th_axis_rad})

#        no_filter = d._derived_data['e_rth_image_straight/no_filter']
#        time_stamp = 0
#        for dset_tmp in no_filter.values():
#            if (('time_stamp' in dset_tmp.attrs) and
#                    (dset_tmp.attrs['time_stamp'] > time_stamp)):
#                time_stamp = dset_tmp.attrs['time_stamp']
#                dset = dset_tmp

#        proj = dset['data'].value.sum(axis=0) / d.data_scaling
        proj = straight_image.sum(axis=0) /d.data_scaling
#        r_axis = dset['r_axis_mm'].value
        proj_list.append(proj)
        if plot:
            for ax in [proj_ax, dev_ax]:
                plt.sca(ax)
                plt.plot(r_axis_mm, proj, label=d.name())

    if plot:
        proj_array_0 = np.array(proj_list)
        for remove_i in range(7):
            m = np.nanmean(proj_array_0, axis=0)
            d = np.nanstd(proj_array_0, axis=0)
            n = np.isfinite(proj_array_0).sum(axis=0)
            dev_ax.errorbar(r_axis_mm, m, d, label=str(remove_i),
                            fmt='.', capsize=0)
            number_ax.plot(r_axis_mm, n, label=str(remove_i))
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
        plt.sca(proj_ax)
        plt.plot(r_axis_mm, proj_mean, linewidth=2, label='bg')
        plt_func.legend_wrapper()

    with h5py.File(_file_name(setting)) as h5:
        dset_name = 'background_{}'.format(len(proj_mean))
        try:
            dset = h5.require_dataset(dset_name, proj_mean.shape,
                                       float, exact=True)

            dset[:] = proj_mean
        except TypeError:
            del h5[dset_name]
            dset = h5.create_dataset(dset_name, data=proj_mean)
        dset.attrs['time_stamp'] = time.time()

    return proj_list

def check_bg_valid(setting, length, verbose=False):
    try:
        calib_data_list = electron_calibration_data.get_data_in_list(
            setting, verbose=False)
    except electron_calibration_data.CenterEnergyError as e:
        print(e)
        return

    if isinstance(length, np.ndarray):
        length = len(length)

    try:
        with h5py.File(_file_name(setting), 'r') as h5:
            bg_time = h5['background_{}'.format(length)].attrs['time_stamp']
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


def load_background(setting, length):
    if isinstance(length, np.ndarray):
        length = len(length)
    with h5py.File(_file_name(setting), 'r') as h5:
        return h5['background_{}'.format(length)].value

if __name__ == '__main__':
# 357 and 500 where done with N2
    setting = 357
#    setting = 500
# 366 and 373 used Kr
#    setting = 373
#    setting = 366

    verbose = True
    plot = True

    if check_bg_valid(setting, r_axis_mm, verbose=verbose):
        if verbose:
            print('bg is valid.')
    elif verbose:
        print('bg is NOT valid.')

    proj_list = make_new_bg(setting, plot=plot, verbose=verbose)
