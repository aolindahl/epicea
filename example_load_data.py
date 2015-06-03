# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:37:48 2015

@author: Anton O. Lindahl
"""
import numpy as np

import plt_func
if 'epicea' not in locals():
    import epicea


data_list = epicea.DataSetList()

data_list.add_dataset(
    'My data set',
    'h5_data/ion_img_0024.h5',
    '../data/ExportedData/Ion_image_calibrations/N2_ion_img_0024/',
    490,
    verbose=True)

# Do once to chenge the data in the hdf5 file
for dataset in data_list:
    dataset.ions.recalculate_polar_coordinates()

x_axis_mm = np.linspace(-40, 40, 2**10)
r_axis_mm = np.linspace(0, 40, 2**10)[1::2]
th_axis_rad = np.linspace(0, 2*np.pi, 2**10+1)[1::2]

for dataset in data_list:
    xy_img = dataset.get_i_xy_image(x_axis_mm)
    rth_img = dataset.get_i_rth_image(r_axis_mm, th_axis_rad)
    r_dist_th_sum = rth_img.sum(axis=0)
    r_dist = r_dist_th_sum / r_axis_mm
    fig = plt_func.figure_wrapper('Ion image of "{}'.format(dataset.name()))

    ax1 = fig.add_subplot(221)
    plt_func.imshow_wrapper(xy_img, x_axis_mm, axes=ax1)

    ax2 = fig.add_subplot(222)
    plt_func.imshow_wrapper(rth_img, r_axis_mm, th_axis_rad,
                            axes=ax2,
                            kw_args={'aspect': 'auto'})

    ax4 = fig.add_subplot(224)
    ax4.plot(r_axis_mm, r_dist_th_sum)

    ax3 = fig.add_subplot(223)
    ax3.plot(r_axis_mm, r_dist)
