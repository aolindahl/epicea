# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:34:36 2015

@author: antlin
"""

import numpy as np
import matplotlib.pyplot as plt

import epicea
import electron_calibration_data
import plt_func

data_list = electron_calibration_data.get_data_in_list('357', True)

r_axis_mm = np.linspace(0, 25, 2**8+1)[1::2]
th_axis_rad = np.linspace(0, 2*np.pi, 2**8+1)[1::2]
th_limits = epicea.limits_from_centers(th_axis_rad)

for data in [data_list[0]]:
#for data in data_list:
    r_th_img = data.get_e_rth_image(r_axis_mm, th_axis_rad)[0]
    r_proj = r_th_img.sum(axis=0)

    th_sums = r_th_img.sum(axis=1)
    max_line_number = np.argmax(th_sums)

    norm_img = (r_th_img.T/th_sums).T

    fig_img = plt_func.figure_wrapper('normalized image')
    ax_img = fig_img.add_subplot(211)
    ax_img_straight = fig_img.add_subplot(212, sharex=ax_img, sharey=ax_img)
    plt_func.imshow_wrapper(norm_img, r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'}, ax=ax_img)

#    max_line = norm_img[max_line_number, :]
    max_line = norm_img.mean(axis=0)

    corr_img = np.array([np.correlate(line, max_line, mode='same')
                         for line in norm_img])
    fig_corr = plt_func.figure_wrapper('correlation')
    ax_corr = fig_corr.add_subplot(111)
    plt_func.imshow_wrapper(corr_img, r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'}, ax=ax_corr)

#    com_corr_img = (np.sum(corr_img * r_axis_mm, axis=1) /
#                    np.sum(corr_img, axis=1))
    com_corr_img = r_axis_mm[np.argmax(corr_img, axis=1)]
    com_corr_max_line = com_corr_img[max_line_number]
#    com_corr_max_line = com_corr_img.mean(axis=0)
    com_shift = com_corr_img - com_corr_max_line
    com_corrected = com_corr_max_line + com_shift

    plt.plot(com_corr_img, th_axis_rad, '.')
    r_factors = com_corrected.mean() / com_corrected


    r = data.electrons.pos_r.value
    th = data.electrons.pos_t.value

    for i_th in range(len(th_axis_rad)):
        mask_th = (th_limits[i_th] < th) & (th <= th_limits[i_th+1])
        r[mask_th] *= r_factors[i_th]

    straigt_img = epicea.center_histogram_2d(r, th, r_axis_mm, th_axis_rad)
    plt_func.imshow_wrapper(straigt_img, r_axis_mm, th_axis_rad,
                            ax=ax_img_straight, kw_args={'aspect': 'auto'})

    straight_proj = straigt_img.sum(axis=0)
    
    plt.figure('projections')
    plt.clf()
    plt.plot(r_axis_mm, r_proj/r_proj.sum())
    plt.plot(r_axis_mm, straight_proj/straight_proj.sum())
    random_slice = r_th_img[np.random.randint(len(th_axis_rad)), :]
    plt.plot(r_axis_mm, random_slice/random_slice.sum())
