# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:34:36 2015

@author: antlin
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit

import epicea
import electron_calibration_data
import plt_func

line_model = epicea.electron_calibration_helper.n_line_fit_model
line = 'voigt'

data_list = electron_calibration_data.get_data_in_list('357', True)

r_axis_mm = np.linspace(0, 25, 2**9+1)[1::2]
th_axis_rad = np.linspace(0, 2*np.pi, 2**9+1)[1::2]
th_limits = epicea.limits_from_centers(th_axis_rad)

#    data = data_list[0]
for data in data_list:
    r_th_img = data.get_e_rth_image(r_axis_mm, th_axis_rad)[0]
    r_proj = r_th_img.sum(axis=0)

    proj_params = epicea.electron_calibration_helper.start_params(
        r_axis_mm, r_proj, n_lines=2)
    lmfit.minimize(line_model,
                   proj_params,
                   args=(r_axis_mm, r_proj),
                   kws={'line_type': line})

    r_th_fig = plt_func.figure_wrapper('theta - r ' + data.name())
    ax_origaninl = plt.subplot(221)
    plt_func.imshow_wrapper(r_th_img,
                            r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'})
    plt_func.colorbar_wrapper()
    ax_origaninl.autoscale(False)

    plt.subplot(223)
    plt.plot(r_axis_mm, r_proj)
    plt.plot(r_axis_mm, line_model(proj_params, r_axis_mm,
                                   line_type=line), '--')

    centers = (r_th_img * r_axis_mm).sum(axis=1) / r_th_img.sum(axis=1)
#    radial_factors = centers.mean()/centers

    # Find the center of the first line
    low_radius_centers = np.empty_like(centers)
    for i_th in range(len(th_axis_rad)):
        y = r_th_img[i_th, :]
        i_min = r_axis_mm.searchsorted(centers[i_th])
        while y[i_min] > y[i_min - 1]:
            i_min -= 1
        while y[i_min] > y[i_min+1]:
            i_min += 1
        I_low_radius = (((centers[i_th] - 3) <= r_axis_mm) &
                        (r_axis_mm <= centers[i_th]))
        low_radius_centers[i_th] = ((r_th_img[i_th, I_low_radius] *
                                     r_axis_mm[I_low_radius]).sum() /
                                    r_th_img[i_th, I_low_radius].sum())

    radial_factors = low_radius_centers.mean() / low_radius_centers

    ax_origaninl.plot(centers, th_axis_rad, 'm')
    ax_origaninl.plot(low_radius_centers, th_axis_rad, 'c')

    plt_func.figure_wrapper('centers ' + data.name())
    plt.subplot(121)
    plt.plot(centers, th_axis_rad, label='full center')
    plt.plot(low_radius_centers, th_axis_rad, label='first center')
    plt.title('center position')
    plt_func.legend_wrapper()
    plt.subplot(122)
    plt.plot(radial_factors, th_axis_rad)
    plt.title('r factors')

    r = data.electrons.pos_r.value
    th = data.electrons.pos_t.value

    for i in range(len(th_axis_rad)):
        selection = (th_limits[i] < th) & (th < th_limits[i+1])
        r[selection] *= radial_factors[i]

    r_th_img_corrected = epicea.center_histogram_2d(r, th, r_axis_mm,
                                                    th_axis_rad)

    r_proj_corrected = r_th_img_corrected.sum(axis=0)
    proj_corrected_params = epicea.electron_calibration_helper.start_params(
        r_axis_mm, r_proj_corrected, n_lines=2)
    lmfit.minimize(line_model,
                   proj_corrected_params,
                   args=(r_axis_mm, r_proj_corrected),
                   kws={'line_type': line})

    ax = r_th_fig.add_subplot(222)
    plt.sca(ax)
    plt_func.imshow_wrapper(r_th_img_corrected,
                            r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'})
    axis = plt.axis()
    plt.plot(centers * radial_factors * np.ones_like(th_axis_rad),
             th_axis_rad, 'm')
    plt.plot(low_radius_centers.mean() * np.ones_like(th_axis_rad),
             th_axis_rad, 'm')
    plt_func.colorbar_wrapper()

    plt.sca(r_th_fig.add_subplot(224))
    plt.plot(r_axis_mm, r_proj_corrected)
    plt.plot(r_axis_mm, line_model(proj_corrected_params, r_axis_mm,
                                   line_type=line), '--')

    plt_func.figure_wrapper('waterfall ' + data.name())
    for i in range(len(th_axis_rad)):
        plt.plot(r_axis_mm, r_th_img[i, :] + i * 20)

    r_th_fig.tight_layout()
