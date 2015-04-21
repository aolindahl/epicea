# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""
import numpy as np
import lmfit


def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x-center)**2 / (2 * width**2))


def start_params(x=None, y=None, params_in=None):
    params = lmfit.Parameters()

    params.add('amplitude_1', 20, min=0)
    params.add('center_1', value=15)
    params.add('width_1', value=0.4, min=0)
    params.add('amp_ratio', value=1, min=0)
    params.add('amplitude_2', expr='amplitude_1 * amp_ratio')
    params.add('center_diff', value=1, min=0.5)
    params.add('center_2', expr='center_1 + center_diff')
    params.add('width_2', value=0.4, min=0)

    if (x is not None) and (y is not None):
        max_idx = np.argmax(y)
        y_max = y[max_idx]
        print 'y_max =', y_max
        params['amplitude_1'].value = y_max
        params['center_1'].value = x[max_idx]
        for i in range(max_idx):
            print 'y[i] =', y[i], 'y_max =', y_max
            if y[i] >= float(y_max) / 20:
                params['center_1'].min = x[i]
                print i
                break
        for i in range(len(x)-1, max_idx-1, -1):
            if y[i] > float(y_max) / 20:
                print 'x[i] =', x[i],
                print 'params["center_1"].min =', params['center_1'].min
                params['center_diff'].max = x[i] - params['center_1'].min
                params['center_diff'].value = params['center_diff'].max / 2
                break

    if isinstance(params_in, lmfit.parameter.Parameters):
        for k in params_in:
            if k in params:
                params[k].value = params_in[k].value

    if isinstance(params_in, list):
        if np.all([isinstance(par, lmfit.parameter.Parameters) for
                  par in params_in]):
            for k in params:
                params[k].value = np.average(
                    [par[k].value for par in params_in if par[k].stderr != 0],
                    weights=[1./par[k].stderr for par in params_in
                             if par[k].stderr != 0])

    return params


def double_line_fit_model(params, x, data=None, eps_data=None):
    amplitude_1 = params['amplitude_1'].value
    center_1 = params['center_1'].value
    width_1 = params['width_1'].value
    amplitude_2 = params['amplitude_2'].value
    center_2 = params['center_2'].value
    width_2 = params['width_2'].value

    model = gaussian(x, amplitude_1, center_1, width_1) * x / center_1
    model += gaussian(x, amplitude_2, center_2, width_2) * x / center_2

    if data is None:
        return model
    if eps_data is None:
        return model - data
    return (model - data) / eps_data


def poly_line(params, x, y=None, err=None):
    mod = np.zeros_like(x)
    for i in range(len(params)):
        mod += params['a_{}'.format(i)] * x**i
    if y is None:
        return mod
    if err is None:
        return mod - y
    return (mod - y) / err


def line_start_params(a_list):
    if not isinstance(a_list, list):
        raise TypeError('Function linse_start_params expected a list' +
                        ' as input parameter, got {}.'.format(type(a_list)))

    params = lmfit.Parameters()
    for i, a in enumerate(a_list):
        params.add('a_{}'.format(i), value=a)

    return params


def find_kr_lines(rth_image, r_axis_mm, th_axis_rad,
                  return_line_params_list=False):
    # Fit to r projection to be used as initial parameters
    r_projection = rth_image.mean(axis=0)
    params_r_proj = start_params(x=r_axis_mm, y=r_projection)
    lmfit.minimize(double_line_fit_model, params_r_proj,
                   args=(r_axis_mm, r_projection))

    # Fit for each line based on the r projection fit
    line_params_list = []
    line_results_list = []
    for i in range(len(th_axis_rad)):
        line_params_list.append(start_params(
            x=r_axis_mm, y=rth_image[i, :]))
        line_results_list.append(
            lmfit.minimize(double_line_fit_model,
                           line_params_list[i],
                           args=(r_axis_mm, rth_image[i, :])))

#     Go trough once more
    num_to_average = len(th_axis_rad)/20
    if num_to_average == 0:
        num_to_average = 1
    for i in range(len(th_axis_rad)):
        selected_par_list = line_params_list[np.maximum(i-num_to_average, 0):
                                             i+num_to_average+1]
        if i < num_to_average:
            selected_par_list.extend(line_params_list[-num_to_average+i:])
        if num_to_average < len(line_params_list) - i:
            selected_par_list.extend(
                line_params_list[:num_to_average -
                                 len(line_params_list) + i + 1])

        line_params_list[i] = start_params(x=r_axis_mm, y=rth_image[i, :],
                                           params_in=selected_par_list)
        line_results_list[i] = lmfit.minimize(
            double_line_fit_model,
            line_params_list[i],
            args=(r_axis_mm, rth_image[i, :]))

    # Get the results in a nicer format
    r_1 = np.empty_like(line_params_list, dtype=float)
    r_2 = np.empty_like(r_1)
    w_1 = np.empty_like(r_1)
    w_2 = np.empty_like(r_1)
    red_chi2 = np.empty_like(r_1)
    for i in range(len(line_params_list)):
        r_1[i] = line_params_list[i]['center_1'].value
        w_1[i] = (line_params_list[i]['center_1'].stderr)
#                  line_results_list[i].redchi)
#        w_1[i] = line_params['width_1'].value
        r_2[i] = line_params_list[i]['center_2'].value
        w_2[i] = (line_params_list[i]['center_2'].stderr)
#                  line_results_list[i].redchi)
#        w_2[i] = line_params['width_2'].value
        red_chi2[i] = line_results_list[i].redchi

    w_1[w_1 <= 0] = np.inf
    w_2[w_2 <= 0] = np.inf
    if return_line_params_list:
        return r_1, w_1, r_2, w_2, red_chi2, line_params_list
    return r_1, w_1, r_2, w_2, red_chi2
