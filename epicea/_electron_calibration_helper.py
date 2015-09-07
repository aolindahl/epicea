# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""
import numpy as np
import lmfit

from progress import update_progress

_2pi = 2 * np.pi
_gauss_fwhm_factor = 2 * np.sqrt(2 * np.log(2))


def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x-center)**2 / (2 * width**2))


def lorentzian(x, amplitude, center, width):
    return amplitude / _2pi * width / ((x - center)**2 + width**2 / 4)


def start_params(x=None, y=None, params_in=None, n_lines=2, verbose=False,
                 line_type=None):
    params = lmfit.Parameters()

    params.add('amplitude_1', 20, min=0)
    params.add('center_1', value=15)
    params.add('width_1', value=0.4, min=0)
    if line_type == 'voigt':
        pass
#        params.add('gamma_1', value=0.4e-3, min=0)
#        params['width_1'].value *= 1e-3
#        params.add('skew_1', value=0)
    if n_lines > 1:
        params.add('amp_ratio', value=1, min=0)
        params.add('amplitude_2', expr='amplitude_1 * amp_ratio')
        params.add('center_diff', value=1, min=0.5)
        params.add('center_2', expr='center_1 + center_diff')
        params.add('width_2', value=0.4, min=0)
        if line_type == 'voigt':
            pass
#            params.add('gamma_2', value=0.4e-3, min=0)
#            params['width_2'].value *= 1e-3
#            params.add('skew_2', value=0)
    for i_line in range(3, n_lines+1):
        params.add('amplitude_{}'.format(i_line), 20, min=0)
        params.add('center_{}'.format(i_line), value=11)
        params.add('width_{}'.format(i_line), value=0.5, min=0)
        if line_type == 'voigt':
            pass
#            params.add('gamma_{}'.format(i_line), value=0.5, min=0)
#            params['width_{}'.format(i_line)].value *= 1e-7
#            params.add('skew_{}'.format(i_line), value=0)
#    if n_lines > 2:
#        print 'Warning: Only one or two lines implemented in start_params().',
#        print 'Using two lines.'

    if (x is not None) and (y is not None):
        max_idx = np.argmax(y)
        y_max = y[max_idx]
        if verbose:
            print 'y_max =', y_max
        params['amplitude_1'].value = y_max
        params['center_1'].value = x[max_idx]
        if n_lines > 2:
            for i in range(max_idx):
                if verbose:
                    print 'y[i] =', y[i], 'y_max =', y_max
                if y[i] >= float(y_max) / 20:
                    params['center_1'].min = x[i]
                    if verbose:
                        print 'i =', i
                    break
            for i in range(len(x)-1, max_idx-1, -1):
                if y[i] > float(y_max) / 20:
                    if verbose:
                        print 'x[i] =', x[i],
                        print 'params["center_1"].min = {}'.format(
                            params['center_1'].min)
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


def n_line_fit_model(params, x, data=None, eps_data=None,
                     line_type='voigt'):
    i = 1
    model = np.zeros_like(x)
    while 1:
        try:
            amplitude = params['amplitude_{}'.format(i)].value
            center = params['center_{}'.format(i)].value
            width = params['width_{}'.format(i)].value
            gamma_str = 'gamma_{}'.format(i)
            skew_str = 'skew_{}'.format(i)
            if gamma_str in params:
                gamma = params[gamma_str].value
            else:
                gamma = None
            if skew_str in params:
                skew = params[skew_str].value
            else:
                skew = 0

            if line_type == 'gaussian':
                model += gaussian(x, amplitude, center, width)
            elif line_type == 'lorentzian':
                model += lorentzian(x, amplitude, center, width)
            elif line_type == 'voigt':
                model += lmfit.models.skewed_voigt(x, amplitude, center,
                                                   width, gamma, skew)
#                model += (lmfit.models.voigt(x, amplitude, center, width) *
#                          x / center)
            else:
                raise TypeError('No model named ', line_type, ' avaliable.')
        except:
            break
        i += 1
#    amplitude_1 = params['amplitude_1'].value
#    center_1 = params['center_1'].value
#    width_1 = params['width_1'].value
#    amplitude_2 = params['amplitude_2'].value
#    center_2 = params['center_2'].value
#    width_2 = params['width_2'].value

#    model = gaussian(x, amplitude_1, center_1, width_1) * x / center_1
#    model += gaussian(x, amplitude_2, center_2, width_2) * x / center_2

    if data is None:
        return model
    if eps_data is None:
        return model - data
    return (model - data) / eps_data


def n_voigt_with_bg_model(params, x, data=None, eps_data=None, bg=None):
    model = np.zeros_like(x)
    # The two line gaussian background
#    for i in range(2, 4):
#        model += gaussian(x,
#                          params['amplitude_{}'.format(i)].value,
#                          params['center_{}'.format(i)].value,
#                          params['width_{}'.format(i)].value)
    if 'bg_factor' in params:
        if bg is None:
            bg = np.ones_like(x)
        model = params['bg_factor'].value * bg
    else:
        model = np.zeros_like(x)

    # The voigt shaped photoline
    for i in range(2):
        try:
            model += lmfit.models.skewed_voigt(
                x,
                params['amplitude_{}'.format(i+1)].value,
                params['center_{}'.format(i+1)].value,
                params['width_{}'.format(i+1)].value,
                params['gamma_{}'.format(i+1)].value)
#                params['skew_{}'.format(i+1)].value)
        except:
            break

    if data is None:
        return model
    if eps_data is None:
        return model - data
    return (model - data) / eps_data


def n_voigt_with_bg_start_params(x=None, y=None, n_lines=2, bg=False):
    params = lmfit.Parameters()

    params.add('amplitude_1', 8e3, min=0)
    params.add('center_1', value=16.5)
    params.add('width_1', value=1e-3, min=0)
    params.add('gamma_1', value=0.4, min=0)
#    params.add('gamma_1', expr='width_1')
#    params.add('skew_1', value=0.002, min=0)

    if (x is not None) and (y is not None):
        params['center_1'].value = x[np.nanargmax(y)]

    if bg:
        params.add('bg_factor', value=1, min=0.0, max=1.5)

    if n_lines > 1:
#        params.add('amp_ratio', value=0.4, min=0.3, max=0.5)
#        params.add('amplitude_2', expr='amplitude_1 * amp_ratio')
        params.add('amplitude_2',
                   value=params['amplitude_1'].value * 0.7,
                   min=0)
    #    params.add('center_diff', value=5, min=0.3)
    #    params.add('center_2', expr='center_1 + center_diff')
        params.add('center_2',
                   value=params['center_1'].value * 1.09,
                   min=14, max=22)
        params.add('width_2', value=2e-1, min=0)
        params.add('gamma_2', value=0.6, min=0)
#        params.add('gamma_2', expr='width_2')
#        params.add('skew_2', value=0, min=-1, max=1)

#    params.add('amplitude_2', 2e3, min=1e3)
#    params.add('center_2', value=11.6, max=12, min=11.2)
#    params.add('width_2', value=1.1, min=0.9, max=1.3)
#
#    params.add('amplitude_3', 1e3, min=5e2)
#    params.add('center_3', value=19, min=17)
#    params.add('width_3', value=0.5, min=0, max=1)
#
#    params.add('amplitude_4', 5e2, min=2e2)
#    params.add('center_4', value=16, min=15, max=17)
#    params.add('width_4', value=0.5, min=0, max=1)

    return params


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


def find_lines_old(rth_image, r_axis_mm, th_axis_rad,
                   n_lines=2, return_line_params_list=False,
                   rth_image_straight=None):
    # Fit to r projection to be used as initial parameters
    r_projection = rth_image.mean(axis=0)
    params_r_proj = start_params(x=r_axis_mm, y=r_projection, n_lines=n_lines)
    lmfit.minimize(n_line_fit_model, params_r_proj,
                   args=(r_axis_mm, r_projection))

    # Fit for each line based on the r projection fit
    line_params_list = []
    line_results_list = []
    for i in range(len(th_axis_rad)):
        line_params_list.append(start_params(
            x=r_axis_mm, y=rth_image[i, :], n_lines=n_lines))
        line_results_list.append(
            lmfit.minimize(n_line_fit_model,
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

        line_params_list[i] = start_params(x=r_axis_mm,
                                           y=rth_image[i, :],
                                           params_in=selected_par_list,
                                           n_lines=n_lines)
        line_results_list[i] = lmfit.minimize(
            n_line_fit_model,
            line_params_list[i],
            args=(r_axis_mm, rth_image[i, :]))

    # Get the results in a nicer format
    r = np.empty((len(line_params_list), n_lines), dtype=float)
    w = np.empty_like(r)
    a = np.empty_like(r)
    red_chi2 = np.empty(len(r))

    for i in range(len(line_params_list)):
        for line in range(n_lines):
            r[i, line] = line_params_list[i]['center_{}'.format(line+1)].value
            w[i, line] = line_params_list[i]['center_{}'.format(line+1)].stderr
            a[i, line] = line_params_list[i][
                'amplitude_{}'.format(line+1)].value
        red_chi2[i] = line_results_list[i].redchi

#    w_1[w_1 <= 0] = np.inf
#    w_2[w_2 <= 0] = np.inf
    if return_line_params_list:
        return r, w, a, red_chi2, line_params_list
    return r, w, a, red_chi2


def find_lines(rth_image, r_axis_mm, th_axis_rad,
               n_lines_fit=2, n_lines_store=2,
               bg=None, gas='Kr', return_line_params_list=False,
               verbose=False):

    # Fit to r projection
    r_projection = rth_image.sum(axis=0)
    if gas == 'N2':
        kws = {'bg': bg}
        params_r_proj = n_voigt_with_bg_start_params(
            x=r_axis_mm, y=r_projection, n_lines=n_lines_fit,
            bg=True)
        lmfit.minimize(n_voigt_with_bg_model, params_r_proj,
                       args=(r_axis_mm, r_projection), kws=kws)
    elif gas == 'Kr':
        kws = {'line_type': 'voigt'}
        params_r_proj = start_params(
            x=r_axis_mm, y=r_projection, n_lines=n_lines_fit, **kws)
        lmfit.minimize(n_line_fit_model, params_r_proj,
                       args=(r_axis_mm, r_projection), kws=kws)

    # Fit for each line based on the r projection fit
    line_params_list = []
    line_results_list = []
    n_th = len(th_axis_rad)

    # Make space for the results in a nicer format
    r = np.empty((n_th, n_lines_store), dtype=float)
    w = np.empty_like(r)
    a = np.empty_like(r)
    red_chi2 = np.empty(n_th)

    for i_th in range(n_th):
        # Add a new set for parameters for each line
        if gas == 'N2':
            line_params_list.append(n_voigt_with_bg_start_params(
                x=r_axis_mm, y=rth_image[i_th, :], n_lines=n_lines_fit,
                bg=(bg is not None)))
        elif gas == 'Kr':
            line_params_list.append(start_params(
                x=r_axis_mm, y=r_projection, n_lines=n_lines_fit, **kws))

        current_params = line_params_list[-1]

        # map some stuff from the full projection to each line
        amp_scaling = rth_image[i_th, :].sum() / r_projection.sum()
        for k, v in current_params.iteritems():
            v.value = params_r_proj[k].value
            if k.startswith(('amp', 'bg')):
                v.value *= amp_scaling
#            if k.startswith('center'):
#                # Lock the centers
#                v.vary = False

        # Make the actuall fit
        if gas == 'N2':
            line_results_list.append(
                lmfit.minimize(n_voigt_with_bg_model,
                               current_params,
                               args=(r_axis_mm, rth_image[i_th, :]),
                               kws=kws))
        elif gas == 'Kr':
            line_results_list.append(
                lmfit.minimize(n_line_fit_model,
                               current_params,
                               args=(r_axis_mm, rth_image[i_th, :]),
                               kws=kws))

        for line in range(n_lines_store):
            r[i_th, line] = current_params['center_{}'.format(line+1)].value
            # For the width, we need to do some calculations
            # based on
            # https://en.wikipedia.org/wiki/Voigt_profile
            fg = (current_params['width_{}'.format(line+1)].value *
                  _gauss_fwhm_factor)
            try:
                fl = current_params['gamma_{}'.format(line+1)].value * 2
                w[i_th, line] = (0.5346 * fl +
                                 np.sqrt(0.2166 * fl**2 + fg**2)) / 2
            except:
                w[i_th, line] = fg / 2
            w[i_th, line] /= _gauss_fwhm_factor
#            w[i_th, line] = current_params['width_{}'.format(line+1)].value
#            w[i_th, line] = current_params['gamma_{}'.format(line+1)].value
#            w[i_th, line] = current_params[ 'center_{}'.format(line+1)].stderr
            a[i_th, line] = current_params[
                'amplitude_{}'.format(line+1)].value
#        a[i_th, :] = n_line_fit_model(current_params, r[i_th, :], **line_type)
        red_chi2[i_th] = line_results_list[i_th].redchi

        update_progress(i_th, n_th, verbose=verbose)

    if verbose:
        print ''
        lmfit.report_fit(line_results_list[len(line_results_list)/2])

#    w_1[w_1 <= 0] = np.inf
#    w_2[w_2 <= 0] = np.inf
    if return_line_params_list:
        return r, w, a, red_chi2, line_params_list
    return r, w, a, red_chi2
