# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""
import numpy as np
import lmfit

from . progress import update_progress

_2pi = 2 * np.pi
_gauss_fwhm_factor = 2 * np.sqrt(2 * np.log(2))


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2 * sigma**2))


def lorentzian(x, amplitude, center, sigma):
    return amplitude / _2pi * sigma / ((x - center)**2 + sigma**2 / 4)

def skewed_gauss_for_500_eV_start_params(x, y, eps=None, bg=False):
    x0 = x[np.nanargmax(y)]
    n_lines = 1
    if x0 < 15:
        n_lines += 1
#    if x0 < 11.5:
#        n_lines += 1

    
    params = lmfit.Parameters()

    if bg == True:
        params.add('bg_factor', 1, min=0, max=3, vary=True)

    params.add('amplitude_1', 7000, min=4000, max=15000)
    params.add('center_1', x0, min=x0*0.9, max=x0*1.1)
    params.add('sigma_1', 0.0001, min=0)
    params.add('gamma_1', 0.4, min=0)
    params.add('skew_1', 0.0, vary=True)

    if 1< n_lines:
        x2 = x0 * 1.6
        params.add('amplitude_2', 3000, min=0, max=4000)
        params.add('center_2', x2, min=x2*0.9, max=x2*1.1)
        params.add('sigma_2', 0.1, min=0, max=1)
    #    params.add('gamma_2', 0.1, min=0)
        params.add('skew_2', 0.0, vary=False)        

    return params

def skewed_gauss_for_500_eV(params, x, y=None, eps=None, bg=None):

    if bg is not None:
       mod = bg.copy()
    else:
        mod = np.zeros_like(x)

    if 'bg_factor' in params:
        mod *= params['bg_factor'].value

    i = 1
    while 'center_{}'.format(i) in params:
        amplitude = params['amplitude_{}'.format(i)].value
        center = params['center_{}'.format(i)].value
        sigma = params['sigma_{}'.format(i)].value
        if 'gamma_{}'.format(i) in params:
            gamma = params['gamma_{}'.format(i)].value
        else:
            gamma = None
        skew = params['skew_{}'.format(i)].value

        mod += lmfit.models.skewed_voigt(x, amplitude, center,
                                         sigma, gamma, skew)

#        mod += lmfit.models.skewed_gaussian(x, amplitude, center,
#                                            sigma, skew)

        i += 1

    if y is None:
        return mod
    if eps is None:
        return mod-y
    return (mod-y)/eps


def start_params(x=None, y=None, params_in=None, n_lines=2, verbose=False,
                 line_type=None):
    params = lmfit.Parameters()

    params.add('amplitude_1', 20, min=0)
    params.add('center_1', value=15)
    params.add('sigma_1', value=0.4, min=0)
    if line_type == 'voigt':
        pass
#        params.add('gamma_1', value=0.4e-3, min=0)
#        params['sigma_1'].value *= 1e-3
#        params.add('skew_1', value=0)
    if n_lines > 1:
        params.add('amp_ratio', value=1, min=0)
        params.add('amplitude_2', expr='amplitude_1 * amp_ratio')
        params.add('center_diff', value=1, min=0.5)
        params.add('center_2', expr='center_1 + center_diff')
        params.add('sigma_2', value=0.4, min=0)
        if line_type == 'voigt':
            pass
#            params.add('gamma_2', value=0.4e-3, min=0)
#            params['sigma_2'].value *= 1e-3
#            params.add('skew_2', value=0)
    for i_line in range(3, n_lines+1):
        params.add('amplitude_{}'.format(i_line), 20, min=0)
        params.add('center_{}'.format(i_line), value=11)
        params.add('sigma_{}'.format(i_line), value=0.5, min=0)
        if line_type == 'voigt':
            pass
#            params.add('gamma_{}'.format(i_line), value=0.5, min=0)
#            params['sigma_{}'.format(i_line)].value *= 1e-7
#            params.add('skew_{}'.format(i_line), value=0)
#    if n_lines > 2:
#        print('Warning: Only one or two lines implemented in start_params().',
#        print('Using two lines.'

    if (x is not None) and (y is not None):
        max_idx = np.argmax(y)
        y_max = y[max_idx]
        if verbose:
            print('y_max =', y_max)
        params['amplitude_1'].value = y_max
        params['center_1'].value = x[max_idx]
        if n_lines > 2:
            for i in range(max_idx):
                if verbose:
                    print('y[i] =', y[i], 'y_max =', y_max)
                if y[i] >= float(y_max) / 20:
                    params['center_1'].min = x[i]
                    if verbose:
                        print('i =', i)
                    break
            for i in range(len(x)-1, max_idx-1, -1):
                if y[i] > float(y_max) / 20:
                    if verbose:
                        print('x[i] =', x[i],
                              'params["center_1"].min = {}'.format(
                              params['center_1'].min))
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
            sigma = params['sigma_{}'.format(i)].value
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
                model += gaussian(x, amplitude, center, sigma)
            elif line_type == 'lorentzian':
                model += lorentzian(x, amplitude, center, sigma)
            elif line_type == 'voigt':
                model += lmfit.models.skewed_voigt(x, amplitude, center,
                                                   sigma, gamma, skew)
#                model += (lmfit.models.voigt(x, amplitude, center, sigma) *
#                          x / center)
            else:
                raise TypeError('No model named ', line_type, ' avaliable.')
        except:
            break
        i += 1
#    amplitude_1 = params['amplitude_1'].value
#    center_1 = params['center_1'].value
#    sigma_1 = params['sigma_1'].value
#    amplitude_2 = params['amplitude_2'].value
#    center_2 = params['center_2'].value
#    sigma_2 = params['sigma_2'].value

#    model = gaussian(x, amplitude_1, center_1, sigma_1) * x / center_1
#    model += gaussian(x, amplitude_2, center_2, sigma_2) * x / center_2

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
#                          params['sigma_{}'.format(i)].value)
    if 'bg_factor' in params:
        if bg is None:
            bg = np.ones_like(x)
        model = params['bg_factor'].value * bg
    else:
        model = np.zeros_like(x)

    # The voigt shaped photoline
    for i in range(3):
        try:
            model += lmfit.models.skewed_voigt(
                x,
                params['amplitude_{}'.format(i+1)].value,
                params['center_{}'.format(i+1)].value,
                params['sigma_{}'.format(i+1)].value,
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
    params.add('sigma_1', value=1e-3, min=0, max=1)
    params.add('gamma_1', value=0.4, min=0, max=1)
#    params.add('gamma_1', expr='sigma_1')
#    params.add('skew_1', value=0.002, min=0)

    if (x is not None) and (y is not None):
        params['center_1'].value = x[np.nanargmax(y)]

    if bg:
        params.add('bg_factor', value=1, min=0.0, max=1.5)

    if 1 < n_lines:
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
        params.add('sigma_2', value=1e-4, min=0, max=1)
        params.add('gamma_2', value=0.4, min=0, max=1)
#        params.add('gamma_2', expr='sigma_2')
#        params.add('skew_2', value=0, min=-1, max=1)

    if 2 < n_lines:
        params.add('amplitude_3', 2e3, min=0)
        params.add('center_3', value=11.6, max=12, min=11.2)
        params.add('sigma_3', value=1e-4, min=0, max=1)
        params.add('gamma_3', value=0.4, min=0, max=1)
#
#    params.add('amplitude_3', 1e3, min=5e2)
#    params.add('center_3', value=19, min=17)
#    params.add('sigma_3', value=0.5, min=0, max=1)
#
#    params.add('amplitude_4', 5e2, min=2e2)
#    params.add('center_4', value=16, min=15, max=17)
#    params.add('sigma_4', value=0.5, min=0, max=1)

    return params

def n_voight_with_bg_500_eV_start_params(x, y):
    x0 = x[np.nanargmax(y)]
    n_lines = 1
    if x0 < 15:
        n_lines += 1
    if x0 < 11.5:
        n_lines += 1

    params = n_voigt_with_bg_start_params(x, y, n_lines=n_lines, bg=False)

    params['center_1'].value = x0
    if 'center_2' in params:
        params['center_2'].value = x0 * 1.5
    if 'center_3' in params:
        params['center_3'].value = x0 * 2.0
    
    centers = [x0, x0 * 1.55, x0 * 2]
    amplitudes = [15000, 1000, 10]
    for i in range(3):
        if 'center_{}'.format(i+1) not in params:
            continue
        p = params['center_{}'.format(i+1)]
        p.min = centers[i] * 0.9
        p.max = centers[i] * 1.1
        p.value = centers[i]
        params['amplitude_{}'.format(i+1)].value = amplitudes[i]
    
    return params

def poly_line(params, x, y=None, err=None):
    mod = np.zeros_like(x)
    for i in range(len(params)):
#        print('\n{}({})'.format(i, len(params)))
#        print(type(params))
#        try:
#            print(params)
#        except:
#            pass
#        print('a_{}'.format(i))
#        print(params['a_{}'.format(i)] * x**i)
#        print(mod)
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


def r_to_e_conversion(params, r, e=None):
    r0 = params['r0'].value
    E0 = params['E0'].value
    a = params['a'].value
    b = params['b'].value
    
    mod = E0 + a * (r - r0) + b * (1/r - 1/r0)
    if e is None:
        return mod
    return mod - e


def r_to_e_conversion_start_params(r=[],e=[]):
    params = lmfit.Parameters()
    params.add('r0', 15, vary=False)
    params.add('E0', e.mean() if len(e) > 0 else 350)
    params.add('a', 1)
    params.add('b', 1)
    return params

def get_params_and_funktion(setting, gas, x, y, bg=None):
    if gas == 'Kr':
        kws = {'line_type': 'voigt'}
        params = start_params(x, y, n_lines=2, **kws)
        fit_funk = n_line_fit_model
    elif gas == 'N2':
        kws = {'bg': bg}
        # get start parameters
        if setting == 500:
            params = skewed_gauss_for_500_eV_start_params(
                x, y, bg=True)
#            params_r_proj['bg_factor'].value = 1
            fit_funk = skewed_gauss_for_500_eV
            for k, v in params.items():
                if k.startswith('skew'):
                    v.value = 0
                    v.vary = False
        else:
            params = n_voigt_with_bg_start_params(x, y, n_lines=1,
                                                         bg=True)
            if 'bg_factor' in params:
                params['bg_factor'].vary = False
            fit_funk = n_voigt_with_bg_model

    return params, fit_funk, kws, 'powel'


def find_lines(rth_image, r_axis_mm, th_axis_rad,
               setting,
               n_lines_fit=2, n_lines_store=2,
               bg=None, gas='Kr', return_line_params_list=False,
               verbose=False):

    # Fit to r projection
    r_projection = rth_image.sum(axis=0)

    params_r_proj_initial, fit_funk, kws, method = get_params_and_funktion(
        setting, gas, r_axis_mm, r_projection, bg)

    r_proj_result = lmfit.minimize(fit_funk, params_r_proj_initial,
                                   method=method,
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

    # Get initial parameters
    line_initial_params, ... = get_params_and_funktion(setting, gas, r_axis_mm,
                                                       r_projection, bg)

    for i_th in range(n_th):
        # map some stuff from the full projection to each line
        amp_scaling = rth_image[i_th, :].sum() / r_projection.sum()
        for k, v in line_initial_params.items():
            v.value = r_proj_result.params[k].value
            if k.startswith(('amp', 'bg')):
                v.value *= amp_scaling
                v.min *= amp_scaling
                v.max *= amp_scaling
            if k.startswith('skew'):
                v.vary = False
#            if k.startswith('center'):
#                # Lock the centers
#                v.vary = False

        line_results_list.append(
            lmfit.minimize(model,
                           line_initial_params, method=method,
                           args=(r_axis_mm, rth_image[i_th, :]),
                           kws=kws))

        line_params_list.append(line_results_list[-1].params)

        for line in range(n_lines_store):
            r[i_th, line] = line_params_list[-1][
                'center_{}'.format(line+1)].value
            # For the width, we need to do some calculations
            # based on
            # https://en.wikipedia.org/wiki/Voigt_profile
            fg = (line_params_list[-1]['sigma_{}'.format(line+1)].value
                * _gauss_fwhm_factor)
            try:
                fl = (line_params_list[-1][
                    'gamma_{}'.format(line+1)].value * 2)
                w[i_th, line] = (0.5346 * fl +
                                 np.sqrt(0.2166 * fl**2 + fg**2)) / 2
            except:
                w[i_th, line] = fg / 2
            w[i_th, line] /= _gauss_fwhm_factor
            a[i_th, line] = line_params_list[-1][
                'amplitude_{}'.format(line+1)].value
        red_chi2[i_th] = line_results_list[-1].redchi

        update_progress(i_th, n_th, verbose=verbose)

    if verbose:
        print('')
        lmfit.report_fit(line_results_list[len(line_results_list)//2])

#    w_1[w_1 <= 0] = np.inf
#    w_2[w_2 <= 0] = np.inf
    if return_line_params_list:
        return r, w, a, red_chi2, line_params_list
    return r, w, a, red_chi2
