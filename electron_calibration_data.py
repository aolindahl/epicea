# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:23 2015

@author: Anton O Lindhal
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import plt_func
if 'epicea' not in locals():
    import epicea

h5_base_path = 'h5_data'
raw_data_base_path = '../data/ExportedData'
if not os.path.exists(h5_base_path):
    os.mkdir(h5_base_path)


class ModuleError(Exception):
    pass


class CenterEnergyError(ModuleError):
    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg


def get_data_in_list(center_energy, verbose=False):
    try:
        center_energy = int(center_energy)
    except:
        raise TypeError('Function get_data_in_list() called with ' +
                        'center_energy of type ' +
                        '{} '.format(type(center_energy)) +
                        'that could not be converted to type int.')
    # Add the data sets to the list.
    # And store the corresponding photon energy values

    calib_data_list = epicea.DataSetList()

    ###########################################################################
    # 373 eV pass energy
    if 373 == center_energy:
        energy_and_name = {
#            471.0: 'CalibrationsForKE373eV/Kr_0023',
#            470.0: 'CalibrationsForKE373eV/Kr_0024',
            469.0: 'CalibrationsForKE373eV/Kr_0022',
            467.0: 'CalibrationsForKE373eV/Kr_calib_00160021',
            465.0: 'CalibrationsForKE373eV/Kr_calib_00160020',
            463.0: 'CalibrationsForKE373eV/Kr_calib_00150019',
            462.0: 'CalibrationsForKE373eV/Kr_0025',
            }

        data_scaling = {
            471.0: 227.547,
            470.0: 195.831,
            469.0: 402.566,
            467.0: 236.233,
            465.0: 234.042,
            463.0: 219.345,
            462.0: 260.802
            }

        for photon_energy, data_path in energy_and_name.iteritems():
            name = 'Kr_{}_eV'.format(photon_energy).replace('.', '_')
            h5_name = 'center_{}_eV_{}.h5'.format(center_energy, name)
            calib_data_list.add_dataset(
                name=name,
                h5_path=os.path.join(h5_base_path, h5_name),
                raw_data_path=os.path.join(raw_data_base_path, data_path),
                photon_energy=photon_energy,
                electron_center_energy=center_energy,
                verbose=verbose,
                data_scaling=data_scaling[photon_energy])

    ###########################################################################
    # 366 eV pass energy
    elif center_energy == 366:
        energy_and_name = {
#            453.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE453eV0007',
            455.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE455eV0006',
            457.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE457eV0005',
            459.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE459eV0002',
            461.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE461eV0003',
            # 463.3: 'CalibrationsForKE366eV/Kr_KEc366eV_PE463eV0004',
            }

        data_scaling = {
            453.3: 401.248,
            455.3: 414.556,
            457.3: 401.537,
            459.3: 401.599,
            461.3: 400.553,
            463.3: 400.602,
            }

        for photon_energy, data_path in energy_and_name.iteritems():
            name = 'Kr_{}_eV'.format(photon_energy).replace('.', '_')
            h5_name = 'center_{}_eV_{}.h5'.format(center_energy, name)
            calib_data_list.add_dataset(
                name=name,
                h5_path=os.path.join(h5_base_path, h5_name),
                raw_data_path=os.path.join(raw_data_base_path, data_path),
                photon_energy=photon_energy,
                electron_center_energy=center_energy,
                verbose=verbose,
                data_scaling=data_scaling[photon_energy])

    ###########################################################################
    # 357 eV pass energy
    elif center_energy == 357:
        energy_and_name = {
#            749.0: 'CalibrationsForKE357eV/N2_calib_0038',
            753.5: 'CalibrationsForKE357eV/N2_calib_0039',
            758.0: 'CalibrationsForKE357eV/N2_calib_0040',
            762.5: 'CalibrationsForKE357eV/N2_calib_0041',
            767.0: 'CalibrationsForKE357eV/N2_calib_0042',
            771.5: 'CalibrationsForKE357eV/N2_calib_0043',
            774.5: 'CalibrationsForKE357eV/N2_calib_0045',
            }

        data_scaling = {
            749.0: 200.494,
            753.5: 200.753,
            758.0: 200.809,
            762.5: 215.432,
            767.0: 248.55,
            771.5: 250.622,
            774.5: 251.431,
            }

        for photon_energy, data_path in energy_and_name.iteritems():
            name = 'N2_{}_eV'.format(photon_energy).replace('.', '_')
            h5_name = 'center_{}_eV_{}.h5'.format(center_energy, name)
            calib_data_list.add_dataset(
                name=name,
                h5_path=os.path.join(h5_base_path, h5_name),
                raw_data_path=os.path.join(raw_data_base_path, data_path),
                photon_energy=photon_energy,
                electron_center_energy=center_energy,
                verbose=verbose,
                data_scaling=data_scaling[photon_energy])

    ###########################################################################
    # 357 eV pass energy
    elif center_energy == 500:
        energy_and_name = {
            910.0: 'CalibrationsForKE500eV/N2_el_img_0025',
            912.0: 'CalibrationsForKE500eV/N2_el_img_0026',
            914.0: 'CalibrationsForKE500eV/N2_el_img_0027',
            916.0: 'CalibrationsForKE500eV/N2_el_img_0028',
            906.0: 'CalibrationsForKE500eV/N2_el_img_0029',
            900.0: 'CalibrationsForKE500eV/N2_el_img_0030',
            896.0: 'CalibrationsForKE500eV/N2_el_img_0031',
            893.0: 'CalibrationsForKE500eV/N2_el_img_0032',
            891.0: 'CalibrationsForKE500eV/N2_el_img_0033',
            }

        data_scaling = {
            910.0: 453.734,
            912.0: 302.521,
            914.0: 301.536,
            916.0: 300.659,
            906.0: 301.808,
            900.0: 303.124,
            896.0: 302.087,
            893.0: 301.353,
            891.0: 159.264,
            }

        for photon_energy, data_path in energy_and_name.iteritems():
            name = 'N2_{}_eV'.format(photon_energy).replace('.', '_')
            h5_name = 'center_{}_eV_{}.h5'.format(center_energy, name)
            calib_data_list.add_dataset(
                name=name,
                h5_path=os.path.join(h5_base_path, h5_name),
                raw_data_path=os.path.join(raw_data_base_path, data_path),
                photon_energy=photon_energy,
                electron_center_energy=center_energy,
                verbose=verbose,
                data_scaling=data_scaling[photon_energy])

    else:
        raise CenterEnergyError('No data for center energy {} eV.'.format(
            center_energy))

    return calib_data_list


def center_check(data_list):
    x_axis_mm = np.linspace(-23, 23, 2**9)
    r_axis_mm = np.linspace(0, 23, 2**9+1)[1::2]
    th_axis_rad = np.linspace(0, np.pi*2, 2**9+1)[1::2]
    xy_center_slice = slice(x_axis_mm.searchsorted(-1),
                            x_axis_mm.searchsorted(1, side='right'))
    x = []
    y = []
    r = []
    th = []
    for data in data_list:
        x.extend(data.electrons.pos_x.value)
        y.extend(data.electrons.pos_y.value)

        if np.any(data.electrons.pos_t.value > np.pi*2):
            data.electrons.recalculate_polar_coordinates()

        r.extend(data.electrons.pos_r.value)
        th.extend(data.electrons.pos_t.value)

    image_xy = epicea.center_histogram_2d(x, y, x_axis_mm)
    x_slice = image_xy[xy_center_slice, :].sum(axis=0)
    y_slice = image_xy[:, xy_center_slice].sum(axis=1)
    image_rt = epicea.center_histogram_2d(r, th, r_axis_mm, th_axis_rad)

    plt_func.figure_wrapper('Electron data {} eV pass'.format(
        data_list[0].electron_center_energy()))
    plt.subplot(231)
    plt_func.imshow_wrapper(image_xy, x_axis_mm)

    plt.subplot(234)
    plt_func.imshow_wrapper(image_rt, r_axis_mm, th_axis_rad,
                            kw_args={'aspect': 'auto'})

    plt.subplot(132)
    plt.plot(x_axis_mm, x_slice, label='normal')
    plt.plot(x_axis_mm[::-1], x_slice, label='flipped')
    plt_func.title_wrapper('x slice')
    plt_func.legend_wrapper()
    plt.xlim(xmin=7)

    plt.subplot(133)
    plt.plot(x_axis_mm, y_slice, label='normal')
    plt.plot(x_axis_mm[::-1], y_slice, label='flipped')
    plt_func.title_wrapper('y slice')
    plt_func.legend_wrapper()
    plt.xlim(xmin=7)


if __name__ == '__main__':
    verbose = True
    center_energy_list = [373, 366, 357, 500]
    center_energy_list = [500]
    center_energy_list = [373]
    data_list = {}
    for center_energy in center_energy_list:
        data_list[center_energy] = get_data_in_list(center_energy, verbose)

        #     Need to be done only once
#        for data in data_list[center_energy]:
#            data.electrons.recalculate_polar_coordinates()

        center_check(data_list[center_energy])
