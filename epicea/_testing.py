from _data_classes import DataSetList
import _filter_functions as ff


def test_run():
    import matplotlib.pyplot as plt
    plt.ion()
    import os
    import numpy as np

    if not os.path.exists('test_data'):
        os.mkdir('test_data')

    if 'test_data_list' not in locals():
        test_data_list = DataSetList()

    test_data_list.add_dataset(
        name='test_dataset',
        h5_path='test_data/test.h5',
        raw_data_path='test_data',
        verbose=True)

    tof_axis = np.linspace(0, 6.5e6, 0.01e6)
    for test_data in test_data_list:

        tof_hist = test_data.get_i_tof_spectrum(tof_axis)
        plt.figure('tof {}'.format(test_data.name()))
        plt.clf()
        plt.plot(tof_axis, tof_hist)

        test_data.get_filter('e_start_events', ff.e_start_events)
        test_data.get_filter('e_start_ions', ff.events_filtered_ions,
                             {'events_filter_name': 'e_start_events'})
    #    e_start_events = testData.events.num_e.value > 0
    #    e_start_ions = testData.get_ions_filter(e_start_events)
        tof_hist_e_start = test_data.get_i_tof_spectrum(
            tof_axis, test_data.get_filter('e_start_ions'))
        plt.plot(tof_axis, tof_hist_e_start)
