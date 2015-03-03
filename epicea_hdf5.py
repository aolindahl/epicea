# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:38:15 2015

@author: Anton O. Lindahl
"""

import numpy as np
import os
import h5py


def load_file(file_path, verbose=False):
    """Load a tab delimitered txt file.

    Returns a numpy array with the data and a dictionary based of on the
    column header that can be used for indexing.

    file_path
        Path of the text file to load.
    verbose
        bool value that determines if the function outputs diagnostics.
    """
    if verbose:
        print 'Reading data from "{}".'.format(file_path)
    with open(file_path, 'r') as fp:
        head = []
        for i in range(2):
            head.append(fp.readline())
        data = np.loadtxt(fp, delimiter='\t')

    change = [
        [' ', '_'],
        ['id.', 'id'],
        ['.', '_'],
        ['(', '_'],
        [')', ''],
        ['+', ''],
        ['__', '_'],
        ['-', '']
        ]

    h1 = head[1].rstrip('\r\n .')
    for change_from, change_to in change:
        h1 = h1.replace(change_from, change_to)
    col_names = h1.split('\t')
    col_idx = {}
    for i, col in enumerate(col_names):
        col_idx[col] = i
    return data, col_idx


def add_file_as_h5_group(h5_file, name, file_path, verbose=False):
    """Add data from file as group to open hdf5 file.

    h5_file
        Reference to an open hdf5 file.

    name
        Name that should be used for the new group in the hd5f file.

    file_path
        Path to the text file.

    verbose
        bool value that determines if the function outputs diagnostics.
    """
    if name in h5_file.keys():
        if verbose:
            print 'Group "{}" already exists.'.format(name)
        return h5_file[name]
    if verbose:
        print 'Creating group "{}".'.format(name)
    group = h5_file.create_group(name)
    data, col_idx = load_file(file_path, verbose=verbose)

    for name, col in col_idx.iteritems():
        if name in ['num_i', 'start_falling_edge', 'electrons_wave_index',
                    'coincidence_order', 'num_random_hits', 'num_e',
                    'event_id', 'ions_wave_index',
                    'extra_sig_random_wave_index']:
            dtype = 'i'
        else:
            dtype = 'f'

        if name == 'event_id':
            offset = -1
        else:
            offset = 0

        group.create_dataset(name,
                             data=data[:, col] + offset,
                             compression=None,
                             # compression='lzf',
                             dtype=dtype)

    return group


def events_particles_count(data, particles, events_filter=slice(None)):
    """Count the electons/ions per event.

    data
        Data structure of type DataSet that should be investigated.

    particles
        The name of the particles to count. Presently possibllities are
        'ions' and 'electrons'.

    events_filter
        Loogical index array to select certain events before retreaving the
        particles. Default is all events implemented as slice(None).
    """

    # Get the number of particles for all the events
    if particles == 'ions':
        return data.events.num_i[events_filter]
    elif particles == 'electrons':
        return data.events.num_e[events_filter]
    else:
        raise AttributeError('"{}" is not a valid'.format(particles) +
                             ' particle type, use "electrons" or "ions".')


def particles_from_events(data, particles, events_filter=slice(None),
                          verbose=False):
    """Get particle filter from event filter.

    Return a particle filter based on an event filter.

    data
        Data structure of type DataSet that should be investigated.

    particles
        The name of the particles to count. Presently possibllities are
        'ions' and 'electrons'.

    events_filter
        Loogical index array to select certain events before retreaving the
        particles. Default is all events implemented as slice(None).
    """

    particles_number = events_particles_count(data, particles, events_filter)

    particles_start_index = getattr(
        data.events, '{}_wave_index'.format(particles))[events_filter]

    particles_mask = np.zeros(getattr(data, particles).len(), dtype=bool)

    # print 'Partickles start index:', particles_start_index.astype(int)
    # print 'Partickles number:', particles_number.astype(int)
    for start, number in zip(particles_start_index.astype(int),
                             particles_number.astype(int)):
        if number == 0:
            continue
        particles_mask[start:start+number] = True

    return particles_mask


def events_from_particles(data, particles, particles_filter, logic='any'):
    """Get events filter from particles filter.

    Return an events filter based on a particles filter.

    data
        Data structure of type DataSet that should be investigated.

    particles
        The name of the particles the filter is based one.
        Presently possibllities are 'ions' and 'electrons'.

    particles_filter
        Loogical index array to select certain particles.

    logic
        Logic operator to be applied when creating the events filter.
        Can be 'any', 'all', int or list of int, to select the particles.
    """
    # Get the event numbers for the selected particles
    event_id_from_source, num_valid_particles = np.unique(
        getattr(data, particles).event_id[particles_filter])

    # Make the event mask vector
    events_mask = np.zeros(data.events.len(), dtype=bool)
    # Make a first version
    events_mask[event_id_from_source] = True
    # This is enought for the 'any' logic
    if logic == 'any':
        return events_mask

    # Get the number of particles per event
    particles_number = events_particles_count(data, particles,
                                              events_mask)
    if logic == 'all':
        valid_event_id = event_id_from_source[
            num_valid_particles == particles_number]
    elif isinstance(logic, int):
        valid_event_id = event_id_from_source[num_valid_particles == logic]
    elif isinstance(logic, list):
        valid_event_id = event_id_from_source[num_valid_particles in logic]

    # Update the event mask
    events_mask[:] = False
    events_mask[valid_event_id] = True
    return events_mask


def limits_from_centers(centers):
    """Get the bin limits correspondinc to given centers vector"""

    centers = centers.astype(float)
    limits = np.empty(len(centers)+1)
    limits[0] = centers[0] - np.diff(centers[0:2])/2
    limits[-1] = centers[-1] + np.diff(centers[-2:])/2
    limits[1:-1] = centers[:-1] + np.diff(centers)/2
    return limits


def center_histogram(data, centers):
    """Rerurn histogram vector corresponding to given centers"""

    limits = limits_from_centers(centers)
    hist, _ = np.histogram(data, limits)
    return hist


def center_histogram_2d(x_data, y_data, x_centers, y_centers=None):
    """Rerurn histogram array corresponding to given centers"""

    x_limits = limits_from_centers(x_centers)
    if y_centers is None:
        y_limits = x_limits
    else:
        y_limits = limits_from_centers(y_centers)
    hist, _, _ = np.histogram2d(x_data, y_data, [x_limits, y_limits])
    return hist.T


class GroupContainer(object):
    """Wrapper class for a group in an hdf5 file.

    Costructed to me used together with the DataSet class.
    """
    def __init__(self, h5_file, path, name, verbose=False):
        if not isinstance(h5_file, h5py._hl.files.File):
            if verbose:
                print '"{}" is not a valid hdf5 file.'.format(h5_file)
            return
        else:
            self._h5_file = h5_file

        source_file = os.path.join(path, '.'.join([name, 'txt']))
        self._group = add_file_as_h5_group(self._h5_file, name, source_file,
                                           verbose=verbose)
        if verbose:
            print 'Reference "{}" stored in "_group".'.format(
                self._group)

        # Expose datasets
        for k, v in self._group.iteritems():
            setattr(self, k, v)
            if verbose:
                print '\t{}'.format(k)
        setattr(self, 'len', self.event_id.len)

        self._verbose = verbose

    def get_hdf5_group(self):
        return self._group

    def correct_center(self, x_shift=0., y_shift=0.):
        if self._verbose:
            print 'Correcting center of {} with to x = {} y = {}.'.format(
                self._group.name.lstrip('/'), x_shift, y_shift)
        if (not hasattr(self, 'pos_x')) or (not hasattr(self, 'pos_y')):
            return
        dset_name = 'xy_shift'
        if dset_name not in self._group.keys():
            self._group.create_dataset(dset_name, data=(0, 0), dtype=float)
        x_shift_old, y_shift_old = self._group[dset_name].value
        if self._verbose:
            print 'Old correction was x = {} y = {}.'.format(x_shift_old,
                                                             y_shift_old)
        x_shift_change = x_shift - x_shift_old
        y_shift_change = y_shift - y_shift_old
        if np.isclose(x_shift_change, 0) and np.isclose(y_shift_change, 0):
            if self._verbose:
                print 'No adjustment to the position.'
            return
        if self._verbose:
            print 'Adjustment is x = {} y = {}.'.format(x_shift_change,
                                                        y_shift_change)
        self.pos_x[:] += x_shift_change
        self.pos_y[:] += y_shift_change
        self.recalculate_polar_coordinates()

        self._group[dset_name][:] = [x_shift, y_shift]

        setattr(self, dset_name, self._group[dset_name])

    def recalculate_polar_coordinates(self):
        if self._verbose:
            print 'Recalculating the polar coordinates of {}.'.format(
                self._group.name.lstrip('/'))
        x = self.pos_x.value
        y = self.pos_y.value
        self.pos_r[:] = np.sqrt(x**2 + y**2)
        self.pos_t[:] = np.arctan(y/x)
        self.pos_t[x < 0] += np.pi
        self.pos_t[(x > 0) & (y < 0)] += 2*np.pi

_GROUP_NAMES = ['electrons', 'ions', 'events']


class DataSet(object):
    def __init__(self, data_path, h5_path, verbose=False):
        """Setup link to hdf5 file and its groups"""
        self._data_path = data_path.rstrip('/')
        self._h5_path = h5_path
        self._verbose = verbose

        if verbose:
            print 'Open or create hdf5 file "{}".'.format(self._h5_path)
        self._h5_file = h5py.File(self._h5_path, mode='a')

        for name in _GROUP_NAMES:
            if verbose:
                print 'Adding the group "{}" to the hd5f file.'.format(name)
            setattr(self, name,
                    GroupContainer(self._h5_file, data_path,
                                   name, verbose=verbose))

        self._filters = {}

    def __del__(self):
        """Close the hdf5 file."""
        if self._verbose:
            print 'DataSet->destructor, closing hdf5 file "{}".'.format(
                self._h5_file.filename)
        self._h5_file.close()

    def list_filters(self):
        """Print a list of existing filters."""
        print ('Object "{}" connected to the hdf5 file "{}" have' +
               ' the following filters defined.').format(self.__class__,
                                                         self._h5_path)
        for k in self._filters:
            print '\t{}'.format(k)

    def get_filter_name_list(self):
        return [k for k in self._filters]

    def get_filter(self, filter_name, filter_function=None,
                   filter_kw_params={}, update=False, verbose=None):
        """Return mask for previously made filter or make a new one."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print 'Check if the filter "{}" is already created.'.format(
                filter_name)
        if (filter_name in self._filters) and (update is False):
            if verbose:
                print 'Returning existing filter.'
            return self._filters[filter_name].copy()

        if filter_function is None:
            if verbose:
                print ('If no filter function is given and the filter does' +
                       ' not exist, an exception is raised.')
            raise NameError('No filter named "{}"'.format(filter_name) +
                            ' created previously and no mask given.')

        filter_kw_params['verbose'] = verbose
        if verbose:
            print ('Construct the filter from the function {} with' +
                   ' kwyword parameters: {}.').format(filter_function,
                                                      filter_kw_params)
        self._filters[filter_name] = filter_function(self, **filter_kw_params)
        if self._verbose:
            print 'Return the filter mask.'
        return self._filters[filter_name].copy()

    def get_events_filter(self, source, source_filter, logic='any'):
        """Return event maske based on ions or electrons mask."""
        return events_from_particles(self, source, source_filter, logic=logic)

    def get_electrons_filter(self, events_filter):
        """Return electrons maske based on events mask."""
        return particles_from_events(self, 'electrons', events_filter)

    def get_ions_filter(self, events_filter):
        """Return ions mask based on events mask."""
        return particles_from_events(self, 'ions', events_filter)

    def resolve_filters(self, target, events_filter=None,
                        ions_filter=None, ions_logic='any',
                        electrons_filter=None, electrons_logic='any'):
        """Combine events, ions and electrons filter masks."""
        if events_filter is None:
            events_filter = np.ones(self.events.len(), dtype=bool)
        if ions_filter is None:
            ions_filter = np.ones(self.ions.len(), dtype=bool)
        if electrons_filter is None:
            electrons_filter = np.ones(self.electrons.len(), dtype=bool)

        events_filter *= (self.get_events_filter('electrons',
                                                 electrons_filter,
                                                 electrons_logic) *
                          self.get_events_filter('ions',
                                                 ions_filter,
                                                 ions_logic))
        if target == 'evetns':
            return events_filter

        if target == 'ions':
            return ions_filter * self.get_ions_filter(events_filter)

        if target == 'electrons':
            return electrons_filter * self.get_electrons_filter(events_filter)

    def e_start_events(self):
        """Get mask for electron start events."""
        return self.events.num_e > 0

    def get_i_tof_spectrum(self, t_axis, ions_filter=None, verbose=None):
        """Get ion tof spectrum on given time axis and ions filter"""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print 'In get_i_tof_spectrum.'

        has_tof = self.get_filter('has_tof_ions',
                                  filter_function=ff_has_tof_ions)
        if ions_filter is None:
            ions_filter = has_tof
        else:
            ions_filter *= has_tof

        if verbose:
            print 'Getting ion tof data.'
        tof = self.ions.tof_falling_edge[ions_filter]
        if verbose:
            print 'Making histogram.'
        hist = center_histogram(tof, t_axis)
        if verbose:
            print 'Returning.'
        return hist

    def get_i_xy_image(self, x_axis_mm, y_axis_mm=None, ions_filter=None):
        """Get the ion image based on ions_filter."""
        if self._verbose:
            print 'Get the has_position mask.'
        has_pos = self.get_filter('has_position_ions',
                                  ff_has_position_particles,
                                  {'particles': 'ions'})
        if self._verbose:
            print 'Merge has_pos filter and any given ions_filter.'
        if ions_filter is None:
            ions_filter = has_pos
        else:
            ions_filter *= has_pos

        if y_axis_mm is None:
            y_axis_mm = x_axis_mm

        if self._verbose:
            print 'Calculate and return image histogram.'
        return center_histogram_2d(self.ions.pos_x[ions_filter],
                                   self.ions.pos_y[ions_filter],
                                   x_axis_mm, y_axis_mm)

    def get_i_rth_image(self,
                        r_axis_mm,
                        th_axis_rad=np.linspace(0, 2*np.pi, 513)[1::2],
                        ions_filter=None):
        """Get the ions image in polar coordinates."""
        has_pos = self.get_filter('has_position_ions',
                                  ff_has_position_particles,
                                  {'particles': 'ions'})
        if ions_filter is None:
            ions_filter = has_pos
        else:
            ions_filter *= has_pos

        return center_histogram_2d(self.ions.pos_r[ions_filter],
                                   self.ions.pos_t[ions_filter],
                                   r_axis_mm, th_axis_rad)

    def get_e_xy_image(self, x_axis_mm, y_axis_mm=None,
                       electrons_filter=None):
        """Get the electron image based on electrons_filter."""
        if self._verbose:
            print 'Get the has_position mask.'
        has_pos = self.get_filter('has_position_electrons',
                                  ff_has_position_particles,
                                  {'particles': 'electrons'})
        if self._verbose:
            print 'Merge has_pos filter and any given electrons_filter.'
        if electrons_filter is None:
            electrons_filter = has_pos
        else:
            electrons_filter *= has_pos

        if y_axis_mm is None:
            y_axis_mm = x_axis_mm

        if self._verbose:
            print 'Calculate and return electons image histogram.'
        return center_histogram_2d(self.electrons.pos_x[electrons_filter],
                                   self.electrons.pos_y[electrons_filter],
                                   x_axis_mm, y_axis_mm)

    def get_e_rth_image(self,
                        r_axis_mm,
                        th_axis_rad=np.linspace(0, 2*np.pi, 513)[1::2],
                        electrons_filter=None):
        """Get the electrons image in polar coordinates."""
        has_pos = self.get_filter('has_position_electrons',
                                  ff_has_position_particles,
                                  {'particles': 'electrons'})
        if electrons_filter is None:
            electrons_filter = has_pos
        else:
            electrons_filter *= has_pos

        return center_histogram_2d(self.electrons.pos_r[electrons_filter],
                                   self.electrons.pos_t[electrons_filter],
                                   r_axis_mm, th_axis_rad)


# Filter functions are defined below.
# Names start with ff_ for FilterFunction and ends with _events, _ions or
# _electrons to identify the filter target.
def ff_e_start_events(data, verbose=False):
    """Get electrons start mask."""
    return data.events.num_e.value > 0


def ff_events_filtered_ions(data, events_filter_name, verbose=False):
    """Get ions mask base on predefined events filter."""
    if verbose:
        print 'Retrieving ions from events filter "{}".'.format(
            events_filter_name)
    return data.get_ions_filter(data.get_filter(events_filter_name))


def ff_events_filtered_electrons(data, events_filter_name, verbose=False):
    """Get elecctrons mask base on predefined events filter."""
    if verbose:
        print 'Retrieving electrons from events filter "{}".'.format(
            events_filter_name)
    if events_filter_name not in data.get_filter_name_list():
        raise NameError('Filter "{}" does not exist.'.format(
            events_filter_name))
    return data.get_electrons_filter(data.get_filter(events_filter_name))


def ff_has_position_particles(data, particles, verbose=False):
    if verbose:
        print 'In ff_has_position_particles() function'
        print 'Called with:'
        print '\tdata =', data
        print '\tparticles =', particles
        print 'Get hdf5 group reference.'
    group = getattr(data, particles)
    if verbose:
        print 'get x'
    x = group.pos_x.value
    if verbose:
        print 'get y'
    y = group.pos_y.value
    if verbose:
        print 'Make mask.'
    mask = np.isfinite(x) & np.isfinite(y)
    if verbose:
        print 'Return mask.'
    return mask


def ff_has_tof_ions(data, verbose=False):
    return np.isfinite(data.ions.tof_falling_edge)


def ff_num_ion_events(data, min_ions, max_ions=None, verbose=False):
    if max_ions is None:
        max_ions = min_ions
    num_ions = data.events.num_i.value
    return (min_ions <= num_ions) & (num_ions <= max_ions)


def ff_two_ions_e_start_events(data, verbose=False):
    # Get two ion e start filter
    data.get_filter('two_ions_events', ff_num_ion_events,
                    {'min_ions': 2, 'max_ions': 2},
                    verbose=verbose)
    data.get_filter('e_start_events', ff_e_start_events, verbose=verbose)
    # Combine and return the filters
    return ff_combine(data, ['two_ions_events', 'e_start_events'],
                      verbose=verbose)


def ff_two_ions_time_sum_events(data,
                                t_sum_min_us=-np.inf,
                                t_sum_max_us=np.inf,
                                verbose=False):
    two_ions_e_start_events = data.get_filter('two_ions_e_start_events',
                                              ff_two_ions_e_start_events,
                                              verbose=verbose)
    two_ions_e_start_events_ions = data.get_filter(
        'two_ions_e_start_events_ions',
        ff_events_filtered_ions,
        {'events_filter_name': 'two_ions_e_start_events'},
        verbose=verbose)

    time_sums = data.ions.tof_falling_edge[two_ions_e_start_events_ions]
    time_sums = time_sums.reshape(-1, 2).sum(axis=1)
    if verbose:
        print 'time_sums =', time_sums
    wanted_time_sums = ((t_sum_min_us*1e6 < time_sums) &
                        (time_sums < t_sum_max_us*1e6))

    mask = np.zeros_like(two_ions_e_start_events, dtype=bool)
    partial_mask = np.zeros_like(wanted_time_sums, dtype=bool)
    partial_mask[wanted_time_sums] = True
    mask[two_ions_e_start_events] = partial_mask

    return mask


def ff_combine(data, filter_name_list, logic=np.all, verbose=False):
    filter_list = [data.get_filter(filter_name) for filter_name in
                   filter_name_list]
    return logic(filter_list, axis=0)


def ff_invert(data, filter_name, verbose=False):
    return ~data.get_filter(filter_name)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    testData = None

    folder = '../data/ExportedData/N2O_0029_KE373_hv430eV/'
    h5_name = 'test_data/N20_430_high.h5'
    if not os.path.exists('test_data'):
        os.mkdir('test_data')
    testData = DataSet(folder, h5_name, verbose=True)

    tof_axis = np.linspace(0, 6.5e6, 0.01e6)
    tof_hist = testData.get_i_tof_spectrum(tof_axis)
    plt.figure('tof')
    plt.clf()
    plt.plot(tof_axis, tof_hist)

    testData.get_filter('e_start_events', ff_e_start_events)
    testData.get_filter('e_start_ions', ff_events_filtered_ions,
                        {'events_filter_name': 'e_start_events'})
#    e_start_events = testData.events.num_e.value > 0
#    e_start_ions = testData.get_ions_filter(e_start_events)
    tof_hist_e_start = testData.get_i_tof_spectrum(
        tof_axis, testData.get_filter('e_start_ions'))
    plt.plot(tof_axis, tof_hist_e_start)
