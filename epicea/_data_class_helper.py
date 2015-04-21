# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:38:15 2015

@author: Anton O. Lindahl
"""

import numpy as _np


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
        data = _np.loadtxt(fp, delimiter='\t')

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
    elif file_path is None:
        raise AttributeError('No raw data given and not an existing group.')
    if verbose:
        print 'Creating group "{}".'.format(name)
    group = h5_file.create_group(name)
    data, col_idx = load_file(file_path, verbose=verbose)

    for name, col in col_idx.iteritems():
        if name in ['num_i', 'electrons_wave_index',
                    'coincidence_order', 'num_random_hits', 'num_e',
                    'event_id', 'ions_wave_index',
                    'extra_sig_random_wave_index']:
            dtype = 'i'
        elif name in ['start_falling_edge']:
            dtype = 'int64'
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

    particles_mask = _np.zeros(getattr(data, particles).len(), dtype=bool)

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
    event_id_from_source, num_valid_particles = _np.unique(
        getattr(data, particles).event_id[particles_filter])

    # Make the event mask vector
    events_mask = _np.zeros(data.events.len(), dtype=bool)
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
    limits = _np.empty(len(centers)+1)
    limits[0] = centers[0] - _np.diff(centers[0:2])/2
    limits[-1] = centers[-1] + _np.diff(centers[-2:])/2
    limits[1:-1] = centers[:-1] + _np.diff(centers)/2
    return limits


def center_histogram(data, centers):
    """Rerurn histogram vector corresponding to given centers"""

    limits = limits_from_centers(centers)
    hist, _ = _np.histogram(data, limits)
    return hist


def center_histogram_2d(x_data, y_data, x_centers, y_centers=None):
    """Rerurn histogram array corresponding to given centers"""

    x_limits = limits_from_centers(x_centers)
    if y_centers is None:
        y_limits = x_limits
    else:
        y_limits = limits_from_centers(y_centers)
    hist, _, _ = _np.histogram2d(x_data, y_data, [x_limits, y_limits])
    return hist.T
