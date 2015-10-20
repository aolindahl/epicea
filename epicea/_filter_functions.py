# Filter functions are defined below.
# Names start with  for FilterFunction and ends with _events, _ions or
# _electrons to identify the filter target.
import numpy as np


def e_start_events(data, verbose=False):
    """Get electrons start mask."""
    return data.events.num_e.value > 0


def events_filtered_ions(data, events_filter_name, verbose=False):
    """Get ions mask base on predefined events filter."""
    if verbose:
        print('Retrieving ions from events filter "{}".'.format(
            events_filter_name))
    return data.get_ions_filter(data.get_filter(events_filter_name,
                                                verbose=verbose),
                                verbose=verbose)


def events_filtered_electrons(data, events_filter_name, verbose=False):
    """Get elecctrons mask base on predefined events filter."""
    if verbose:
        print('Retrieving electrons from events filter "{}".'.format(
              events_filter_name))
    if events_filter_name not in data.get_filter_name_list():
        raise NameError('Filter "{}" does not exist.'.format(
            events_filter_name))
    return data.get_electrons_filter(data.get_filter(events_filter_name,
                                                     verbose=verbose),
                                     verbose=verbose)


def has_position_particles(data, particles, verbose=False):
    if verbose:
        print('In has_position_particles() function')
        print('Called with:')
        print('\tdata =', data)
        print('\tparticles =', particles)
        print('Get hdf5 group reference.')
    group = getattr(data, particles)
    if verbose:
        print('get x')
    x = group.pos_x.value
    if verbose:
        print('get y')
    y = group.pos_y.value
    if verbose:
        print('Make mask.')
    mask = np.isfinite(x) & np.isfinite(y)
    if verbose:
        print('Return mask.')
    return mask


def has_tof_ions(data, verbose=False):
    return np.isfinite(data.ions.tof_falling_edge)


def num_ion_events(data, min_ions, max_ions=None, verbose=False):
    if max_ions is None:
        max_ions = min_ions
    num_ions = data.events.num_i.value
    return (min_ions <= num_ions) & (num_ions <= max_ions)


def two_ions_e_start_events(data, verbose=False):
    # Get two ion e start filter
    data.get_filter('two_ions_events', num_ion_events,
                    {'min_ions': 2, 'max_ions': 2},
                    verbose=verbose)
    data.get_filter('e_start_events', e_start_events, verbose=verbose)
    # Combine and return the filters
    return combine(data, ['two_ions_events', 'e_start_events'],
                   verbose=verbose)


def two_ions_time_sum_events(data,
                             t_sum_min_us=-np.inf,
                             t_sum_max_us=np.inf,
                             verbose=False):
    two_ions_e_start_events_mask = data.get_filter(
        'two_ions_e_start_events',
        two_ions_e_start_events,
        verbose=verbose)
    two_ions_e_start_events_ions = data.get_filter(
        'two_ions_e_start_events_ions',
        events_filtered_ions,
        {'events_filter_name': 'two_ions_e_start_events'},
        verbose=verbose)

    time_sums = data.ions.tof_falling_edge[two_ions_e_start_events_ions]
    time_sums = time_sums.reshape(-1, 2).sum(axis=1)
    if verbose:
        print('time_sums =', time_sums)
    wanted_time_sums = ((t_sum_min_us*1e6 < time_sums) &
                        (time_sums < t_sum_max_us*1e6))

    mask = np.zeros_like(two_ions_e_start_events_mask, dtype=bool)
    partial_mask = np.zeros_like(wanted_time_sums, dtype=bool)
    partial_mask[wanted_time_sums] = True
    mask[two_ions_e_start_events_mask] = partial_mask

    return mask


def two_ions_time_diff_events(data,
                             t_diff_min_us=-np.inf,
                             t_diff_max_us=np.inf,
                             verbose=False):
    two_ions_e_start_events_mask = data.get_filter(
        'two_ions_e_start_events',
        two_ions_e_start_events,
        verbose=verbose)
    two_ions_e_start_events_ions = data.get_filter(
        'two_ions_e_start_events_ions',
        events_filtered_ions,
        {'events_filter_name': 'two_ions_e_start_events'},
        verbose=verbose)

    time_diffs = data.ions.tof_falling_edge[two_ions_e_start_events_ions]
    time_diffs = -np.diff(time_diffs.reshape(-1, 2),axis=1).flatten()
    if verbose:
        print('time_diffs =', time_diffs)
    wanted_time_diffs = ((t_diff_min_us * 1e6 < time_diffs) &
                        (time_diffs < t_diff_max_us * 1e6))

    mask = np.zeros_like(two_ions_e_start_events_mask, dtype=bool)
    partial_mask = np.zeros_like(wanted_time_diffs, dtype=bool)
    partial_mask[wanted_time_diffs] = True
    mask[two_ions_e_start_events_mask] = partial_mask

    return mask


def has_energy_electrons(data, verbose=False):
    return np.isfinite(data.electrons.energy)


def electron_energy_uncertainty(data,
                                max_uncertainty=np.inf,
                                verbose=False):
    has_energy_electrons_mask = data.get_filter(
        'has_energy_electrons',
        has_energy_electrons,
        verbose=verbose)

    mask = np.zeros_like(has_energy_electrons_mask, dtype=bool)
    uncertainties = data.electrons.energy_uncertainty[
        has_energy_electrons_mask]
    mask[has_energy_electrons_mask] = uncertainties <= max_uncertainty

    return mask


def electron_angle_range(data, angle_range_list=[0, np.pi], verbose=False):
    has_energy_electrons_mask = data.get_filter(
        'has_energy_electrons',
        has_energy_electrons,
        verbose=verbose)

    try:
        len(angle_range_list[0])
    except TypeError:
        angle_range_list = [angle_range_list]
            

    mask = np.ones_like(has_energy_electrons_mask, dtype=bool)
    angles = data.electrons.pos_t[has_energy_electrons_mask]
    for angle_range in angle_range_list:
#        print('Angle range = {}'.format(angle)}
        mask[has_energy_electrons_mask] *= ((angle_range[0] <= angles) &
                                            (angles <= angle_range[1]))

    return mask

def combine(data, filter_name_list, logic=np.all, verbose=False):
    filter_list = [data.get_filter(filter_name, verbose=verbose) for
                   filter_name in filter_name_list]
    return logic(filter_list, axis=0)


def invert(data, filter_name, verbose=False):
    return ~data.get_filter(filter_name, verbose=verbose)
