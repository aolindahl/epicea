# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:38:15 2015

@author: Anton O. Lindahl
"""
import h5py
import os.path
import numpy as np
import sys
import time

from . import _filter_functions as ff
from . import _data_class_helper as _helper


class GroupContainer(object):
    """Wrapper class for a group in an hdf5 file.

    Costructed to me used together with the DataSet class.
    """
    def __init__(self, h5_file, path, name, verbose=False):
        if not isinstance(h5_file, h5py.File):
            if verbose:
                print('"{}" is not a valid hdf5 file.'.format(h5_file))
                sys.stdout.flush()
            return
        else:
            self._h5_file = h5_file

        source_file = os.path.join(path, '.'.join([name, 'txt']))
        self._group = _helper.add_file_as_h5_group(self._h5_file, name,
                                                   source_file,
                                                   verbose=verbose)
        if verbose:
            print('Reference "{}" stored in "_group".'.format(
                self._group), flush=True)

        # Expose datasets
        for k, v in self._group.items():
            setattr(self, k, v)
            if verbose:
                print('\t{}'.format(k), flush=True)
        setattr(self, 'len', self.event_id.len)

        self._verbose = verbose

    def get_hdf5_group(self):
        return self._group

    def correct_center(self, x_shift=0., y_shift=0.):
        if self._verbose:
            print('Correcting center of {} with to x = {} y = {}.'.format(
                self._group.name.lstrip('/'), x_shift, y_shift), flush=True)
        if (not hasattr(self, 'pos_x')) or (not hasattr(self, 'pos_y')):
            return
        dset_name = 'xy_shift'
        if dset_name not in self._group.keys():
            self._group.create_dataset(dset_name, data=(0, 0), dtype=float)
            new_dataset = True
        else:
            new_dataset = False

        x_shift_old, y_shift_old = self._group[dset_name].value

        if self._verbose:
            print('Old correction was x = {} y = {}.'.format(x_shift_old,
                                                             y_shift_old),
                  flush=True)
        x_shift_change = x_shift - x_shift_old
        y_shift_change = y_shift - y_shift_old
        if np.isclose(x_shift_change, 0) and np.isclose(y_shift_change, 0):
            if self._verbose:
                print('No adjustment to the position.', flush=True)
            if new_dataset:
                if self._verbose:
                    print('New dataset, recalculating polar coordinates.',
                          flush=True)
                self.recalculate_polar_coordinates()
            return
        if self._verbose:
            print('Adjustment is x = {} y = {}.'.format(x_shift_change,
                                                        y_shift_change),
                  flush=True)
        self.pos_x[:] += x_shift_change
        self.pos_y[:] += y_shift_change
        self.recalculate_polar_coordinates()

        self._group[dset_name][:] = [x_shift, y_shift]

        setattr(self, dset_name, self._group[dset_name])

    def recalculate_polar_coordinates(self):
        if self._verbose:
            print('Recalculating the polar coordinates of {}.'.format(
                  self._group.name.lstrip('/')), flush=True)
        x = self.pos_x.value
        y = self.pos_y.value
        self.pos_r[:] = np.sqrt(x**2 + y**2)
        self.pos_t[:] = np.arctan(y/x)
        self.pos_t[x < 0] += np.pi
        self.pos_t[(x > 0) & (y < 0)] += 2*np.pi

    def add_parameter(self, name, data):
        if len(data) != self.len():
            raise IndexError('First dimension of new "data" must be the' +
                             ' same size as the rest of the datasets.')

        if name in self._group.keys():
            ds = self._group[name]
            if (ds.dtype != float) or (ds.shape != data.shape):
                del self._group[name]
            ds[:] = data

        if name not in self._group.keys():
            ds = self._group.create_dataset(name, dtype=float,
                                            data=data)
        setattr(self, name, ds)

#    def calculate_hit_time(self):
#        if hasattr(self, 'hit_time'):
#            return
#
#        if hasattr(self, 'psd_sig_w1'):
#            t = np.nanmean([self.psd_sig_u1.value, self.psd_sig_u2.value,
#                            self.psd_sig_v1.value, self.psd_sig_v2.value,
#                            self.psd_sig_w1.value, self.psd_sig_w2.value],
#                            axis=0)
#
#        elif hasattr(self, 'psd_sig_u1'):
#            t = np.nanmean([self.psd_sig_u1.value, self.psd_sig_u2.value,
#                            self.psd_sig_v1.value, self.psd_sig_v2.value],
#                            axis=0)
#
#        else:
#            return
#
#        self.add_parameter('hit_time', t)

_GROUP_NAMES = ['electrons', 'ions', 'events']


class DataSet(object):
    def __init__(self, data_name, h5_path, raw_data_path='',
                 photon_energy=None, electron_center_energy=None,
                 verbose=False,
                 **kwargs):
        """Setup link to hdf5 file and its groups"""
        self._data_path = raw_data_path.rstrip('/')
        self._h5_path = h5_path
        self._name = data_name
        self._photon_energy = photon_energy
        self._electron_center_energy = electron_center_energy
        self._verbose = verbose

        if self._verbose:
            print('Open or create hdf5 file "{}".'.format(self._h5_path),
                  flush=True)
        self._h5_file = h5py.File(self._h5_path, mode='a')

        # Iterate over all the group names
        for group_name in _GROUP_NAMES:
            # Print status message
            self._verbose_message('Looking for {}.'.format(group_name))

            # Look for the raw data file of the current group
            if not os.path.exists(os.path.join(self._data_path,
                                               '.'.join([group_name, 'txt']))):
                # Print a message if it does not exist
                self._verbose_message(
                    'No data file for {} in the folder "{}".'.format(
                        group_name, self._data_path))
                # and continue to the next group
                continue

            # If it does exist add it to the hd5f file
            self._verbose_message(
                'Adding the group "{}" to the hd5f file.'.format(group_name))
            setattr(self, group_name,
                    GroupContainer(self._h5_file, self._data_path,
                                   group_name, verbose=self._verbose))

#            getattr(self, group_name).calculate_hit_time()

        try:
            if 'polar_recalculated' not in self._h5_file['electrons'].attrs:
                self.electrons.recalculate_polar_coordinates()
                self._h5_file['electrons'].attrs['polar_recalculated'] = True
        except:
            print('Problems with the polar coordinates.')

#        self.calculate_event_times()

        self._filters = {}

        derived_name = self._h5_file.filename.replace('.h5', '_derived.h5')
        try:
            self._derived_data = h5py.File(derived_name, 'a')
        except IOError:
            os.remove(derived_name)
            self._derived_data = h5py.File(derived_name, 'a')

        # Set properties from the kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __del__(self):
        """Close the hdf5 file."""
        if self._verbose:
            print('DataSet->destructor, closing hdf5 file "{}".'.format(
                self._h5_file.filename), flush=True)
        self._h5_file.close()
        self._derived_data.close()

    def _verbose_message(self, message):
        """Print and flush message if verbose."""
        if self._verbose:
                print(message, flush=True)

    def name(self):
        return self._name

    def photon_energy(self):
        return self._photon_energy

    def electron_center_energy(self):
        return self._electron_center_energy

#    def calculate_event_times(self):
#        e_idx = self.events.electrons_wave_index.value
#        has_e = e_idx >= 0
#        e1_time = np.nan * np.ones_like(e_idx, dtype=float)
#        e1_time[has_e] = self.electrons.hit_time.value[e_idx[has_e]]
#        
#        self.events.add_parameter('e1_time', e1_time)
#
#        ions_hit_times = self.ions.hit_time.value
#        i_idx = self.events.ions_wave_index.value
#        i_num = self.events.num_i.value
#
#        for i in range(2):
#            has_i = i < i_num
#            i_time = np.nan * np.ones_like(i_idx, dtype=float)
#            i_time[has_i] = ions_hit_times[i_idx[has_i] + i]
#        
#            self.events.add_parameter('i{}_time'.format(i+1), i_time)

    def list_filters(self):
        """Print a list of existing filters."""
        print(('Object "{}" connected to the hdf5 file "{}" have' +
               ' the following filters defined.').format(self.__class__,
                                                         self._h5_path),
              flush=True)
        for k in self._filters:
            print('\t{}'.format(k))

    def get_filter_name_list(self):
        return [k for k in self._filters]

    def get_filter(self, filter_name, filter_function=None,
                   filter_kw_params={}, update=False, verbose=None):
        """Return mask for previously made filter or make a new one."""
        # If no verbose parameter is given:
        if verbose is None:
            # use the value from the class
            verbose = self._verbose
        if verbose:
            print('Check if the filter "{}" is already created.'.format(
                  filter_name), flush=True)

        # If the filter name is in the filter list and should not be updated...
        if (filter_name in self._filters) and (update is False):
            if verbose:
                print('Returning existing filter.', flush=True)
            # ...return the exixting filter
            return self._filters[filter_name].copy()

        # Make sure that the filters group is in the derived data hdf5 file
        filters = self._derived_data.require_group('filters')
        # If it should not be updates and if is already exists and it has a
        # time_stamp attribute, which is newer than the time_stamp of the
        # correspnding function...
        if ((update is False) and (filter_name in filters.keys()) and
            ('time_stamp' in filters[filter_name].attrs.keys()) and
            ((not hasattr(filter_function, 'time_stamp')) or
             (filters[filter_name].attrs['time_stamp'] <
              filter_function.time_stamp))):
            # ...get the filter from file
            if verbose:
                print('Get filter "{}" from hdf5 file.'.format(filter_name),
                      flush=True)
            self._filters[filter_name] = filters[filter_name].value
            # And return a copy
            return self._filters[filter_name].copy()

        # If there is no filter already the filter function have to be provided
        if filter_function is None:
            if verbose:
                print('If no filter function is given and the filter does',
                      'not exist, an exception is raised.', flush=True)
            raise NameError('No filter named "{}"'.format(filter_name) +
                            ' created previously and no mask given.')

        # Add the verbose parameter to the filter keyword parameters
        filter_kw_params['verbose'] = verbose
        if verbose:
            print(('Construct the filter from the function {} with' +
                   ' kwyword parameters: {}.').format(filter_function,
                                                      filter_kw_params),
                  flush=True)

        # Make the filter vector
        self._filters[filter_name] = filter_function(self, **filter_kw_params)
        # Store it in the hdf5 file
        try:
            # Get existing, matching dataset or create new one
            dset = filters.require_dataset(filter_name,
                                           self._filters[filter_name].shape,
                                           self._filters[filter_name].dtype)
            # Write the data
            dset[:] = self._filters[filter_name]
        except TypeError:
            # Exception occures if the new filter does not match the on the the
            # file (dtype or shape)
            # if so remove the old datset...
            del filters[filter_name]
            # ...and make a new one.
            dset = filters.create_dataset(filter_name,
                                          data=self._filters[filter_name])

        # Time stamp the dataset in the file
        dset.attrs['time_stamp'] = time.time()

        if verbose:
            print('Return the filter mask.', flush=True)
        return self._filters[filter_name].copy()

    def get_events_filter(self, source, source_filter, logic='any'):
        """Return event maske based on ions or electrons mask."""
        return _helper.events_from_particles(self, source, source_filter,
                                             logic=logic)

    def get_electrons_filter(self, events_filter, verbose=None):
        """Return electrons maske based on events mask."""
        if verbose is None:
            verbose = self._verbose
        return _helper.particles_from_events(self, 'electrons', events_filter,
                                             verbose=verbose)

    def get_ions_filter(self, events_filter, verbose=None):
        """Return ions mask based on events mask."""
        if verbose is None:
            verbose = self._verbose
        return _helper.particles_from_events(self, 'ions', events_filter,
                                             verbose=verbose)

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

    def _get_derived_data_group(self, data_name, filter_sum_string,
                                match_data_dict={}, verbose=False):
        # Get the data mame group
        data_name_group = self._derived_data.require_group(data_name)
        # Get the filter sum group
        filter_sum_group = data_name_group.require_group(filter_sum_string)
        # Iterate through all the dataset_groups
        for dataset_group in filter_sum_group.values():
            # Compare the match_data_dict data
            # first that variables are correct
            dataset_group_set = set(dataset_group)
            if 'data' in dataset_group_set:
                dataset_group_set.remove('data')
            match_data_set = set(match_data_dict)
            if dataset_group_set != match_data_set:
                # if missmatch go to the next iteration of the dataset groups
                continue
            for match_key, match_value in match_data_dict.items():
                if (np.any(dataset_group[match_key].value != match_value)):
                        # A missmatch was found for the current dataset_group
                        # break out oc the match_data comparison
                        break
            else:
                # All match_data was matching: match found!
                # return the current dataset_group
                return dataset_group

        # No match has been found at this point, make the group
        # First find a name that works
        name_counter = 0
        while True:
            dset_group_name = 'dset_' + str(name_counter)
            if dset_group_name not in filter_sum_group:
                break
            name_counter += 1

        # then make the group
        dset_group = filter_sum_group.create_group(dset_group_name)

        # Fill with the match data
        for match_key, match_value in match_data_dict.items():
            dset_group.create_dataset(match_key, data=match_value)

        # Add a dummy time stamp
        dset_group.attrs['time_stamp'] = np.nan

        # return the group
        return dset_group

    def load_derived_data(self, data_name, filter_sum_string,
                          compare_time_stamp=0,
                          verbose=None, match_data_dict={}):
        if verbose is None:
            verbose = self._verbose

        dset_group = self._get_derived_data_group(
            data_name, filter_sum_string,
            match_data_dict=match_data_dict, verbose=verbose)

        if (('data' not in dset_group) or
                not (compare_time_stamp < dset_group.attrs.get('time_stamp'))):
            return (np.empty((0,)), 0.)

        return (dset_group['data'].value, dset_group.attrs['time_stamp'])

    def store_derived_data(self, data, data_name, filter_sum_string,
                           match_data_dict={}, verbose=False):
        dset_group = self._get_derived_data_group(data_name,
                                                  filter_sum_string,
                                                  match_data_dict,
                                                  verbose=verbose)

        try:
            dset = dset_group.require_dataset('data', data.shape, data.dtype)
            dset[:] = data
        except TypeError:
            del dset_group['data']
            dset_group.create_dataset('data', data=data)

        time_stamp = time.time()
        dset_group.attrs['time_stamp'] = time_stamp
        return time_stamp

    def get_i_tof_spectrum(self, t_axis, ions_filter=None, verbose=None):
        """Get ion tof spectrum on given time axis and ions filter"""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print('In get_i_tof_spectrum.', flush=True)

        has_tof = self.get_filter('has_tof_ions',
                                  filter_function=ff.has_tof_ions,
                                  verbose=verbose)
        if ions_filter is None:
            ions_filter = has_tof
        else:
            ions_filter *= has_tof

        if verbose:
            print('Getting ion tof data.', flush=True)
        tof = self.ions.tof_falling_edge[ions_filter]
        if verbose:
            print('Making histogram.', flush=True)
        hist = _helper.center_histogram(tof, t_axis)
        if verbose:
            print('Returning.', flush=True)
        return hist

    def get_i_xy_image(self, x_axis_mm, y_axis_mm=None, ions_filter=None,
                       verbose=None):
        """Get the ion image based on ions_filter."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print('Get the has_position mask.', flush=True)
        has_pos = self.get_filter('has_position_ions',
                                  ff.has_position_particles,
                                  {'particles': 'ions'},
                                  verbose=verbose)
        if self._verbose:
            print('Merge has_pos filter and any given ions_filter.',
                  flush=True)
        if ions_filter is None:
            ions_filter = has_pos
        else:
            ions_filter *= has_pos

        if y_axis_mm is None:
            y_axis_mm = x_axis_mm

        if self._verbose:
            print('Calculate and return image histogram.', flush=True)
        return _helper.center_histogram_2d(self.ions.pos_x[ions_filter],
                                           self.ions.pos_y[ions_filter],
                                           x_axis_mm, y_axis_mm)

    def get_i_rth_image(self,
                        r_axis_mm,
                        th_axis_rad=np.linspace(0, 2*np.pi, 513)[1::2],
                        ions_filter=None,
                        verbose=None):
        """Get the ions image in polar coordinates."""
        if verbose is None:
            verbose = self._verbose
        has_pos = self.get_filter('has_position_ions',
                                  ff.has_position_particles,
                                  {'particles': 'ions'},
                                  verbose=verbose)
        if ions_filter is None:
            ions_filter = has_pos
        else:
            ions_filter *= has_pos

        return _helper.center_histogram_2d(self.ions.pos_r[ions_filter],
                                           self.ions.pos_t[ions_filter],
                                           r_axis_mm, th_axis_rad)

    def get_e_xy_image(self, x_axis_mm, y_axis_mm=None,
                       electrons_filter=None,
                       compare_time_stamp=0,
                       verbose=None):
        """Get the electron image based on electrons_filter."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print('Get the has_position mask.', flush=True)
        has_pos = self.get_filter('has_position_electrons',
                                  ff.has_position_particles,
                                  {'particles': 'electrons'},
                                  verbose=verbose)
        if verbose:
            print('Merge has_pos filter and any given electrons_filter.',
                  flush=True)
        if electrons_filter is None:
            electrons_filter = has_pos
        else:
            electrons_filter *= has_pos

        if y_axis_mm is None:
            y_axis_mm = x_axis_mm

        # Check if the image already exists
        data_name = 'e_xy_image'
        filter_sum_string = str(electrons_filter.sum())
        match_data_dict = {'x_axis_mm': x_axis_mm,
                           'y_axis_mm': y_axis_mm}

        img, img_time_stamp = self.load_derived_data(
            data_name, filter_sum_string, match_data_dict=match_data_dict,
            compare_time_stamp=compare_time_stamp, verbose=verbose)

        if img.size == 0:
            img = _helper.center_histogram_2d(
                self.electrons.pos_x[electrons_filter],
                self.electrons.pos_y[electrons_filter],
                x_axis_mm, y_axis_mm)

            img_time_stamp = self.store_derived_data(
                img, data_name, filter_sum_string, match_data_dict,
                verbose=verbose)

        return img, img_time_stamp

    def get_e_rth_image(self,
                        r_axis_mm,
                        th_axis_rad=np.linspace(0, 2*np.pi, 513)[1::2],
                        electrons_filter=None, compare_time_stamp=0,
                        verbose=None):
        """Get the electrons image in polar coordinates."""
        if verbose is None:
            verbose = self._verbose

        # Get the position filter
        has_pos = self.get_filter('has_position_electrons',
                                  ff.has_position_particles,
                                  {'particles': 'electrons'},
                                  verbose=verbose)
        if electrons_filter is None:
            electrons_filter = has_pos
        else:
            electrons_filter *= has_pos

        # Check if the image already exists
        data_name = 'e_rth_image'
        filter_sum_string = str(electrons_filter.sum())
        match_data_dict = {'r_axis_mm': r_axis_mm,
                           'th_axis_rad': th_axis_rad}

        img, img_time_stamp = self.load_derived_data(
            data_name, filter_sum_string, match_data_dict=match_data_dict,
            compare_time_stamp=compare_time_stamp, verbose=verbose)

        if img.size == 0:
            img = _helper.center_histogram_2d(
                self.electrons.pos_r[electrons_filter],
                self.electrons.pos_t[electrons_filter],
                r_axis_mm, th_axis_rad)

            img_time_stamp = self.store_derived_data(
                img, data_name, filter_sum_string, match_data_dict,
                verbose=verbose)

        return img, img_time_stamp

    def calculate_electron_energy(self, calibration, verbose=False):
        """Add the electron energy to the datset.

        The energy calibration should be given in the calibration object."""

        try:
            old_time_stamp = self.electrons.energy.attrs['time_stamp']
        except:
            old_time_stamp = 0

        if old_time_stamp < max((calibration.conversion_time_stamp,
                                 2439382216)):
            if verbose:
                print('Recalculate electron energies.', flush=True)
            energies, errors, weights = calibration.get_energies(
                self.electrons.pos_r, self.electrons.pos_t)
            self.electrons.add_parameter('energy', energies)
            self.electrons.energy.attrs['time_stamp'] = time.time()
            self.electrons.add_parameter('energy_uncertainty', errors)
            self.electrons.add_parameter('spectral_weight', weights)

        else:
            if verbose:
                print('Energy data up to date.', flush=True)

    def get_e_spectrum(self,
                       e_axis_eV,
                       electrons_filter=None,
                       compare_time_stamp=0,
                       verbose=None):
        """Get the electron spectrum."""
        if verbose is None:
            verbose = self._verbose

        # Get the position filter since a position is needed in order to have
        # an energy
        has_pos = self.get_filter('has_position_electrons',
                                  ff.has_position_particles,
                                  {'particles': 'electrons'},
                                  verbose=verbose)
        if electrons_filter is None:
            electrons_filter = has_pos
        else:
            electrons_filter *= has_pos

        # Check if the image already exists
        data_name = 'e_spectrum'
        filter_sum_string = str(electrons_filter.sum())
        match_data_dict = {'e_axis_eV': e_axis_eV}

        spectrum, spectrum_time_stamp = self.load_derived_data(
            data_name, filter_sum_string,
            match_data_dict=match_data_dict,
            compare_time_stamp=max(
                (compare_time_stamp,
                 self.electrons.energy.attrs['time_stamp'],
                 1445269060)),
            verbose=verbose)

        if spectrum.size == 0:
#            spectrum = _helper.center_histogram(
#                self.electrons.energy[electrons_filter],
#                e_axis_eV,
#                weights=self.electrons.spectral_weight[electrons_filter])
            spectrum = _helper.center_histogram(
                self.electrons.energy[electrons_filter],
                e_axis_eV)

            spectrum_time_stamp = self.store_derived_data(
                spectrum, data_name, filter_sum_string, match_data_dict,
                verbose=verbose)

        return spectrum, spectrum_time_stamp


class DataSetList(object):
    def __init__(self):
        self._dataset_list = []
        self._name_index_dict = {}

    def __len__(self):
        return len(self._dataset_list)

    def __iter__(self):
        for data_set in self._dataset_list:
            yield data_set

    def _check_index_validity(self, name):
        # If the index is an int
        if isinstance(name, int):
            # Check if it is in range
            if (0 <= name) and (name < len(self._dataset_list)):
                return name
            raise IndexError('Index={} out of'.format(name) +
                             ' bounds for lenght {} list'.format(
                len(self._dataset_list)))
        if name in self._name_index_dict:
            return self._name_index_dict[name]
        raise IndexError('The data set list cannot be indexed with' +
                         ' "{}", it is not a valid name.'.format(name))
        return None

    def __getitem__(self, name):
        return self._dataset_list[self._check_index_validity(name)]

    def add_dataset(self, name, h5_path, raw_data_path='',
                    photon_energy=None, electron_center_energy=None,
                    verbose=False,
                    **kwargs):
        if name in self._name_index_dict:
            print(('A data set with the name "{}" already in list.' +
                   ' No action taken.').format(name), flush=True)
            return

        self._dataset_list.append(
            DataSet(name, h5_path, raw_data_path,
                    photon_energy=photon_energy,
                    electron_center_energy=electron_center_energy,
                    verbose=verbose,
                    **kwargs))

        self._name_index_dict[name] = len(self._dataset_list) - 1

    def pop(self, name):
        try:
            index = self._check_index_validity(name)
        except IndexError as e:
            print(e.args)
            print('No action taken by pop.')
            return None
            
        # Remove the dataset from the list
        dataset = self._dataset_list.pop(index)
        # Remove the name from the name index list
        del self._name_index_dict[dataset.name()]
        # Adjust the remaining indexes
        for k, v in self._name_index_dict.items():
            if index < v:
                self._name_index_dict[k] -= 1

        return dataset
        

    def keys(self):
        return [k for k in self._name_index_dict]

    def len(self):
        return len(self._dataset_list)

    def sort(self, reverse=False):
        self._dataset_list.sort(key=(lambda dataset: dataset.name()),
                                reverse=reverse)
        name_list = [dataset.name() for dataset in self._dataset_list]
        for i, name in enumerate(name_list):
            self._name_index_dict[name] = i
