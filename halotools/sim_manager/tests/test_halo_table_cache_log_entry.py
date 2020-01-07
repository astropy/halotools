"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
import pytest
import os
import shutil

import numpy as np
from copy import deepcopy

from astropy.table import Table

from astropy.config.paths import _find_home

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from . import helper_functions
from ..halo_table_cache_log_entry import HaloTableCacheLogEntry

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestHaloTableCacheLogEntry')


class TestHaloTableCacheLogEntry(TestCase):
    """ Class providing unit testing for `~halotools.sim_manager.HaloTableCacheLogEntry`.
    """

    hard_coded_log_attrs = ['simname', 'halo_finder', 'version_name', 'redshift', 'fname']

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests.
        """

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

        self.simnames = ('bolshoi', 'consuelo', 'bolshoi', 'bolshoi', 'multidark')
        self.halo_finders = ('rockstar', 'bdm', 'bdm', 'rockstar', 'bdm')
        self.version_names = ('v0', 'v1', 'v2', 'v3', 'v4')
        self.redshifts = (1.2, -0.1, 1.339, 1.3, 100.)

        self.basenames = ('non_existent.hdf5', 'existent.file',
            'existent.hdf5', 'existent.hdf5', 'good.hdf5')
        self.fnames = tuple(os.path.join(self.dummy_cache_baseloc, name)
            for name in self.basenames)

        self.table1 = Table({'x': [1, 2, 3]})
        self.table2 = Table({'halo_x': [1, 2, 3]})

        self.table3 = Table(
            {'halo_id': [1, 2, 3],
            'halo_x': [-1, 2, 3],
            'halo_y': [1, 2, 3],
            'halo_z': [1, 2, 3],
            'halo_mass': [1, 2, 3],
             })

        self.table4 = Table(
            {'halo_id': [1, 2, 2],
            'halo_x': [1, 2, 3],
            'halo_y': [1, 2, 3],
            'halo_z': [1, 2, 3],
            'halo_mass': [1, 2, 3],
             })

        self.good_table = Table(
            {'halo_id': [1, 2, 3],
            'halo_x': [1, 2, 3],
            'halo_y': [1, 2, 3],
            'halo_z': [1, 2, 3],
            'halo_mass': [1, 2, 3],
             })

    def get_scenario_kwargs(self, num_scenario):
        return ({'simname': self.simnames[num_scenario], 'halo_finder': self.halo_finders[num_scenario],
            'version_name': self.version_names[num_scenario], 'redshift': self.redshifts[num_scenario],
            'fname': self.fnames[num_scenario]})

    @pytest.mark.skipif('not HAS_H5PY')
    def test_instantiation(self):
        """ We can instantiate the log entry with a complete set of metadata
        """

        for i in range(len(self.simnames)):
            constructor_kwargs = self.get_scenario_kwargs(i)
            log_entry = HaloTableCacheLogEntry(**constructor_kwargs)
            assert set(log_entry.log_attributes) == set(self.hard_coded_log_attrs)

    @pytest.mark.skipif('not HAS_H5PY')
    def test_comparison_override1(self):
        constructor_kwargs = self.get_scenario_kwargs(1)
        log_entry1 = HaloTableCacheLogEntry(**constructor_kwargs)
        constructor_kwargs = self.get_scenario_kwargs(2)
        log_entry2 = HaloTableCacheLogEntry(**constructor_kwargs)

        assert log_entry1 != log_entry2
        assert log_entry1 != 7

    @pytest.mark.skipif('not HAS_H5PY')
    def test_comparison_override2(self):
        constructor_kwargs = self.get_scenario_kwargs(1)
        log_entry1 = HaloTableCacheLogEntry(**constructor_kwargs)
        constructor_kwargs = self.get_scenario_kwargs(2)
        log_entry2 = HaloTableCacheLogEntry(**constructor_kwargs)

        assert log_entry1 > log_entry2
        assert log_entry1 >= log_entry2
        with pytest.raises(TypeError) as err:
            _ = log_entry1 > 0
        substr = "You cannot compare the order"
        assert substr in err.value.args[0]

    @pytest.mark.skipif('not HAS_H5PY')
    def test_comparison_override3(self):
        constructor_kwargs = self.get_scenario_kwargs(1)
        log_entry1 = HaloTableCacheLogEntry(**constructor_kwargs)
        _ = hash(log_entry1)

    @pytest.mark.skipif('not HAS_H5PY')
    def test_comparison_override4(self):
        num_scenario = 1
        constructor_kwargs = self.get_scenario_kwargs(num_scenario)
        log_entry1 = HaloTableCacheLogEntry(**constructor_kwargs)
        msg = str(log_entry1)
        fname = self.fnames[num_scenario]
        assert str(msg) == "('consuelo', 'bdm', 'v1', '-0.1000', '"+fname+"')"

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario0(self):
        num_scenario = 0
        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "The input filename does not exist." in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario1(self):
        num_scenario = 1

        with open(self.fnames[num_scenario], 'w') as f:
            f.write('abc')
        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "The input filename does not exist." not in log_entry._cache_safety_message
        assert "The input file must have '.hdf5' extension" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario2(self):
        num_scenario = 2

        with open(self.fnames[num_scenario], 'w') as f:
            f.write('abc')
        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "The input filename does not exist." not in log_entry._cache_safety_message
        assert "The input file must have '.hdf5' extension" not in log_entry._cache_safety_message
        assert "access the hdf5 metadata raised an exception." in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario2a(self):
        num_scenario = 2

        try:
            os.remove(self.fnames[num_scenario])
        except OSError:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        f = h5py.File(self.fnames[num_scenario], 'r')
        k = list(f.attrs.keys())
        f.close()

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "access the hdf5 metadata raised an exception." not in log_entry._cache_safety_message
        assert "missing the following metadata" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario2b(self):
        num_scenario = 2

        try:
            os.remove(self.fnames[num_scenario])
        except OSError:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.close()

        assert log_entry.safe_for_cache is False
        assert "``particle_mass``" in log_entry._cache_safety_message

        f = h5py.File(self.fnames[num_scenario], 'a')
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()
        _ = log_entry.safe_for_cache
        assert "``particle_mass``" not in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario2c(self):
        num_scenario = 2

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            if attr != 'redshift':
                f.attrs[attr] = getattr(log_entry, attr)
            else:
                f.attrs[attr] = 0.4
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "does not match" in log_entry._cache_safety_message

        f = h5py.File(self.fnames[num_scenario], 'a')
        f.attrs['redshift'] = 1.3390001
        f.close()
        assert log_entry.safe_for_cache is False
        assert "does not match" not in log_entry._cache_safety_message

        f = h5py.File(self.fnames[num_scenario], 'a')
        f.attrs['redshift'] = '1.3390001'
        f.close()
        assert log_entry.safe_for_cache is False
        assert "does not match" not in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario3(self):
        num_scenario = 3

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must begin with the following five characters" in log_entry._cache_safety_message

        self.table2.write(self.fnames[num_scenario], path='data', overwrite=True)
        assert log_entry.safe_for_cache is False
        assert "must begin with the following five characters" not in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario3b(self):
        num_scenario = 3

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must begin with the following five characters:" in log_entry._cache_safety_message
        assert "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``" in log_entry._cache_safety_message

        self.table2.write(self.fnames[num_scenario], path='data', overwrite=True)
        assert log_entry.safe_for_cache is False
        assert "must begin with the following five characters:" not in log_entry._cache_safety_message
        assert "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario3c(self):
        num_scenario = 3

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        self.table3.write(self.fnames[num_scenario], path='data')
        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must begin with the following five characters:" not in log_entry._cache_safety_message
        assert "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``" not in log_entry._cache_safety_message
        assert "must be bounded by [0, Lbox]" in log_entry._cache_safety_message
        assert "must contain a unique set of integers" not in log_entry._cache_safety_message

        self.table4.write(self.fnames[num_scenario], path='data', overwrite=True)
        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must be bounded by [0, Lbox]" not in log_entry._cache_safety_message
        assert "must contain a unique set of integers" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario4a(self):
        num_scenario = 4

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        bad_table = deepcopy(self.good_table)
        del bad_table['halo_id']
        bad_table['halo_id'] = np.arange(len(bad_table), dtype=float)
        bad_table['halo_id'][0] = 0.1
        bad_table.write(self.fnames[num_scenario], path='data')
        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must contain a unique set of integers" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario4b(self):
        num_scenario = 4

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        bad_table = deepcopy(self.good_table)
        bad_table['halo_rvir'] = 0.
        bad_table['halo_rvir'][0] = 51
        bad_table.write(self.fnames[num_scenario], path='data')
        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        assert "must be less than 50" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_passing_scenario(self):
        num_scenario = 4

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass

        log_entry = HaloTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        self.good_table.write(self.fnames[num_scenario], path='data')
        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is True
        assert "The halo catalog is safe to add to the cache log." == log_entry._cache_safety_message

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
