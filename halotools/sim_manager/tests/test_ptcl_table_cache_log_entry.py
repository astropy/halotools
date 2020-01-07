"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
import pytest
import os
import shutil

from astropy.table import Table

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from astropy.config.paths import _find_home

from . import helper_functions
from ..ptcl_table_cache_log_entry import PtclTableCacheLogEntry

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

__all__ = ('TestPtclTableCacheLogEntry',)


class TestPtclTableCacheLogEntry(TestCase):
    """ Class providing unit testing for `~halotools.sim_manager.PtclTableCacheLogEntry`.
    """
    hard_coded_log_attrs = ['simname', 'version_name', 'redshift', 'fname']

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

        self.simnames = ('bolshoi', 'consuelo',
            'bolshoi', 'bolshoi', 'multidark')
        self.version_names = ('v0', 'v1', 'v2', 'v3', 'v4')
        self.redshifts = (1.2, -0.1, 1.339, 1.3, 100.)

        self.basenames = ('non_existent.hdf5', 'existent.file',
            'existent.hdf5', 'existent.hdf5', 'good.hdf5')
        self.fnames = tuple(os.path.join(self.dummy_cache_baseloc, name)
            for name in self.basenames)

        self.table1 = Table({'x': [1, 2, 3]})

        self.good_table = Table(
            {'ptcl_id': [1, 2, 3],
            'x': [1, 2, 3],
            'y': [1, 2, 3],
            'z': [1, 2, 3],
            'vx': [1, 2, 3],
            'vy': [1, 2, 3],
            'vz': [1, 2, 3],
             })

    def get_scenario_kwargs(self, num_scenario):
        return ({'simname': self.simnames[num_scenario],
            'version_name': self.version_names[num_scenario],
            'redshift': self.redshifts[num_scenario],
            'fname': self.fnames[num_scenario]})

    @pytest.mark.skipif('not HAS_H5PY')
    def test_instantiation(self):
        """ We can instantiate the log entry with a complete set of metadata
        """

        for i in range(len(self.simnames)):
            constructor_kwargs = self.get_scenario_kwargs(i)
            log_entry = PtclTableCacheLogEntry(**constructor_kwargs)
            assert set(log_entry.log_attributes) == set(self.hard_coded_log_attrs)

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario0(self):
        num_scenario = 0
        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "The input filename does not exist." in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario1(self):
        num_scenario = 1

        with open(self.fnames[num_scenario], 'w') as f:
            f.write('abc')
        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
        assert log_entry.safe_for_cache is False
        assert "The input filename does not exist." not in log_entry._cache_safety_message
        assert "The input file must have '.hdf5' extension" in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario2(self):
        num_scenario = 2

        with open(self.fnames[num_scenario], 'w') as f:
            f.write('abc')
        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
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

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))
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

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

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
        except OSError:
            pass
        self.table1.write(self.fnames[num_scenario], path='data')

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

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

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

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
        substr = "must at a minimum have the following columns"
        assert substr in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_scenario4(self):
        num_scenario = 4

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass
        self.good_table.write(self.fnames[num_scenario], path='data')

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 2.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is False
        substr = "must be bounded by [0, Lbox]."
        assert substr in log_entry._cache_safety_message

    @pytest.mark.skipif('not HAS_H5PY')
    def test_passing_scenario(self):
        num_scenario = 4

        try:
            os.remove(self.fnames[num_scenario])
        except:
            pass
        self.good_table.write(self.fnames[num_scenario], path='data')

        log_entry = PtclTableCacheLogEntry(**self.get_scenario_kwargs(num_scenario))

        f = h5py.File(self.fnames[num_scenario], 'a')
        for attr in self.hard_coded_log_attrs:
            f.attrs[attr] = getattr(log_entry, attr)
        f.attrs['Lbox'] = 100.
        f.attrs['particle_mass'] = 1.e8
        f.close()

        assert log_entry.safe_for_cache is True, log_entry._cache_safety_message
        substr = "The particle catalog is safe to add to the cache log."
        assert substr in log_entry._cache_safety_message

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
