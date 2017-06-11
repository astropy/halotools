"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
import os
import shutil

from astropy.config.paths import _find_home
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import numpy as np
from copy import copy, deepcopy

from . import helper_functions

from astropy.table import Table

from ..user_supplied_ptcl_catalog import UserSuppliedPtclCatalog
from ..ptcl_table_cache import PtclTableCache

from ...custom_exceptions import HalotoolsError

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

__all__ = ('TestUserSuppliedPtclCatalog', )


class TestUserSuppliedPtclCatalog(TestCase):
    """ Class providing tests of the `~halotools.sim_manager.UserSuppliedPtclCatalog`.
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        self.Nptcls = int(1e4)
        self.Lbox = 100
        self.redshift = 0.0
        self.x = np.linspace(0, self.Lbox, self.Nptcls)
        self.y = np.linspace(0, self.Lbox, self.Nptcls)
        self.z = np.linspace(0, self.Lbox, self.Nptcls)

        self.good_ptclcat_args = (
            {'x': self.x, 'y': self.y,
            'z': self.z}
            )

        self.good_ptcl_table = Table(self.good_ptclcat_args)

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc

        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

    def test_particle_mass_requirement(self):

        with pytest.raises(HalotoolsError):
            ptclcat = UserSuppliedPtclCatalog(Lbox=200,
                **self.good_ptclcat_args)

    def test_lbox_requirement(self):

        with pytest.raises(HalotoolsError):
            ptclcat = UserSuppliedPtclCatalog(particle_mass=200,
                **self.good_ptclcat_args)

    def test_ptcls_contained_inside_lbox(self):

        with pytest.raises(HalotoolsError):
            ptclcat = UserSuppliedPtclCatalog(Lbox=20, particle_mass=100,
                **self.good_ptclcat_args)

    def test_redshift_is_float(self):

        with pytest.raises(HalotoolsError) as err:
            ptclcat = UserSuppliedPtclCatalog(
                Lbox=200, particle_mass=100, redshift='',
                **self.good_ptclcat_args)
        substr = "The ``redshift`` metadata must be a float."
        assert substr in err.value.args[0]

    def test_successful_load(self):

        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)
        assert hasattr(ptclcat, 'Lbox')
        assert (ptclcat.Lbox == 200.).all()
        assert hasattr(ptclcat, 'particle_mass')
        assert ptclcat.particle_mass == 100

    def test_successful_load_vector_Lbox(self):

        ptclcat = UserSuppliedPtclCatalog(Lbox=[100,200,300],
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)
        assert hasattr(ptclcat, 'Lbox')
        assert (ptclcat.Lbox == [100,200,300]).all()

    def test_additional_metadata(self):

        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            arnold_schwarzenegger='Stick around!',
            **self.good_ptclcat_args)
        assert hasattr(ptclcat, 'arnold_schwarzenegger')
        assert ptclcat.arnold_schwarzenegger == 'Stick around!'

    def test_all_halo_columns_have_length_nhalos(self):

        # All halo catalog columns must have length-Nhalos
        bad_ptclcat_args = deepcopy(self.good_ptclcat_args)
        with pytest.raises(HalotoolsError):
            bad_ptclcat_args['x'][0] = -1
            ptclcat = UserSuppliedPtclCatalog(Lbox=200,
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

    def test_positions_contained_inside_lbox_alt_test(self):
        # positions must be < Lbox
        bad_ptclcat_args = deepcopy(self.good_ptclcat_args)
        with pytest.raises(HalotoolsError):
            bad_ptclcat_args['x'][0] = 10000
            ptclcat = UserSuppliedPtclCatalog(Lbox=200,
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

    def test_positions_contained_inside_anisotropic_lbox(self):
        # positions must be < Lbox
        bad_ptclcat_args = deepcopy(self.good_ptclcat_args)
        with pytest.raises(HalotoolsError):
            bad_ptclcat_args['x'][0] = 125
            ptclcat = UserSuppliedPtclCatalog(Lbox=[100, 150, 200],
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

        with pytest.raises(HalotoolsError):
            bad_ptclcat_args['y'][0] = 175
            ptclcat = UserSuppliedPtclCatalog(Lbox=[100, 150, 200],
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

        with pytest.raises(HalotoolsError):
            bad_ptclcat_args['z'][0] = 225
            ptclcat = UserSuppliedPtclCatalog(Lbox=[100, 150, 200],
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

    def test_has_x_column(self):
        # must have x column
        bad_ptclcat_args = deepcopy(self.good_ptclcat_args)
        with pytest.raises(HalotoolsError):
            del bad_ptclcat_args['x']
            ptclcat = UserSuppliedPtclCatalog(Lbox=200,
                particle_mass=100, redshift=self.redshift,
                **bad_ptclcat_args)

    @pytest.mark.skipif('not HAS_H5PY')
    def test_add_ptclcat_to_cache1(self):
        """ Verify the overwrite requirement is enforced
        """
        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)

        basename = 'abc'
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({'x': [0]})
        _t.write(fname, format='ascii')
        assert os.path.isfile(fname)

        dummy_string = '  '
        with pytest.raises(HalotoolsError) as err:
            ptclcat.add_ptclcat_to_cache(
                fname, dummy_string, dummy_string, dummy_string)
        substr = "Either choose a different fname or set ``overwrite`` to True"
        assert substr in err.value.args[0]

        with pytest.raises(HalotoolsError) as err:
            ptclcat.add_ptclcat_to_cache(
                fname, dummy_string, dummy_string, dummy_string,
                overwrite=True)
        assert substr not in err.value.args[0]

    @pytest.mark.skipif('not HAS_H5PY')
    def test_add_ptclcat_to_cache2(self):
        """ Verify that the appropriate message is issued when trying to save the file to a non-existent directory.
        """
        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)

        basename = 'abc'

        dummy_string = '  '
        with pytest.raises(HalotoolsError) as err:
            ptclcat.add_ptclcat_to_cache(
                basename, dummy_string, dummy_string, dummy_string, dummy_string)
        substr = "The directory you are trying to store the file does not exist."
        assert substr in err.value.args[0]

    @pytest.mark.skipif('not HAS_H5PY')
    def test_add_ptclcat_to_cache3(self):
        """ Verify that the .hdf5 extension requirement is enforced.
        """
        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)

        basename = 'abc'
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({'x': [0]})
        _t.write(fname, format='ascii')
        assert os.path.isfile(fname)

        dummy_string = '  '
        with pytest.raises(HalotoolsError) as err:
            ptclcat.add_ptclcat_to_cache(
                fname, dummy_string, dummy_string, dummy_string,
                overwrite=True)
        substr = "The fname must end with an ``.hdf5`` extension."
        assert substr in err.value.args[0]

    @pytest.mark.skipif('not HAS_H5PY')
    def test_add_ptclcat_to_cache4(self):
        """ Enforce string representation of positional arguments
        """
        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)

        basename = 'abc.hdf5'
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({'x': [0]})
        _t.write(fname, format='ascii')
        assert os.path.isfile(fname)

        dummy_string = '  '

        class Dummy(object):
            pass

            def __str__(self):
                raise TypeError
        not_representable_as_string = Dummy()

        with pytest.raises(HalotoolsError) as err:
            ptclcat.add_ptclcat_to_cache(
                fname, not_representable_as_string, dummy_string, dummy_string,
                overwrite=True)
        substr = "must all be strings."
        assert substr in err.value.args[0]

    @pytest.mark.skipif('not HAS_H5PY')
    def test_add_ptclcat_to_cache6(self):
        ptclcat = UserSuppliedPtclCatalog(Lbox=200,
            particle_mass=100, redshift=self.redshift,
            **self.good_ptclcat_args)

        basename = 'abc.hdf5'
        fname = os.path.join(self.dummy_cache_baseloc, basename)

        simname = 'dummy_simname'
        version_name = 'dummy_version_name'
        processing_notes = 'dummy processing notes'

        assert 'x' in list(ptclcat.ptcl_table.keys())
        assert 'y' in list(ptclcat.ptcl_table.keys())
        assert 'z' in list(ptclcat.ptcl_table.keys())

        ptclcat.add_ptclcat_to_cache(
            fname, simname, version_name, processing_notes, overwrite=True)

        cache = PtclTableCache()
        assert ptclcat.log_entry in cache.log

        cache.remove_entry_from_cache_log(
            ptclcat.log_entry.simname,
            ptclcat.log_entry.version_name,
            ptclcat.log_entry.redshift,
            ptclcat.log_entry.fname,
            raise_non_existence_exception=True,
            update_ascii=True,
            delete_corresponding_ptcl_catalog=True)

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
