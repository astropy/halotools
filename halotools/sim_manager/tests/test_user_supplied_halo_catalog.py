"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
import warnings
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
from copy import deepcopy

from . import helper_functions

from astropy.table import Table

from .. import UserSuppliedHaloCatalog
from ..user_supplied_ptcl_catalog import UserSuppliedPtclCatalog
from ..halo_table_cache import HaloTableCache

from ...custom_exceptions import HalotoolsError

# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = "/Users/aphearin"
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ("TestUserSuppliedHaloCatalog",)


class TestUserSuppliedHaloCatalog(TestCase):
    """Class providing tests of the `~halotools.sim_manager.UserSuppliedHaloCatalog`."""

    def setUp(self):
        """Pre-load various arrays into memory for use by all tests."""
        self.Nhalos = 100
        self.Lbox = 100
        self.redshift = 0.0
        self.halo_x = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_y = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_z = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_mass = np.logspace(10, 15, self.Nhalos)
        self.halo_id = np.arange(0, self.Nhalos).astype(int)
        self.good_halocat_args = {
            "halo_x": self.halo_x,
            "halo_y": self.halo_y,
            "halo_z": self.halo_z,
            "halo_id": self.halo_id,
            "halo_mass": self.halo_mass,
        }
        self.toy_list = [elt for elt in self.halo_x]

        self.num_ptcl = int(1e4)
        self.good_ptcl_table = Table(
            {
                "x": np.zeros(self.num_ptcl),
                "y": np.zeros(self.num_ptcl),
                "z": np.zeros(self.num_ptcl),
            }
        )

        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

    def test_particle_mass_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(Lbox=200, **self.good_halocat_args)

    def test_lbox_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(
                particle_mass=200, **self.good_halocat_args
            )

    def test_halos_contained_inside_lbox(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(
                Lbox=20, particle_mass=100, **self.good_halocat_args
            )

    def test_redshift_is_float(self):

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift="", **self.good_halocat_args
            )
        substr = "The ``redshift`` metadata must be a float."
        assert substr in err.value.args[0]

    def test_successful_load(self):

        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )
        assert hasattr(halocat, "Lbox")
        assert (halocat.Lbox == 200).all()
        assert hasattr(halocat, "particle_mass")
        assert halocat.particle_mass == 100

    def test_successful_load_vector_Lbox(self):

        halocat = UserSuppliedHaloCatalog(
            Lbox=[100, 200, 300],
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )
        assert hasattr(halocat, "Lbox")
        assert (halocat.Lbox == [100, 200, 300]).all()

    def test_additional_metadata(self):

        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            arnold_schwarzenegger="Stick around!",
            **self.good_halocat_args
        )
        assert hasattr(halocat, "arnold_schwarzenegger")
        assert halocat.arnold_schwarzenegger == "Stick around!"

    def test_all_halo_columns_have_length_nhalos(self):

        # All halo catalog columns must have length-Nhalos
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args["halo_x"][0] = -1
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )

    def test_positions_contained_inside_lbox_alt_test(self):
        # positions must be < Lbox
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args["halo_x"][0] = 10000
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )

    def test_positions_contained_inside_anisotropic_lbox(self):
        # positions must be < Lbox
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args["halo_x"][0] = 125
            halocat = UserSuppliedHaloCatalog(
                Lbox=[100, 150, 200],
                particle_mass=100,
                redshift=self.redshift,
                **bad_halocat_args
            )

        with pytest.raises(HalotoolsError):
            bad_halocat_args["halo_y"][0] = 175
            halocat = UserSuppliedHaloCatalog(
                Lbox=[100, 150, 200],
                particle_mass=100,
                redshift=self.redshift,
                **bad_halocat_args
            )

        with pytest.raises(HalotoolsError):
            bad_halocat_args["halo_z"][0] = 225
            halocat = UserSuppliedHaloCatalog(
                Lbox=[100, 150, 200],
                particle_mass=100,
                redshift=self.redshift,
                **bad_halocat_args
            )

    def test_has_halo_x_column(self):
        # must have halo_x column
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args["halo_x"]
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )

    def test_has_halo_id_column(self):
        # Must have halo_id column
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args["halo_id"]
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )

    def test_has_halo_mass_column(self):
        # Must have some column storing a mass-like variable
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args["halo_mass"]
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )

    def test_halo_prefix_warning(self):
        # Must raise warning if a length-Nhalos array is passed with
        # a keyword argument that does not begin with 'halo_'
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            bad_halocat_args["s"] = np.ones(self.Nhalos)
            halocat = UserSuppliedHaloCatalog(
                Lbox=200, particle_mass=100, redshift=self.redshift, **bad_halocat_args
            )
            assert "interpreted as metadata" in str(w[-1].message)

    def test_ptcl_table(self):
        """Method performs various existence and consistency tests on the input ptcl_table.

        * Enforce that instances do *not* have ``ptcl_table`` attributes if none is passed.

        * Enforce that instances *do* have ``ptcl_table`` attributes if a legitimate one is passed.

        * Enforce that ptcl_table have ``x``, ``y`` and ``z`` columns.

        * Enforce that ptcl_table input is an Astropy `~astropy.table.Table` object, not a Numpy recarray
        """
        pass

    def test_ptcl_table_dne(self):
        # Must not have a ptcl_table attribute when none is passed
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )
        assert not hasattr(halocat, "ptcl_table")

    def test_ptcl_table_exists_when_given_goodargs(self):

        # Must have ptcl_table attribute when argument is legitimate
        ptclcat = UserSuppliedPtclCatalog(
            x=np.zeros(self.num_ptcl),
            y=np.zeros(self.num_ptcl),
            z=np.zeros(self.num_ptcl),
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
        )

        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            user_supplied_ptclcat=ptclcat,
            **self.good_halocat_args
        )
        assert hasattr(halocat, "ptcl_table")

    def test_ptcl_table_bad_args1(self):

        # Must have ptcl_table attribute when argument is legitimate
        ptclcat = UserSuppliedPtclCatalog(
            x=np.zeros(self.num_ptcl),
            y=np.zeros(self.num_ptcl),
            z=np.zeros(self.num_ptcl),
            Lbox=100,
            particle_mass=100,
            redshift=self.redshift,
        )

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedHaloCatalog(
                Lbox=200,
                particle_mass=100,
                redshift=self.redshift,
                user_supplied_ptclcat=ptclcat,
                **self.good_halocat_args
            )
        substr = "Inconsistent values of Lbox"
        assert substr in err.value.args[0]

    def test_ptcl_table_bad_args2(self):

        # Must have ptcl_table attribute when argument is legitimate
        ptclcat = UserSuppliedPtclCatalog(
            x=np.zeros(self.num_ptcl),
            y=np.zeros(self.num_ptcl),
            z=np.zeros(self.num_ptcl),
            Lbox=200,
            particle_mass=200,
            redshift=self.redshift,
        )

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedHaloCatalog(
                Lbox=200,
                particle_mass=100,
                redshift=self.redshift,
                user_supplied_ptclcat=ptclcat,
                **self.good_halocat_args
            )
        substr = "Inconsistent values of particle_mass"
        assert substr in err.value.args[0]

    def test_ptcl_table_bad_args3(self):

        # Must have ptcl_table attribute when argument is legitimate
        ptclcat = UserSuppliedPtclCatalog(
            x=np.zeros(self.num_ptcl),
            y=np.zeros(self.num_ptcl),
            z=np.zeros(self.num_ptcl),
            Lbox=200,
            particle_mass=200,
            redshift=self.redshift + 0.1,
        )

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedHaloCatalog(
                Lbox=200,
                particle_mass=200,
                redshift=self.redshift,
                user_supplied_ptclcat=ptclcat,
                **self.good_halocat_args
            )
        substr = "Inconsistent values of redshift"
        assert substr in err.value.args[0]

    def test_ptcl_table_bad_args4(self):

        with pytest.raises(HalotoolsError) as err:
            halocat = UserSuppliedHaloCatalog(
                Lbox=200,
                particle_mass=200,
                redshift=self.redshift,
                user_supplied_ptclcat=98,
                **self.good_halocat_args
            )
        substr = "an instance of UserSuppliedPtclCatalog"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache1(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc"
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")
        assert os.path.isfile(fname)

        dummy_string = "  "
        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                fname, dummy_string, dummy_string, dummy_string, dummy_string
            )
        substr = "Either choose a different fname or set ``overwrite`` to True"
        assert substr in err.value.args[0]

        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                fname,
                dummy_string,
                dummy_string,
                dummy_string,
                dummy_string,
                overwrite=True,
            )
        assert substr not in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache2(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc"

        dummy_string = "  "
        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                basename, dummy_string, dummy_string, dummy_string, dummy_string
            )
        substr = "The directory you are trying to store the file does not exist."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache3(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc"
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")
        assert os.path.isfile(fname)

        dummy_string = "  "
        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                fname,
                dummy_string,
                dummy_string,
                dummy_string,
                dummy_string,
                overwrite=True,
            )
        substr = "The fname must end with an ``.hdf5`` extension."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache4(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc.hdf5"
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")
        assert os.path.isfile(fname)

        dummy_string = "  "

        class Dummy(object):
            pass

            def __str__(self):
                raise TypeError

        not_representable_as_string = Dummy()

        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                fname,
                not_representable_as_string,
                dummy_string,
                dummy_string,
                dummy_string,
                overwrite=True,
            )
        substr = "must all be strings."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache5(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc.hdf5"
        fname = os.path.join(self.dummy_cache_baseloc, basename)
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")
        assert os.path.isfile(fname)

        dummy_string = "  "

        class Dummy(object):
            pass

            def __str__(self):
                raise TypeError

        not_representable_as_string = Dummy()

        with pytest.raises(HalotoolsError) as err:
            halocat.add_halocat_to_cache(
                fname,
                dummy_string,
                dummy_string,
                dummy_string,
                dummy_string,
                overwrite=True,
                some_more_metadata=not_representable_as_string,
            )
        substr = "keyword is not representable as a string."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not HAS_H5PY")
    def test_add_halocat_to_cache6(self):
        halocat = UserSuppliedHaloCatalog(
            Lbox=200,
            particle_mass=100,
            redshift=self.redshift,
            **self.good_halocat_args
        )

        basename = "abc.hdf5"
        fname = os.path.join(self.dummy_cache_baseloc, basename)

        simname = "dummy_simname"
        halo_finder = "dummy_halo_finder"
        version_name = "dummy_version_name"
        processing_notes = "dummy processing notes"

        halocat.add_halocat_to_cache(
            fname,
            simname,
            halo_finder,
            version_name,
            processing_notes,
            overwrite=True,
            some_additional_metadata=processing_notes,
        )

        cache = HaloTableCache()
        assert halocat.log_entry in cache.log

        cache.remove_entry_from_cache_log(
            halocat.log_entry.simname,
            halocat.log_entry.halo_finder,
            halocat.log_entry.version_name,
            halocat.log_entry.redshift,
            halocat.log_entry.fname,
            raise_non_existence_exception=True,
            update_ascii=True,
            delete_corresponding_halo_catalog=True,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass


def test_support_for_empty_halo_catalogs():
    """Regression test for #960."""
    Nhalos = 0
    Lbox = 100
    redshift = 0.0
    halo_x = np.linspace(0, Lbox, Nhalos)
    halo_y = np.linspace(0, Lbox, Nhalos)
    halo_z = np.linspace(0, Lbox, Nhalos)
    halo_mass = np.logspace(10, 15, Nhalos)
    halo_id = np.arange(0, Nhalos).astype(int)
    good_halocat_args = {
        "halo_x": halo_x,
        "halo_y": halo_y,
        "halo_z": halo_z,
        "halo_id": halo_id,
        "halo_mass": halo_mass,
    }
    halocat = UserSuppliedHaloCatalog(
        Lbox=Lbox, particle_mass=100, redshift=redshift, **good_halocat_args
    )
    assert halocat.halo_table["halo_x"].shape == (0,)
