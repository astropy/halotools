"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
import os
import shutil

from astropy.config.paths import _find_home
from astropy.tests.helper import pytest
from astropy.table import Table

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import numpy as np

from . import helper_functions

from ..cached_halo_catalog import CachedHaloCatalog
from ..halo_table_cache import HaloTableCache
from ..ptcl_table_cache import PtclTableCache
from ..download_manager import DownloadManager
from ...utils.python_string_comparisons import (
    compare_strings_py23_safe,
    _passively_decode_string,
)
from ...custom_exceptions import HalotoolsError, InvalidCacheLogEntry

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

__all__ = ("TestCachedHaloCatalog",)


class TestCachedHaloCatalog(TestCase):
    """ """

    def setUp(self):
        """Pre-load various arrays into memory for use by all tests."""
        self.dummy_cache_baseloc = helper_functions.dummy_cache_baseloc
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
        os.makedirs(self.dummy_cache_baseloc)

    def test_raises_bad_constructor_args_exception1(self):
        with pytest.raises(HalotoolsError) as err:
            _ = CachedHaloCatalog("bolshoi")
        substr = "CachedHaloCatalog only accepts keyword arguments,"
        assert substr in err.value.args[0]

    def test_raises_bad_constructor_args_exception2(self):
        with pytest.raises(HalotoolsError) as err:
            _ = CachedHaloCatalog(z=0.04)
        substr = "CachedHaloCatalog got an unexpected keyword"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_all_catalogs(self):
        """Verify that all halo catalogs in cache successfully load.
        This test is only run on APH_MACHINE because there is no need to enforce
        users to have a clean cache log if they do not want to bother cleaning it up.
        """
        cache = HaloTableCache()
        for entry in cache.log:
            constructor_kwargs = {
                attr: getattr(entry, attr) for attr in entry.log_attributes
            }
            del constructor_kwargs["fname"]
            halocat = CachedHaloCatalog(**constructor_kwargs)
            assert hasattr(halocat, "redshift")
            assert hasattr(halocat, "Lbox")
            assert hasattr(halocat, "num_ptcl_per_dim")
            assert hasattr(halocat, "cosmology")

    @pytest.mark.skipif("not HAS_H5PY")
    def test_halo_ptcl_consistency(self):
        """ """
        type_mismatch_msg = (
            "\nThe redshift attribute of your particle catalog\n"
            "is formatted as a float, not a string, \nwhich conflicts with the "
            "formatting of the redshift attribute \nof the corresponding halo catalog.\n"
            "This is due to a now-fixed bug in the production of the \n"
            "Halotools-provided particle catalogs. \n"
            "To resolve this, just run the scripts/download_additional_halocat.py script \n"
            "and throw the -ptcls_only and -overwrite flags"
        )

        cache = HaloTableCache()
        for entry in cache.log:
            constructor_kwargs = {
                attr: getattr(entry, attr) for attr in entry.log_attributes
            }
            del constructor_kwargs["fname"]
            halocat = CachedHaloCatalog(**constructor_kwargs)
            halo_log_entry = halocat.log_entry
            try:
                ptcl_log_entry = halocat._retrieve_matching_ptcl_cache_log_entry()
                assert halo_log_entry.simname == ptcl_log_entry.simname
                assert halo_log_entry.redshift == ptcl_log_entry.redshift

                hf = h5py.File(halo_log_entry.fname, "r")
                pf = h5py.File(ptcl_log_entry.fname, "r")

                assert hf.attrs["simname"] == pf.attrs["simname"]

                try:
                    assert type(hf.attrs["redshift"]) == type(pf.attrs["redshift"])
                except AssertionError:
                    msg = (
                        "Type error for the redshift attribute of the ``"
                        + hf.attrs["simname"]
                        + "`` simulation.\n"
                    )
                    msg += type_mismatch_msg
                    raise HalotoolsError(msg)

                hf.close()
                pf.close()

            except (HalotoolsError, InvalidCacheLogEntry):
                fname = halo_log_entry.fname
                simname = halo_log_entry.simname
                redshift = float(halo_log_entry.redshift)
                if APH_MACHINE:
                    allowed_failure = (redshift > 0) & (
                        simname == "bolshoi" or simname == "multidark"
                    )
                    if allowed_failure:
                        pass
                    else:
                        msg = (
                            "APH_MACHINE should never have inconsistent halo/ptcl tables\n"
                            "fname = {0}\nsimname = {1}\nredshift = {2}"
                        )
                        raise HalotoolsError(msg.format(fname, simname, redshift))
                else:
                    pass

    def test_default_catalog(self):
        """Verify that the default halo catalog loads if it is available"""
        try:
            halocat = CachedHaloCatalog()
        except:
            if APH_MACHINE:
                raise ValueError(
                    "This test should pass on APH_MACHINE since \n"
                    "this machine should have the requested catalog"
                )
            else:
                return
        assert hasattr(halocat, "redshift")
        assert hasattr(halocat, "Lbox")

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog1(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi",
                halo_finder="bdm",
                version_name="halotools_alpha_version2",
                redshift=5,
                dz_tol=1,
            )
        assert "The following entries in the cache log" in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog2(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi",
                halo_finder="bdm",
                version_name="halotools_alpha_version2",
                redshift=5,
                dz_tol=1,
            )
        assert "The following entries in the cache log" in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog3(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi", halo_finder="bdm", version_name="Jose Canseco"
            )
        assert "The following entries in the cache log" in err.value.args[0]
        assert "(set by sim_defaults.default_redshift)" in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog4(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname="bolshoi", halo_finder="Jose Canseco")
        assert "The following entries in the cache log" in err.value.args[0]
        assert "(set by sim_defaults.default_version_name)" in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog5(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(simname="Jose Canseco")
        assert (
            "There are no simulations matching your input simname" in err.value.args[0]
        )
        assert "(set by sim_defaults.default_halo_finder)" in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog6(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="Jose Canseco",
                halo_finder="bdm",
                version_name="halotools_alpha_version2",
                redshift=5,
                dz_tol=1,
            )
        assert (
            "There are no simulations matching your input simname" in err.value.args[0]
        )

    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_bad_catalog7(self):
        """Verify that the appropriate errors are raised when
        attempting to load catalogs without matches in cache.
        """
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(dz_tol=100)
        assert "There are multiple entries in the cache log" in err.value.args[0]
        assert "(set by sim_defaults.default_simname)" in err.value.args[0]

    @pytest.mark.slow
    @pytest.mark.skipif("not APH_MACHINE")
    def test_load_ptcl_table(self):
        """Verify that the default particle catalog loads."""
        halocat = CachedHaloCatalog()
        ptcls = halocat.ptcl_table

    @pytest.mark.skipif("not APH_MACHINE")
    def test_fname_optional_load(self):
        fname = "/Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolplanck/rockstar/hlist_0.33406.list.halotools_v0p4.hdf5"
        halocat = CachedHaloCatalog(fname=fname)
        assert compare_strings_py23_safe(halocat.simname, "bolplanck")

    @pytest.mark.slow
    @pytest.mark.skipif("not APH_MACHINE")
    def test_all_fname_loads(self):
        cache = HaloTableCache()
        for entry in cache.log:
            fname = entry.fname
            halocat = CachedHaloCatalog(fname=fname)
            for attr in entry.log_attributes:
                if attr == "redshift":
                    assert float(getattr(entry, attr)) == float(getattr(halocat, attr))
                else:
                    assert _passively_decode_string(
                        getattr(entry, attr)
                    ) == _passively_decode_string(getattr(halocat, attr))

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments1(self):
        fname = os.path.join(self.dummy_cache_baseloc, "abc.hdf5")
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname=fname, simname="bolshoi")
        substr = "If you specify an input ``fname``"
        assert substr in err.value.args[0]
        substr = "do not also specify ``simname``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments2(self):
        fname = os.path.join(self.dummy_cache_baseloc, "abc.hdf5")
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname=fname, version_name="dummy")
        substr = "If you specify an input ``fname``"
        assert substr in err.value.args[0]
        substr = "do not also specify ``version_name``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments3(self):
        fname = os.path.join(self.dummy_cache_baseloc, "abc.hdf5")
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname=fname, halo_finder="dummy")
        substr = "If you specify an input ``fname``"
        assert substr in err.value.args[0]
        substr = "do not also specify ``halo_finder``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments4(self):
        fname = os.path.join(self.dummy_cache_baseloc, "abc.hdf5")
        _t = Table({"x": [0]})
        _t.write(fname, format="ascii")

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname=fname, redshift=0)
        substr = "If you specify an input ``fname``"
        assert substr in err.value.args[0]
        substr = "do not also specify ``redshift``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments5(self):
        fname = "abc"

        with pytest.raises(HalotoolsError) as err:
            halocat = CachedHaloCatalog(fname=fname, redshift=0)
        substr = "non-existent path"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_acceptable_arguments6(self):
        cache = HaloTableCache()
        fname = cache.log[0].fname
        halocat = CachedHaloCatalog(fname=fname)

    @pytest.mark.skipif("not APH_MACHINE")
    def test_relocate_simulation_data(self):

        dman = DownloadManager()
        cache = HaloTableCache()

        ######################################################
        # Make sure the file does not already exist on disk or in cache
        tmp_fname = "/Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.07835.list.halotools_alpha_version2.hdf5"

        if os.path.isfile(tmp_fname):
            matching_log_entry = cache.determine_log_entry_from_fname(tmp_fname)

            cache.remove_entry_from_cache_log(
                simname=matching_log_entry.simname,
                halo_finder=matching_log_entry.halo_finder,
                version_name=matching_log_entry.version_name,
                redshift=matching_log_entry.redshift,
                fname=matching_log_entry.fname,
                update_ascii=True,
                delete_corresponding_halo_catalog=True,
                raise_non_existence_exception=False,
            )

            assert matching_log_entry not in cache.log

        ######################################################
        # Enforce it does not exist on disk or in the log
        assert not os.path.isfile(tmp_fname)

        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name="halotools_alpha_version2",
                redshift=11.7632,
            )

        #####################################################
        # Now download the file and store it in cache

        dman.download_processed_halo_table(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7632,
            overwrite=True,
        )

        ######################################################
        # Enforce that the file is on disk, in cache, and loads
        assert os.path.isfile(tmp_fname)

        entry = cache.determine_log_entry_from_fname(tmp_fname)
        assert entry not in cache.log
        cache.update_log_from_current_ascii()
        assert entry in cache.log

        halocat = CachedHaloCatalog(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7632,
        )

        #####################################################
        # Now move the file to a new location
        new_fname = os.path.join(self.dummy_cache_baseloc, os.path.basename(tmp_fname))
        assert not os.path.isfile(new_fname)
        os.system("cp " + tmp_fname + " " + new_fname)
        os.system("rm " + tmp_fname)
        assert not os.path.isfile(tmp_fname)
        assert os.path.isfile(new_fname)

        ######################################################
        # Verify that we can no longer load the catalog from metadata
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name="halotools_alpha_version2",
                redshift=11.7632,
            )
        substr = "The following input fname does not exist: "
        assert substr in err.value.args[0]
        assert tmp_fname in err.value.args[0]

        ######################################################
        # Update the cache location using the CachedHaloCatalog

        del halocat
        halocat = CachedHaloCatalog(fname=new_fname, update_cached_fname=True)
        assert halocat.fname == new_fname
        del halocat
        ######################################################

        ######################################################
        # Verify that we can load the catalog from metadata again
        halocat = CachedHaloCatalog(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7632,
        )

        # ######################################################
        # # Now clean up and remove the file again
        cache.update_log_from_current_ascii()

        matching_log_entries = cache.matching_log_entry_generator(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7632,
            dz_tol=0.05,
        )

        for matching_log_entry in matching_log_entries:
            cache.remove_entry_from_cache_log(
                simname=matching_log_entry.simname,
                halo_finder=matching_log_entry.halo_finder,
                version_name=matching_log_entry.version_name,
                redshift=matching_log_entry.redshift,
                fname=matching_log_entry.fname,
                update_ascii=True,
                delete_corresponding_halo_catalog=True,
            )
        # ######################################################

        # ######################################################
        # # Enforce that the file is really gone
        with pytest.raises(InvalidCacheLogEntry) as err:
            halocat = CachedHaloCatalog(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name="halotools_alpha_version2",
                redshift=11.7632,
            )
        # ######################################################

    @pytest.mark.skipif("not HAS_H5PY")
    @pytest.mark.skipif("not APH_MACHINE")
    def test_user_supplied_ptcl_consistency(self):

        from ..user_supplied_ptcl_catalog import UserSuppliedPtclCatalog

        halocat = CachedHaloCatalog()

        ptclcat = UserSuppliedPtclCatalog(
            redshift=halocat.redshift,
            Lbox=halocat.Lbox,
            particle_mass=halocat.particle_mass,
            x=np.array(halocat.ptcl_table["x"]),
            y=np.array(halocat.ptcl_table["y"]),
            z=np.array(halocat.ptcl_table["z"]),
            vx=np.array(halocat.ptcl_table["vx"]),
            vy=np.array(halocat.ptcl_table["vy"]),
            vz=np.array(halocat.ptcl_table["vz"]),
        )
        ptclcat.ptcl_table["x"] = 0.0

        fname = os.path.join(self.dummy_cache_baseloc, "temp_particles.hdf5")

        ptclcat.add_ptclcat_to_cache(
            fname, halocat.simname, "temp_testing_version_name", "dummy string"
        )

        assert os.path.isfile(ptclcat.log_entry.fname)

        ptcl_cache = PtclTableCache()
        assert ptclcat.log_entry in ptcl_cache.log

        halocat2 = CachedHaloCatalog(ptcl_version_name="temp_testing_version_name")

        assert np.all(halocat2.ptcl_table["x"] == 0)
        assert not np.all(halocat.ptcl_table["x"] == 0)

        ptcl_cache.remove_entry_from_cache_log(
            ptclcat.log_entry.simname,
            ptclcat.log_entry.version_name,
            ptclcat.log_entry.redshift,
            ptclcat.log_entry.fname,
            raise_non_existence_exception=True,
            update_ascii=True,
            delete_corresponding_ptcl_catalog=True,
        )

        assert ptclcat.log_entry not in ptcl_cache.log
        assert os.path.isfile(ptclcat.log_entry.fname) is False

    def tearDown(self):
        try:
            shutil.rmtree(self.dummy_cache_baseloc)
        except:
            pass
