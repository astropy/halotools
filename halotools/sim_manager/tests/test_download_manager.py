""" Module providing unit-testing for `~halotools.sim_manager.DownloadManager`.
"""

import os
import shutil
import numpy as np
from astropy.config.paths import _find_home
from astropy.tests.helper import pytest
from astropy.table import Table
from unittest import TestCase

from ..download_manager import DownloadManager
from ..halo_table_cache import HaloTableCache
from .. import sim_defaults
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


__all__ = ("TestDownloadManager",)


class TestDownloadManager(TestCase):
    def setUp(self):

        homedir = _find_home()

        self.downman = DownloadManager()

        def defensively_create_empty_dir(dirname):

            if os.path.isdir(dirname) is False:
                os.mkdir(dirname)
            else:
                shutil.rmtree(dirname)
                os.mkdir(dirname)

        # First create an empty directory where we will
        # temporarily store a collection of empty files
        self.base_dummydir = os.path.join(
            homedir, ".temp_directory_for_halotools_testing"
        )
        defensively_create_empty_dir(self.base_dummydir)
        self.dummyloc = os.path.join(self.base_dummydir, "halotools")
        defensively_create_empty_dir(self.dummyloc)

        self.halocat_dir = os.path.join(self.dummyloc, "halo_catalogs")
        defensively_create_empty_dir(self.halocat_dir)

        self.ptclcat_dir = os.path.join(self.dummyloc, "particle_catalogs")
        defensively_create_empty_dir(self.ptclcat_dir)

        self.raw_halo_table_dir = os.path.join(self.dummyloc, "raw_halo_catalogs")
        defensively_create_empty_dir(self.raw_halo_table_dir)

        self.simnames = ["bolshoi", "bolplanck", "multidark", "consuelo"]
        self.halo_finders = ["rockstar", "bdm"]
        self.dummy_version_names = ["halotools.alpha"]
        self.extension = ".hdf5"

        self.bolshoi_fnames = [
            "hlist_0.33035",
            "hlist_0.54435",
            "hlist_0.67035",
            "hlist_1.00035",
        ]
        self.bolshoi_bdm_fnames = [
            "hlist_0.33030",
            "hlist_0.49830",
            "hlist_0.66430",
            "hlist_1.00035",
        ]
        self.bolplanck_fnames = [
            "hlist_0.33035",
            "hlist_0.54435",
            "hlist_0.67035",
            "hlist_1.00035",
        ]
        self.consuelo_fnames = [
            "hlist_0.33324",
            "hlist_0.50648",
            "hlist_0.67540",
            "hlist_1.00000",
        ]
        self.multidark_fnames = [
            "hlist_0.31765",
            "hlist_0.49990",
            "hlist_0.68215",
            "hlist_1.00109",
        ]

        # make all relevant subdirectories and dummy files
        for simname in self.simnames:
            simdir = os.path.join(self.halocat_dir, simname)
            defensively_create_empty_dir(simdir)
            rockstardir = os.path.join(simdir, "rockstar")
            defensively_create_empty_dir(rockstardir)

            if simname == "bolshoi":
                fnames = self.bolshoi_fnames
            elif simname == "bolplanck":
                fnames = self.bolplanck_fnames
            elif simname == "consuelo":
                fnames = self.consuelo_fnames
            elif simname == "multidark":
                fnames = self.multidark_fnames

            for name in fnames:
                for version in self.dummy_version_names:
                    full_fname = name + "." + version + self.extension
                    abs_fname = os.path.join(rockstardir, full_fname)
                    _t = Table({"x": [0]})
                    _t.write(abs_fname, format="ascii")

            if simname == "bolshoi":
                simdir = os.path.join(self.halocat_dir, simname)
                bdmdir = os.path.join(simdir, "bdm")
                defensively_create_empty_dir(bdmdir)
                fnames = self.bolshoi_bdm_fnames
                for name in fnames:
                    for version in self.dummy_version_names:
                        full_fname = name + "." + version + self.extension
                        abs_fname = os.path.join(bdmdir, full_fname)
                        _t = Table({"x": [0]})
                        _t.write(abs_fname, format="ascii")

        p = os.path.join(self.halocat_dir, "bolshoi", "bdm")
        assert os.path.isdir(p)
        f = "hlist_0.33030.halotools.alpha.hdf5"
        full_fname = os.path.join(p, f)
        assert os.path.isfile(full_fname)

        self.clear_APH_MACHINE_of_highz_file()

    @pytest.mark.skipif("not APH_MACHINE")
    def test_ptcl_tables_available_for_download(self):

        file_list = self.downman._ptcl_tables_available_for_download(simname="bolshoi")
        assert len(file_list) == 1
        assert "hlist_1.00035.particles.halotools_v0p4.hdf5" == os.path.basename(
            file_list[0]
        )

        file_list = self.downman._ptcl_tables_available_for_download(
            simname="multidark"
        )
        assert len(file_list) == 1
        assert "hlist_1.00109.particles.halotools_v0p4.hdf5" == os.path.basename(
            file_list[0]
        )

        consuelo_set = set(
            [
                "hlist_0.33324.particles.halotools_v0p4.hdf5",
                "hlist_0.50648.particles.halotools_v0p4.hdf5",
                "hlist_0.67540.particles.halotools_v0p4.hdf5",
                "hlist_1.00000.particles.halotools_v0p4.hdf5",
            ]
        )
        file_list = self.downman._ptcl_tables_available_for_download(simname="consuelo")
        assert len(file_list) == 4
        file_set = set([os.path.basename(f) for f in file_list])
        assert file_set == consuelo_set

        bolplanck_set = set(
            [
                "hlist_0.33406.particles.halotools_v0p4.hdf5",
                "hlist_0.50112.particles.halotools_v0p4.hdf5",
                "hlist_0.66818.particles.halotools_v0p4.hdf5",
                "hlist_1.00231.particles.halotools_v0p4.hdf5",
            ]
        )
        file_list = self.downman._ptcl_tables_available_for_download(
            simname="bolplanck"
        )
        assert len(file_list) == 4
        file_set = set([os.path.basename(f) for f in file_list])
        assert file_set == bolplanck_set

    @pytest.mark.skipif("not APH_MACHINE")
    def test_processed_halo_tables_available_for_download1(self):

        file_list = self.downman._processed_halo_tables_available_for_download(
            simname="bolshoi", halo_finder="rockstar"
        )
        assert file_list != []

    @pytest.mark.skipif("not APH_MACHINE")
    def test_processed_halo_tables_available_for_download2(self):

        file_list = self.downman._processed_halo_tables_available_for_download(
            simname="bolshoi"
        )
        assert file_list != []

    @pytest.mark.skipif("not APH_MACHINE")
    def test_processed_halo_tables_available_for_download3(self):

        file_list = self.downman._processed_halo_tables_available_for_download(
            halo_finder="bdm"
        )
        assert file_list != []

    @pytest.mark.skipif("not APH_MACHINE")
    def test_ptcl_tables_available_for_download2(self):
        """Test that there is exactly one ptcl_table available for Bolshoi."""
        x = self.downman._ptcl_tables_available_for_download(simname="bolshoi")
        assert len(x) == 1

        x = self.downman._ptcl_tables_available_for_download(simname="bolplanck")
        assert len(x) == 4

        x = self.downman._ptcl_tables_available_for_download(simname="consuelo")
        assert len(x) == 4

        x = self.downman._ptcl_tables_available_for_download(simname="multidark")
        assert len(x) == 1

    def test_get_scale_factor_substring(self):
        """ """
        f = self.downman._get_scale_factor_substring("hlist_0.50648.particles.hdf5")
        assert f == "0.50648"

    def test_closest_fname(self):
        """ """
        f, z = self.downman._closest_fname(
            [
                "hlist_0.50648.particles.hdf5",
                "hlist_0.67540.particles.hdf5",
                "hlist_0.33324.particles.hdf5",
            ],
            100.0,
        )
        assert (f, np.round(z, 2)) == ("hlist_0.33324.particles.hdf5", 2.0)

        f, z = self.downman._closest_fname(
            [
                "hlist_0.50648.particles.hdf5",
                "hlist_0.67540.particles.hdf5",
                "hlist_0.33324.particles.hdf5",
            ],
            1.0,
        )
        assert (f, np.round(z, 1)) == ("hlist_0.50648.particles.hdf5", 1.0)

    @pytest.mark.skipif("not APH_MACHINE")
    def test_unsupported_sim_download_attempt(self):
        simname = "consuelo"
        redshift = 2
        halo_finder = "bdm"
        with pytest.raises(HalotoolsError) as exc:
            self.downman.download_processed_halo_table(
                simname=simname,
                halo_finder=halo_finder,
                redshift=redshift,
                overwrite=False,
                download_dirname=self.halocat_dir,
            )

    def test_orig_halo_table_web_location(self):
        """Test will fail unless the web locations are held fixed
        to their current, hard-coded values.
        """
        assert (
            "www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM"
            in self.downman._orig_halo_table_web_location(
                simname="bolshoi", halo_finder="bdm"
            )
        )

        assert (
            "www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/"
            in self.downman._orig_halo_table_web_location(
                simname="bolshoi", halo_finder="rockstar"
            )
        )

        assert (
            "tp://www.slac.stanford.edu/~behroozi/BPlanck_Hlists"
            in self.downman._orig_halo_table_web_location(
                simname="bolplanck", halo_finder="rockstar"
            )
        )

        assert (
            "c.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar"
            in self.downman._orig_halo_table_web_location(
                simname="multidark", halo_finder="rockstar"
            )
        )

        assert (
            "/www.slac.stanford.edu/~behroozi/Consuelo_Catalo"
            in self.downman._orig_halo_table_web_location(
                simname="consuelo", halo_finder="rockstar"
            )
        )

    @pytest.mark.skipif("not APH_MACHINE")
    def test_closest_halo_catalog_on_web1(self):
        """ """
        fname, redshift = self.downman._closest_catalog_on_web(
            simname="bolshoi",
            halo_finder="rockstar",
            desired_redshift=0.0,
            catalog_type="halos",
        )
        assert "hlist_1.00035.list.halotools_v0p4.hdf5" in fname

    @pytest.mark.skipif("not APH_MACHINE")
    def test_closest_halo_catalog_on_web2(self):
        """ """
        fname, redshift = self.downman._closest_catalog_on_web(
            simname="bolshoi",
            halo_finder="bdm",
            desired_redshift=0.0,
            catalog_type="halos",
        )
        assert "bolshoi/bdm/hlist_1.00030.list.halotools_v0p4.hdf5" in fname

    @pytest.mark.skipif("not APH_MACHINE")
    def test_closest_halo_catalog_on_web3(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            fname, redshift = self.downman._closest_catalog_on_web(
                simname="bolshoi",
                halo_finder="bdm",
                desired_redshift=0.0,
                catalog_type="Jose Canseco",
            )
        substr = "Input ``catalog_type`` must be either ``particles`` or ``halos``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_closest_ptcl_catalog_on_web(self):
        """This test currently fails because the halo catalogs have not been updated yet."""
        fname, redshift = self.downman._closest_catalog_on_web(
            simname="bolplanck", desired_redshift=2, catalog_type="particles"
        )
        assert "bolplanck/hlist_0.33406.particles.halotools_v0p4.hdf5" in fname

    @classmethod
    def clear_APH_MACHINE_of_highz_file(self, delete_corresponding_halo_catalog=True):

        cache = HaloTableCache()
        matching_log_entries = cache.matching_log_entry_generator(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7,
            dz_tol=0.2,
        )
        for matching_log_entry in matching_log_entries:
            cache.remove_entry_from_cache_log(
                simname=matching_log_entry.simname,
                halo_finder=matching_log_entry.halo_finder,
                version_name=matching_log_entry.version_name,
                redshift=matching_log_entry.redshift,
                fname=matching_log_entry.fname,
                update_ascii=True,
                delete_corresponding_halo_catalog=delete_corresponding_halo_catalog,
            )

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table1(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="Jose Canseco",
                halo_finder="rockstar",
                version_name="halotools_alpha_version2",
                redshift=11.7,
                download_dirname=self.halocat_dir,
            )
        substr = "no web locations"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table2(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name="Jose Canseco",
                redshift=11.7,
                download_dirname=self.halocat_dir,
            )
        substr = "no halo catalogs meeting"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table3(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name=sim_defaults.default_version_name,
                redshift=0,
                overwrite=False,
            )
        substr = "you must set the ``overwrite`` keyword argument to True"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table4(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name=sim_defaults.default_version_name,
                redshift=0,
                overwrite=False,
                download_dirname="abc",
            )
        substr = "Your input ``download_dirname`` is a non-existent path."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table5(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name="halotools_v0p4",
                redshift=11.7,
                dz_tol=200,
                overwrite=True,
                download_dirname="std_cache_loc",
            )
        substr = "the ``ignore_nearby_redshifts`` to True, or decrease ``dz_tol``"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table6(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name=sim_defaults.default_version_name,
                redshift=0.3,
                dz_tol=0.001,
                overwrite=False,
                download_dirname=self.halocat_dir,
            )
        substr = "The closest redshift for these catalogs is"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_processed_halo_table7(self):
        """ """

        self.clear_APH_MACHINE_of_highz_file()

        cache1 = HaloTableCache()
        self.downman.download_processed_halo_table(
            simname="bolshoi",
            halo_finder="rockstar",
            version_name="halotools_alpha_version2",
            redshift=11.7,
            overwrite=True,
        )
        cache2 = HaloTableCache()
        assert len(cache1.log) == len(cache2.log) - 1
        new_entry = list(set(cache2.log) - set(cache1.log))[0]
        assert os.path.isfile(new_entry.fname)

        self.clear_APH_MACHINE_of_highz_file()

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_ptcl_table1(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_ptcl_table(
                simname="Jose Canseco",
                version_name=sim_defaults.default_ptcl_version_name,
                redshift=11.7,
                download_dirname=self.halocat_dir,
            )
        substr = "no web locations"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_ptcl_table2(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_ptcl_table(
                simname="bolshoi",
                version_name="Jose Canseco",
                redshift=11.7,
                download_dirname=self.halocat_dir,
            )
        substr = "There are no particle catalogs meeting your specifications"
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_ptcl_table3(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_ptcl_table(
                simname="bolshoi",
                version_name=sim_defaults.default_ptcl_version_name,
                redshift=0,
                download_dirname=self.halocat_dir,
            )
        substr = "you must set the ``overwrite`` keyword argument to True."
        assert substr in err.value.args[0]

    def test_download_ptcl_table4(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_ptcl_table(
                simname="bolshoi",
                version_name=sim_defaults.default_ptcl_version_name,
                redshift=0,
                download_dirname="abc",
            )
        substr = "Your input ``download_dirname`` is a non-existent path."
        assert substr in err.value.args[0]

    @pytest.mark.skipif("not APH_MACHINE")
    def test_download_ptcl_table5(self):
        """ """
        with pytest.raises(HalotoolsError) as err:
            self.downman.download_ptcl_table(
                simname="bolshoi",
                version_name=sim_defaults.default_ptcl_version_name,
                redshift=0.2,
                dz_tol=0.001,
                download_dirname=self.halocat_dir,
            )
        substr = "The closest redshift for these catalogs is"
        assert substr in err.value.args[0]

    def tearDown(self):
        try:
            shutil.rmtree(self.base_dummydir)
        except:
            pass
