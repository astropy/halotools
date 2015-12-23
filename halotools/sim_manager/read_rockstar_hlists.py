# -*- coding: utf-8 -*-
"""
Module storing the RockstarHlistReader, 
the primary class used by Halotools to process 
publicly available Rockstar hlist files and store them in cache. 

"""

__all__ = ('RockstarHlistReader', )

import os
import gzip
from time import time
import numpy as np
from warnings import warn 
from astropy.table import Table
from astropy.table import vstack as table_vstack 
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir

import datetime

from .tabular_ascii_reader import TabularAsciiReader
from . import manipulate_cache_log 
from .halo_table_cache import HaloTableCache
from .log_entry import HaloTableCacheLogEntry

from ..custom_exceptions import HalotoolsError

class RockstarHlistReader(TabularAsciiReader):
    """ The `RockstarHlistReader` reads Rockstar hlist ASCII files, 
    stores them as hdf5 files in the Halotools cache, and updates the cache log. 


    `RockstarHlistReader` is a subclass of 
    `~halotools.sim_manager.TabularAsciiReader`, and supplements this behavior 
    with the ability to read, update, and search the Halotools cache log. 

    If you are planning to use the Halotools cache manager to store and 
    keep track of your halo catalogs, this is the class to use. For a stand-alone 
    reader of Rockstar hlists, you should instead use the 
    `~halotools.sim_manager.TabularAsciiReader` class. 

    """

    def __init__(self, input_fname, columns_to_keep_dict, 
        output_fname, simname, halo_finder, redshift, version_name, 
        Lbox, particle_mass, header_char='#', 
        row_cut_min_dict = {}, row_cut_max_dict = {}, 
        row_cut_eq_dict = {}, row_cut_neq_dict = {}, 
        overwrite = False, ignore_nearby_redshifts = False, dz_tol = 0.05, 
        processing_notes = ' ', **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        columns_to_keep_dict : dict 
            Dictionary used to define which columns 
            of the tabular ASCII data will be kept.

            Each key of the dictionary will be the name of the 
            column in the returned data table. The value bound to 
            each key is a two-element tuple. 
            The first tuple entry is an integer providing 
            the *index* of the column to be kept, starting from 0. 
            The second tuple entry is a string defining the Numpy dtype 
            of the data in that column, 
            e.g., 'f4' for a float, 'f8' for a double, 
            or 'i8' for a long. 

            Thus an example ``columns_to_keep_dict`` could be 
            {'halo_mvir': (1, 'f4'), 'halo_id': (0, 'i8'), 'halo_spin': (45, 'f4')}

            The columns of all halo tables stored in the Halotools cache must 
            must begin with the substring 'halo_'. 
            At a minimum, any halo table stored in cache 
            must have the following columns:  
            ``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``, 
            plus at least one additional column (typically storing a mass-like variable). 
            These requirements must be met if you want to use the Halotools cache system, 
            or if you want Halotools to populate your halo catalog with mock galaxies. 
            If you do not want to conform to these conventions, just use the 
            `~halotools.sim_manager.TabularAsciiReader` and handle 
            the file storage using your own preferred method. 

        output_fname : string 
            Absolute path to the location where the hdf5 file will be stored. 
            The file extension must be '.hdf5'. 
            If the file already exists, you must set 
            the keyword argument ``overwrite`` to True. 

            If output_fname is set to `std_cache_loc`, Halotools will place the 
            catalog in the following location: 

            $HOME/.astropy/cache/halotools/halo_catalogs/simname/halo_finder/input_fname.version_name.hdf5

        simname : string 
            Nickname of the simulation used as a shorthand way to keep track 
            of the halo catalogs in your cache. The simnames processed by Halotools are 
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'. 

        halo_finder : string 
            Nickname of the halo-finder used to generate the hlist file from particle data. 
            Most likely this should be 'rockstar', though there are also 
            publicly available hlists processed with the 'bdm' halo-finder. 

        redshift : float 
            Redshift of the halo catalog. 

        version_name : string 
            Nickname of the version of the halo catalog you produce using RockstarHlistReader. 
            The ``version_name`` is used as a bookkeeping tool in the cache log.

            It is not permissible to use a value of ``version_name`` that matches 
            the version name(s) used for the default catalogs provided by Halotools. 
            If you process your own halo catalog with the RockstarHlistReader, 
            you should choose your own version name. 

        Lbox : float 
            Box size of the simulation in Mpc/h.
            ``Lbox`` will automatically be added to the ``supplementary_metadata_dict`` 
            so that your hdf5 file will have the box size bound as metadata. 

        particle_mass : float 
            Mass of the dark matter particles of the simulation in Msun/h.
            ``particle_mass`` will automatically be added to the ``supplementary_metadata_dict`` 
            so that your hdf5 file will have the particle mass bound as metadata. 

        row_cut_min_dict : dict, optional 
            Dictionary used to place a lower-bound cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the lower bound on the data stored 
            in that row. A row with a smaller value than this lower bound for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_min_dict = {'mass': 1e10}, then all rows of the 
            returned data table will have a mass greater than 1e10. 

        row_cut_max_dict : dict, optional 
            Dictionary used to place an upper-bound cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the upper bound on the data stored 
            in that row. A row with a larger value than this upper bound for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_min_dict = {'mass': 1e15}, then all rows of the 
            returned data table will have a mass less than 1e15. 

        row_cut_eq_dict : dict, optional 
            Dictionary used to place an equality cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the required value for the data stored 
            in that row. Only rows with a value equal to this required value for the 
            corresponding column will appear in the returned data table. 

            For example, if row_cut_eq_dict = {'upid': -1}, then *all* rows of the 
            returned data table will have a upid of -1. 

        row_cut_neq_dict : dict, optional 
            Dictionary used to place an inequality cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as a forbidden value for the data stored 
            in that row. Rows with a value equal to this forbidden value for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_neq_dict = {'upid': -1}, then *no* rows of the 
            returned data table will have a upid of -1. 

        header_char : str, optional
            String to be interpreted as a header line of the ascii hlist file. 
            Default is '#'. 

        overwrite : bool, optional 
            If the chosen ``output_fname`` already exists, then you must set ``overwrite`` 
            to True in order to write the file to disk. Default is False. 

        ignore_nearby_redshifts : bool, optional 
            Flag used to determine whether nearby redshifts in cache will be ignored. 
            If there are existing halo catalogs in the Halotools cache with matching 
            ``simname``, ``halo_finder`` and ``version_name``, and if one or more of those 
            catalogs has a redshift within ``dz_tol``, 
            then the ignore_nearby_redshifts flag must be set to True 
            for the new halo catalog to be stored in cache. 
            Default is False. 

        dz_tol : float, optional 
            Tolerance determining when another halo catalog in cache is deemed nearby.
            Default is 0.05. 

        processing_notes : string, optional 
            String used to provide supplementary notes that will be attached to 
            the hdf5 file storing your halo catalog. 

        Notes 
        ------
        When the ``row_cut_min_dict``, ``row_cut_max_dict``, 
        ``row_cut_eq_dict`` and ``row_cut_neq_dict`` keyword arguments are used 
        simultaneously, only rows passing all cuts will be kept. 
        """
        try:
            import h5py 
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "use the RockstarHlistReader to store your catalog in the Halotools cache. \n"
                "For a stand-alone reader class, you should instead use TabularAsciiReader.\n")
            raise HalotoolsError(msg)

        TabularAsciiReader.__init__(self, 
            input_fname, columns_to_keep_dict, 
            header_char, row_cut_min_dict, row_cut_max_dict, 
            row_cut_eq_dict, row_cut_neq_dict)

        # Require that the minimum required columns have been selected, 
        # and that they all begin with `halo_`
        self._enforce_halo_catalog_formatting_requirements()

        # Bind the constructor arguments to the instance
        self.simname = simname 
        self.halo_finder = halo_finder
        self.redshift = float(manipulate_cache_log.get_redshift_string(redshift)) 
        self.version_name = version_name
        self.dz_tol = dz_tol
        self.Lbox = Lbox
        self.particle_mass = particle_mass
        self.overwrite = overwrite
        self.ignore_nearby_redshifts = ignore_nearby_redshifts
        self.processing_notes = processing_notes

        self.output_fname = (
            self._retrieve_output_fname(output_fname, self.overwrite, **kwargs)
            )

        self._check_cache_log_for_matching_catalog()

    def _check_cache_log_for_matching_catalog(self):
        """
        """
        self.halo_table_cache = HaloTableCache()
        self.log_entry = HaloTableCacheLogEntry(simname = self.simname, 
            halo_finder = self.halo_finder, version_name = self.version_name, 
            redshift = self.redshift, fname = self.output_fname)

        if (self.log_entry in self.halo_table_cache.log):
            msg = ("\nThere is already an existing entry "
                "in the Halotools cache log\n"
                "that exactly matches the filename and "
                "metadata of the file you intend to write.\n")
            if self.overwrite == True:
                msg += ("Because you have set ``overwrite`` to True, "
                    "\ncalling the read_and_store_halocat_in_cache "
                    "method will overwrite the existing file and log entry.\n")
                warn(msg)
            else:
                msg += ("In order to proceed, "
                    "you must either set ``overwrite`` to True \n"
                    "or manually delete the existing file and log entry.\n")
        else:
            if self.ignore_nearby_redshifts == False:
                pass
            else:
                closely_matching_catalogs = list(
                    self.halo_table_cache.matching_log_entry_generator(
                        simname = self.simname, 
                        halo_finder = self.halo_finder, 
                        version_name = self.version_name, 
                        redshift = self.redshift, 
                        fname = self.output_fname, dz_tol = self.dz_tol)
                        )
                if len(closely_matching_catalogs) > 0:
                    msg = ("\The following filenames appear in the cache log \n"
                        "and exactly matching metadata and closely matching redshifts: \n\n")
                    for entry in closely_matching_catalogs:
                        msg += str(entry.fname) + "\n"
                    msg += ("In order to proceed, you must either set "
                        "the ``ignore_nearby_redshifts`` to True, \n"
                        "or choose a different ``version_name`` for your catalog.\n")
                    raise HalotoolsError(msg)




    def _enforce_halo_catalog_formatting_requirements(self):
        """ Private method enforces the halo_table formatting conventions of the package. 
        """
        try:
            assert 'halo_id' in self.dt.names
            assert 'halo_x' in self.dt.names
            assert 'halo_y' in self.dt.names
            assert 'halo_z' in self.dt.names
        except AssertionError:
            msg = ("\nAll halo tables stored in cache \n"
            "must at least have the following columns:\n"
                "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``.\n"
                "If you do not intend to use store the catalog in cache, \n"
                "you should instead process the file with the "
                "`~halotools.sim_manager.TabularAsciiReader`.\n"
                )
            raise HalotoolsError(msg)

        for name in self.dt.names:
            try:
                assert name[0:5] == 'halo_'
            except AssertionError:
                msg = ("\nAll columns of halo tables stored in the Halotools cache\n"
                    "must begin with the substring ``halo_``.\n")
                msg += ("The column name ``"+name+"`` "
                    "appeared in your input ``columns_to_keep_dict``.\n"
                    "If you do not intend to use store the catalog in cache, \n"
                    "you should instead process the file with the "
                    "`~halotools.sim_manager.TabularAsciiReader`.\n"
                    )
                raise HalotoolsError(msg)

    def _get_default_output_fname(self):
        """
        """
        basedir = get_astropy_cache_dir()
        dirname = os.path.join(basedir, 'halotools', 'halo_catalogs', 
            self.simname, self.halo_finder)
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        basename = (
            os.path.basename(self.input_fname) + 
            '.' + self.version_name + '.hdf5'
            )
        default_fname = os.path.join(dirname, basename)
        return default_fname

    def _retrieve_output_fname(self, output_fname, overwrite, **kwargs):
        """ Private method checks to see whether the chosen 
        ``output_fname`` already exists on disk, and enforces 
        compatibility with ``overwrite``. 
        """
        if output_fname == 'std_cache_loc':
            output_fname = self._get_default_output_fname()

        try:
            assert output_fname[-5:] == '.hdf5'
        except:
            msg = ("\nThe output_fname must be a string or unicode that concludes "
                " with file extension '.hdf5'\n")
            raise HalotoolsError(msg)

        if os.path.isfile(output_fname):
            if overwrite == True:
                msg = ("\nThe chosen ``output_fname``, \n"+output_fname+"\n"
                    "already exists and will be overwritten when the \n"
                    "`store_halo_catalog_in_cache` method is called.\n")
                warn(msg)
            else:
                msg = ("\nThe chosen ``output_fname``, \n"+output_fname+"\n"
                    "already exists. You must set overwrite to True if you want to "
                    "use the `store_halo_catalog_in_cache` method.\n")
                raise IOError(msg)

        return output_fname

    def read_and_store_halocat_in_cache(self, **kwargs):
        """ Method reads the ASCII data, stores the result as an hdf5 file 
        in the Halotools cache, and updates the log. 
        """
        result = self.read_ascii()
        self.halo_table = Table(result)

        try:
            overwrite = kwargs['overwrite']
        except KeyError:
            overwrite = self.overwrite
        self.halo_table.write(self.output_fname, path='data', overwrite = overwrite)

        self._write_metadata()


    def _write_metadata(self):
        """ Private method to add metadata to the hdf5 file. 
        """
        import h5py
        # Now add the metadata 
        f = h5py.File(self.output_fname)
        f.attrs.create('simname', str(self.simname))
        f.attrs.create('halo_finder', str(self.halo_finder))
        redshift_string = str(manipulate_cache_log.get_redshift_string(self.redshift))
        f.attrs.create('redshift', redshift_string)
        f.attrs.create('version_name', str(self.version_name))
        f.attrs.create('fname', str(self.output_fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)
        f.attrs.create('orig_ascii_fname', str(self.input_fname))

        time_right_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.attrs.create('time_of_catalog_production', time_right_now)

        if self.processing_notes != None:
            f.attrs.create('processing_notes', str(self.processing_notes))

        # Store all the choices for row cuts as metadata
        for haloprop_key, cut_value in self.row_cut_min_dict.iteritems():
            attrname = haloprop_key + '_row_cut_min'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_max_dict.iteritems():
            attrname = haloprop_key + '_row_cut_max'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_eq_dict.iteritems():
            attrname = haloprop_key + '_row_cut_eq'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_neq_dict.iteritems():
            attrname = haloprop_key + '_row_cut_neq'
            f.attrs.create(attrname, cut_value)

        f.close()









