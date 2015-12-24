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
from .log_entry import HaloTableCacheLogEntry, get_redshift_string

from ..custom_exceptions import HalotoolsError

def _infer_redshift_from_input_fname(fname):
    """ Method extracts the portion of the Rockstar hlist fname
    that contains the scale factor of the snapshot, and returns a 
    float for the redshift inferred from this substring. 

    Parameters 
    ----------
    fname : string 
        Name of the Rockstar hlist file 

    Returns
    -------
    rounded_redshift : float
        Redshift of the catalog, rounded to four decimals. 

    Notes
    -----
    Assumes that the first character of the relevant substring
    is the one immediately following the first incidence of an underscore,
    and final character is the one immediately preceding the second decimal.
    These assumptions are valid for all catalogs currently on the hipacc website.

    """
    fname = os.path.basename(fname)
    first_index = fname.index('_')+1
    last_index = fname.index('.', fname.index('.')+1)
    scale_factor_substring = fname[first_index:last_index]
    scale_factor = float(scale_factor_substring)
    redshift = (1./scale_factor) - 1
    rounded_redshift = float(get_redshift_string(redshift))
    return rounded_redshift


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

        msg = ("\n\nThe information about your ascii file "
            "and the metadata about the catalog \n"
            "have been processed and no exceptions were raised. \n"
            "Use the ``read_halocat`` method to read the ascii data, \n"
            "setting the write_to_disk and update_cache_log arguments as you like. \n"
            "See the docstring of the ``read_halocat`` method\n"
            "for details about these options. \n")
        print(msg)

    def _check_cache_log_for_matching_catalog(self):
        """ If this method raises no exceptions, 
        then either there are no conflicting log entries 
        or self.overwrite is set to True. In either case, 
        no if there are no exceptions then it is safe to 
        reduce the catalog, write it to disk, and request that 
        the cache be updated. 

        Note, though, that this does not guarantee that the 
        resulting processed catalog will be safe to store in cache, 
        as the processed catalog will be subject to further testing before 
        it can be cached (for example, the halo_id column of the processed catalog 
        must contain a set of unique integers). 
        """
        self.halo_table_cache = HaloTableCache()
        self.log_entry = HaloTableCacheLogEntry(simname = self.simname, 
            halo_finder = self.halo_finder, version_name = self.version_name, 
            redshift = self.redshift, fname = self.output_fname)

        if self.log_entry in self.halo_table_cache.log:
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
                linenum = self.halo_table_cache.log.index(self.log_entry) + 1
                msg += ("In order to proceed, "
                    "you must either set ``overwrite`` to True \n"
                    "or manually delete the existing file and also "
                    "remove the entry from the log.\n"
                    "To delete an entry from the log, \n"
                    "you can either use the `remove_entry_from_cache_log` method \n"
                    "of the HaloTableCache class, or equivalently, you can \n"
                    "manually delete line #"+str(linenum)+"from the log file. \n"
                    "The log file is stored in the following location:\n"
                    +self.halo_table_cache.cache_log_fname+"\n"
                    )
                raise HalotoolsError(msg)
        # there are no exact matches, but there may accidentally be nearby redshifts
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
                    msg = ("\nThe following filenames appear in the cache log \n"
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

        return output_fname

    def read_halocat(self, 
        write_to_disk = False, update_cache_log = False, 
        add_supplementary_halocat_columns = True):
        """ Method reads the ascii data and  
        binds the resulting catalog to ``self.halo_table``.

        Parameters 
        -----------
        write_to_disk : bool, optional 
            If True, the `write_to_disk` method will be called automatically. 
            Default is False, in which case you must call the `write_to_disk` 
            method yourself to store the processed catalog. 

        update_cache_log : bool, optional 
            If True, the `update_cache_log` method will be called automatically. 
            Default is False, in which case you must call the `update_cache_log` 
            method yourself to add the the processed catalog to the cache. 

        add_supplementary_halocat_columns : bool, optional 
            Boolean determining whether the halo_table will have additional 
            columns added to it computed by the add_supplementary_halocat_columns method. 
            Default is True. 
        """
        result = self.read_ascii()
        self.halo_table = Table(result)

        if add_supplementary_halocat_columns == True: 
            self.add_supplementary_halocat_columns()

        if write_to_disk is True: 
            self.write_to_disk()
            self._file_has_been_written_to_disk = True
        else:
            self._file_has_been_written_to_disk = False

        if update_cache_log == True:
            if self._file_has_been_written_to_disk == True: 
                self.update_cache_log()
            else:
                msg = ("\nYou set update_cache_log to True but the \n"
                    "newly processed halo_table has not yet been written to disk.\n")
                if write_to_disk == False:
                    msg += ("This is because you set write_to_disk to False, \n"
                        "in which case the read_halocat method "
                        "will not automatically update the cache.\n")
                    warn(msg)
                else:
                    msg += ("This indicates that there was a problem "
                        "writing the halo_table to disk.\n"
                        "After you resolve the problem, you can manually call \n"
                        "the write_to_disk and update_cache_log methods.\n")
                    raise HalotoolsError(msg)

    def write_to_disk(self):
        """ Method writes ``self.halo_table`` to ``self.output_fname`` 
        and also calls the ``self._write_metadata`` method to place the 
        hdf5 file into standard form. 
        """
        self.halo_table.write(
            self.output_fname, path='data', overwrite = self.overwrite)
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


    def update_cache_log(self):
        """ Method updates the cache log with the new catalog, 
        provided that it is safe to add to the cache. 
        """
        self.halo_table_cache.add_entry_to_cache_log(
            self.log_entry, update_ascii = True)

    def add_supplementary_halocat_columns(self):
        """ Add the halo_nfw_conc and halo_hostid columns. 
        This implementation will eventually change in favor of something 
        more flexible. 
        """
        ### Add the halo_nfw_conc column
        if ('halo_rvir' in self.halo_table.keys()) & ('halo_rs' in self.halo_table.keys()):
            self.halo_table['halo_nfw_conc'] = (
                self.halo_table['halo_rvir'] / self.halo_table['halo_rs']
                )

        ### Add the halo_nfw_conc column
        self.halo_table['halo_hostid'] = self.halo_table['halo_id']
        subhalo_mask = self.halo_table['halo_upid'] != -1
        self.halo_table['halo_hostid'][subhalo_mask] = (
            self.halo_table['halo_upid'][subhalo_mask]
            )







