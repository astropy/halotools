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
        processing_notes = None, **kwargs):
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
            catalog in the default cache location. 

        simname : string 
            Nickname of the simulation used as a shorthand way to keep track 
            of the halo catalogs in your cache. The simnames processed by Halotools are 
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'. 

        halo_finder : string 
            Nickname of the halo-finder used to generate the hlist file from particle data. 
            Most likely this should be 'rockstar', though there are also 
            publicly available hlists processed with the 'bdm' halo-finder. 

        redshift : float 
            Redshift of the halo catalog 

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
            ``simname``, ``halo_finder`` and ``version_name``, and a redshift 
            within ``dz_tol``, then the ignore_nearby_redshifts flag must be set to True 
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

        TabularAsciiReader.__init__(self, 
            input_fname, columns_to_keep_dict, 
            header_char, row_cut_min_dict, row_cut_max_dict, 
            row_cut_eq_dict, row_cut_neq_dict)

        self._enforce_halo_catalog_formatting_requirements()

        self.simname = simname 
        self.halo_finder = halo_finder
        self.redshift = redshift 
        self.version_name = version_name
        self.ignore_nearby_redshifts = ignore_nearby_redshifts
        self.dz_tol = dz_tol
        self.Lbox = Lbox
        self.particle_mass = particle_mass
        self.processing_notes = processing_notes

        try:
            import h5py 
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "use the RockstarHlistReader to store your catalog in the Halotools cache. \n"
                "For a stand-alone reader class, you should instead use TabularAsciiReader.\n")
            raise HalotoolsError(msg)

        self._verify_cache_log(**kwargs)

        self.output_fname = (
            self._retrieve_output_fname(output_fname, overwrite, **kwargs)
            )

        self._check_cache_log_for_matching_catalog(**kwargs)

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
                "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``\n")
            raise HalotoolsError(msg)

        for name in self.dt.names:
            try:
                assert name[0:5] == 'halo_'
            except AssertionError:
                msg = ("\nAll columns of halo tables stored in the Halotools cache\n"
                    "must begin with the substring ``halo_``.\n")
                msg += ("The column name ``"+name+"`` "
                    "appeared in your input ``columns_to_keep_dict``.\n"
                    )
                raise HalotoolsError(msg)

    def data_chunk_generator(self, chunk_size, f):
        """
        Python generator uses f.readline() to march 
        through an input open file object to yield 
        a chunk of data with length equal to the input ``chunk_size``. 
        The generator only yields columns that were included 
        in the ``columns_to_keep_dict`` passed to the constructor. 

        Parameters 
        -----------
        chunk_size : int 
            Number of rows of data in the chunk being generated 

        f : File
            Open file object being read

        Returns 
        --------
        chunk : tuple 
            Tuple of data from the ascii. 
            Only data from ``column_indices_to_keep`` are yielded. 

        """
        TabularAsciiReader.data_chunk_generator(self, chunk_size, f)

    def data_len(self):
        """ 
        Number of rows of data in the input ASCII file. 

        Returns 
        --------
        Nrows_data : int 
            Total number of rows of data. 

        Notes 
        -------
        The returned value is computed as the number of lines 
        between the returned value of `header_len` and 
        the next appearance of "\n" as the sole character on a line. 

        The `data_len` method is the particular section of code 
        where where the following assumptions are made:

        1. The data begins with the first appearance of a non-empty line that does not begin with the character defined by ``self.header_char``. 

        2. The data ends with the next appearance of an empty line. 
        """
        return TabularAsciiReader.data_len(self)

    def header_len(self):
        """ Number of rows in the header of the ASCII file. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        The header is assumed to be those characters at the beginning of the file 
        that begin with ``self.header_char``. 

        All empty lines that appear in header will be included in the count. 

        """
        return TabularAsciiReader.header_len(self)

    def apply_row_cut(self, array_chunk):
        """ Method applies a boolean mask to the input array 
        based on the row-cuts determined by the 
        dictionaries passed to the constructor. 

        Parameters 
        -----------
        array_chunk : Numpy array  

        Returns 
        --------
        cut_array : Numpy array             
        """ 
        return TabularAsciiReader.apply_row_cut(self, array_chunk)

    def read_ascii(self, chunk_memory_size = 500.):
        """ Method reads the input ascii and returns 
        a structured Numpy array of the data 
        that passes the row- and column-cuts. 

        Parameters 
        ----------
        chunk_memory_size : int, optional 
            Determine the approximate amount of Megabytes of memory 
            that will be processed in chunks. This variable 
            must be smaller than the amount of RAM on your machine; 
            choosing larger values typically improves performance. 
            Default is 500 Mb. 

        Returns 
        --------
        full_array : array_like 
            Structured Numpy array storing the rows and columns 
            that pass the input cuts. The columns of this array 
            are those selected by the ``column_indices_to_keep`` 
            argument passed to the constructor. 

        See also 
        ----------
        data_chunk_generator
        """
        return TabularAsciiReader.read_ascii(self, chunk_memory_size = 500.)

    def _verify_cache_log(self, **kwargs):
        """ Private method checks to see whether the cache log exists. 
        If so, verifies that the existing log is kosher. 
        """
        self._cache_log_exists, _ = (
            manipulate_cache_log.verify_cache_log(
                raise_non_existent_cache_exception = False, **kwargs)
            )

        if self._cache_log_exists == True:
            manipulate_cache_log.remove_repeated_cache_lines(**kwargs)

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
        ``output_fname`` already exists on disk, and also whether it 
        already appears in the cache log. If ovewrite is True, 
        the log will be cleaned of any entries with a matching output_fname. 
        If overwrite is False and a match is detected, an exception is raised. 
        The `_retrieve_output_fname` can safely be called even if the cache log 
        does not exist. 
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

        self._cache_log_exists, _ = (
            manipulate_cache_log.verify_cache_log(
                raise_non_existent_cache_exception = False, **kwargs)
            )
        if self._cache_log_exists == True:

            log = manipulate_cache_log.read_halo_table_cache_log(**kwargs)
            exact_match_mask, _ = (
                manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
                    fname = output_fname)
                )
            num_matches = len(log[exact_match_mask])

            if overwrite is False:
                if num_matches > 0:
                    cache_fname = manipulate_cache_log.get_halo_table_cache_log_fname(**kwargs)
                    idx = np.where(exact_match_mask == True)[0] + 2
                    linenum = idx[0] + 2
                    msg = ("\nHalotools detected an existing catalog in your cache \n"
                        "with a filename that matches your chosen ``output_fname``. \n"
                        "If you want to overwrite this existing log entry, \n"
                        "you must set the ``overwrite`` keyword argument to True. \n"
                        "Otherwise, choose a different ``output_fname``.\n\n"
                        "Alternatively, you can delete the existing entry from the log.\n"
                        "The log file is stored in the following location:\n\n"
                        +cache_fname+"\n\n"
                        )
                    msg += "The relevant line to change is line #" + str(linenum) + "\n\n"
                    raise HalotoolsError(msg)

            else:
                manipulate_cache_log.remove_unique_fname_from_halo_table_cache_log(
                    output_fname, raise_warning = False, **kwargs)

    def _check_cache_log_for_matching_catalog(self, **kwargs):
        """ Private method searches the Halotools cache log to see if there are 
        any entries with metadata that matches the RockstarHlistReader constructor inputs.  
        """

        if self._cache_log_exists == True:

            log = manipulate_cache_log.read_halo_table_cache_log(**kwargs)
            exact_match_mask, close_match_mask = (
                manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
                    simname = self.simname, 
                    halo_finder = self.halo_finder, 
                    redshift = self.redshift, 
                    version_name = self.version_name, 
                    dz_tol = self.dz_tol)
                )
            catalogs_with_exactly_matching_metadata = log[exact_match_mask]

            if len(catalogs_with_exactly_matching_metadata) > 0:
                cache_fname = manipulate_cache_log.get_halo_table_cache_log_fname(**kwargs)
                matching_fname = catalogs_with_exactly_matching_metadata['fname'][0]
                idx = np.where(exact_match_mask == True)[0]
                linenum = idx[0] + 2
                msg = ("\nHalotools detected one or more entries in your cache log \n"
                    "with metadata that exactly match your input \n"
                    "``simname``, ``halo_finder``, ``redshift`` and ``version_name``.\n"
                    "The first appearance of a matching entry in the log has the following filename:\n\n"
                    +matching_fname+"\n\n"
                    "If this log entry is invalid, use a text editor to open the log and delete the entry. \n"
                    "The cache log is stored in the following location:\n\n"
                    +cache_fname+"\n\n"
                    "The relevant line to change is line #" + str(linenum) + ",\n"
                    "where line #1 is the first line of the file.\n\n"
                    )
                raise HalotoolsError(msg)

            catalogs_with_close_redshifts = log[close_match_mask]
            if len(catalogs_with_close_redshifts) > 0:

                if self.ignore_nearby_redshifts == False:

                    msg = ("\nThere already exists a halo catalog in cache \n"
                        "with the same metadata as the catalog you are trying to store, \n"
                        "and a very similar redshift. \nThe closely matching "
                        "halo catalog has the following filename:\n\n"
                        +catalogs_with_close_redshifts['fname'][0]+"\n\n"
                        "If you want to proceed anyway, you must set the \n"
                        "``ignore_nearby_redshifts`` keyword argument to ``True``.\n"
                        )
                    raise HalotoolsError(msg)

    def read_and_store_halocat_in_cache(self, **kwargs):
        """ Method reads the ASCII data, stores the result as an hdf5 file 
        in the Halotools cache, and updates the log. 
        """
        self.halo_table = Table(self.read_ascii())
        self._verify_halo_table(self.halo_table)

        self.halo_table.write(self.output_fname, path='data')

        self._write_metadata()

        self._update_cache_log(**kwargs)

    def _update_cache_log(**kwargs):

        new_log_entry = Table(
            {'simname', [self.simname], 
            'halo_finder', [self.halo_finder], 
            'redshift', [self.redshift], 
            'version_name', [self.version_name], 
            'fname', [self.output_fname]}
            )

        if self._cache_log_exists is True:
            existing_log = manipulate_cache_log.read_halo_table_cache_log(**kwargs)
            new_log = table_vstack([existing_log, new_log_entry])
        else:
            new_log = new_log_entry

        manipulate_cache_log.overwrite_halo_table_cache_log(new_log, **kwargs)


    def _write_metadata(self):
        """ Private method to add metadata to the hdf5 file. 
        """
        # Now add the metadata 
        f = h5py.File(self.output_fname)
        f.attrs.create('simname', self.simname)
        f.attrs.create('halo_finder', self.halo_finder)
        redshift_string = manipulate_cache_log.get_redshift_string(self.redshift)
        f.attrs.create('redshift', self.redshift_string)
        f.attrs.create('version_name', self.version_name)
        f.attrs.create('fname', self.output_fname)

        f.attrs.create('orig_ascii_fname', self.input_fname)

        time_right_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.attrs.create('time_of_catalog_production', time_right_now)

        if self.processing_notes != None:
            f.attrs.create('processing_notes', self.processing_notes)

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


    def _verify_halo_table(self, halo_table):
        """
        """

        try:
            halo_id = halo_table['halo_id']
            halo_x = halo_table['halo_x']
            halo_y = halo_table['halo_y']
            halo_z = halo_table['halo_z']
        except KeyError:
            msg = ("\nAll halo tables stored in Haltools cache "
                "must at least have the following columns:\n"
                "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``\n")
            raise HalotoolsError(msg)

        # Check that Lbox properly bounds the halo positions
        try:
            assert np.all(halo_x >= 0)
            assert np.all(halo_y >= 0)
            assert np.all(halo_z >= 0)
            assert np.all(halo_x <= Lbox)
            assert np.all(halo_y <= Lbox)
            assert np.all(halo_z <= Lbox)
        except AssertionError:
            msg = ("\nThere are points in the input halo table that "
                "lie outside [0, Lbox] in some dimension.\n")
            raise HalotoolsError(msg)

        # Check that halo_id column contains a set of unique entries
        try:
            num_halos = len(halo_table)
            unique_halo_ids = list(set(halo_id))
            num_unique_ids = len(unique_halo_ids)
            assert num_halos == num_unique_ids
        except AssertionError:
            msg = ("\nThe ``halo_id`` column of your halo table "
                "must contain a unique integer for every row.\n")
            raise HalotoolsError(msg)











