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

from .tabular_ascii_reader import TabularAsciiReader

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
        header_char='#', row_cut_min_dict = {}, row_cut_max_dict = {}, 
        row_cut_eq_dict = {}, row_cut_neq_dict = {}, 
        overwrite = False, **kwargs):
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
            {'mass': (1, 'f4'), 'obj_id': (0, 'i8'), 'spin': (45, 'f4')}

        output_fname : string 
            Absolute path to the location where the hdf5 file will be stored. 
            The file extension must be '.hdf5'. 
            If the file already exists, you must set 
            the keyword argument ``overwrite`` to True. 

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

        try:
            import h5py 
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "use the RockstarHlistReader to store your catalog in the Halotools cache. \n"
                "For a stand-alone reader class, you should instead use TabularAsciiReader.\n")
            warn(msg)

        self._check_output_fname(output_fname, overwrite)

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

    def _check_output_fname(self, output_fname, overwrite):
        """ Private method checks to see whether the chosen 
        ``output_fname`` already exists. 
        """
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

    def _check_cache_log_for_matching_catalog(self):
        """ Private method searches the Halotools cache log to see if there are 
        any entries with metadata that matches the RockstarHlistReader constructor inputs.  
        """
        pass








