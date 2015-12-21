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
        row_cut_eq_dict = {}, row_cut_neq_dict = {}, **kwargs):
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










