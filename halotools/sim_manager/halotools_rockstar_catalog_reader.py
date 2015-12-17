# -*- coding: utf-8 -*-
"""
Methods and classes to read ASCII files storing simulation data. 

"""

__all__ = ('HalotoolsRockstarCatalogReader', )

import os
import gzip
from time import time
import numpy as np
from difflib import get_close_matches
from astropy.table import Table

from . import catalog_manager, supported_sims, sim_defaults, cache_config
from .read_rockstar_hlists import RockstarHlistReader

from . import catalog_version_info

from ..utils import convert_to_ndarray

from ..custom_exceptions import *

class HalotoolsRockstarCatalogReader(RockstarHlistReader):
    """ Class containing methods used to read raw ASCII data of the 
    Rockstar catalogs provided by Halotools. 

    The `HalotoolsRockstarCatalogReader` is only intended to work for 
    the frozen versions of the Rockstar ASCII data downloaded from hipacc 
    and uploaded to the Halotools server. If you are processing our own 
    Rockstar catalog, or if you downloaded an alternate version/redshift of the Rockstar 
    catalogs not managed by the Halotools developers, then there can be no 
    guarantees that the Rockstar catalogs have not changed, and so you should 
    instead process the ASCII data with the `RockstarHlistReader` class. 
    The only difference is that you will need to manually write the dtypes 
    of all the columsn you need, whereas here they are determined for you automatically 
    by specifing the string name of the column. 
    """

    def __init__(self, input_fname, simname, halo_finder, 
        catalog_version = 'most_recent_version', **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

            For your own versions of Rockstar run on these or other simulations, 
            use the more general `~halotools.sim_manager.RockstarHlistReader` class. 

        halo_finder : string
            Nickname of the halo-finder, e.g. ``rockstar`` or ``bdm``. 

        catalog_version : string, optional 
            String used to set which version of the catalogs are being read. 
            The choice for ``catalog_version`` determines the data types and names 
            of the columns of the Halotools-provided ASCII data. Must be a module 
            name stored in the ``catalog_version_info`` sub-package. 
            Default is ``most_recent_version``. 

        column_names_to_keep : list, optional 
            List of indices of columns that will be stored in the processed catalog. 
            Default behavior is to keep all columns. 

        row_cuts : list, optional 
            List of tuples used to define which rows of the ASCII data will be kept.
            Default behavior is to make no cuts. 

            The row-cut is determined from a list of tuples as follows. 
            Each element of the ``row_cuts`` list is a three-element tuple. 
            The first tuple element will be interpreted as the name of the column 
            upon which your cut is made. 

            For example, if you only want to keep halos 
            with :math:`M_{\\rm peak} > 1e10`, 
            then you would set row_cuts = [('halo_mpeak', 1e10, float("inf"))]. 
            Note that all Halotools-provided columns begin with `halo_`. 

            For any column-index appearing in ``row_cuts``, this index must also 
            appear in ``column_names_to_keep``: for purposes of good bookeeping, 
            you are not permitted to place a cut on a column that you do not keep. 

        header_char : str, optional
            String to be interpreted as a header line of the ascii hlist file. 
            Default is '#'. 

        Notes 
        -----
        Making even very conservative cuts on 
        either present-day or peak halo mass 
        can result in dramatic reductions in file size; 
        most halos in a typical raw catalog are right on the hairy edge 
        of the numerical resolution limit. 

        Also note that the processed halo catalogs 
        provided by Halotools *do* make cuts on rows. So if you run the 
        `RockstarHlistReader` with default settings on one of the raw 
        catalogs provided by Halotools, you will *not* get a  
        result that matches the corresponding processed catalog 
        provided by Halotools. 

        """
        try:
            assert simname in ('bolshoi', 'bolplanck', 'multidark', 'consuelo')
            self.simname = simname
        except AssertionError:
            msg = ("\nThe input ``simname`` must be one of of the following:\n"
                "'bolshoi', 'bolplanck', 'multidark' or 'consuelo'\n")
            raise HalotoolsError(msg)

        try:
            assert halo_finder in ('rockstar', 'bdm')
        except AssertionError:
            msg = ("\nThe input ``halo_finder`` must be either \n"
                "'rockstar' or 'bdm'\n")
        try:
            if halo_finder == 'bdm':
                assert self.simname == 'bolshoi'
        except AssertionError:
            msg = ("\nIf your input ``halo_finder`` is ``bdm``,\n"
                "your input ``halo_finder`` must be ``bolshoi``.\n")
            raise HalotoolsError(msg)
        self.halo_finder = halo_finder

        try:
            self.catalog_version = getattr(catalog_version_info, catalog_version)
        except:
            msg = ("\nThe input ``catalog_version`` must be a module \n"
                "stored in the ``catalog_version_info`` subpackage.\n")
            raise HalotoolsError(msg)

        dtype_name = 'dtype_' + self.simname + '_' + self.halo_finder
        self._full_dt = getattr(self.catalog_version, dtype_name)

        RockstarHlistReader.__init__(self, input_fname, **kwargs)

    def _interpret_input_row_cuts(self, **kwargs):
        """
        """

        try:
            input_row_cuts = kwargs['row_cuts']
            assert type(input_row_cuts) == list
            assert len(input_row_cuts) <= len(self.dt)
            for entry in input_row_cuts:
                assert type(entry) == tuple
                assert len(entry) == 3
                assert entry[0] in self._full_dt.names
        except KeyError:
            input_row_cuts = []
        except AssertionError:
            msg = ("\nInput ``row_cuts`` must be a list of 3-element tuples. \n"
                "The first entry is a string that must be a column name of the catalog.\n"
                "The available column names for this catalog are:\n\n")
            for ii, name in enumerate(self._full_dt.names):
                msg += name + ', '
                if ii % 4 == 0:
                    msg += "\n"
            msg += "\n\n"

            raise HalotoolsError(msg)

        names_array = np.array(self._full_dt.names)
        reformatted_row_cuts = []
        for row in input_row_cuts:
            index = np.where(names_array == row[0])[0][0]
            reformatted_row_cuts.append((index, row[1], row[2]))

        return reformatted_row_cuts


    def _interpret_input_dt(self, **kwargs):
        """
        """
        self._interpret_column_indices_to_keep(**kwargs)

        tmp = []
        for name in self.column_names_to_keep:
            tmp.append((name, self._full_dt[name]))
        self.dt = np.dtype(tmp)

    def _interpret_column_indices_to_keep(self, **kwargs):

        try:
            column_names_to_keep = kwargs['column_names_to_keep']
            assert type(column_names_to_keep) == list
            assert len(column_names_to_keep) <= len(self._full_dt)
            assert set(column_names_to_keep).issubset(set(self._full_dt.names))
        except KeyError:
            column_names_to_keep = list(self._full_dt.names)
        except AssertionError:
            msg = ("\nInput ``column_names_to_keep`` must be a list of strings\n"
                "storing the names of the ascii data columns to keep\n."
                "The available column names for this catalog are:\n\n")
            for ii, name in enumerate(self._full_dt.names):
                msg += name + ', '
                if ii % 4 == 0:
                    msg += "\n"
            msg += "\n\n"

            raise HalotoolsError(msg)
        self.column_names_to_keep = column_names_to_keep
        names_array = np.array(self._full_dt.names)
        self.column_indices_to_keep = []
        for name in self.column_names_to_keep:
            index = np.where(names_array == name)[0][0]
            self.column_indices_to_keep.append(index)







