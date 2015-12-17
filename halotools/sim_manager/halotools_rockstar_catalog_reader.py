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

from .catalog_column_info import * 

from ..utils import convert_to_ndarray

from ..custom_exceptions import *

class HalotoolsRockstarCatalogReader(RockstarHlistReader):
    """ Class containing methods used to read raw ASCII data generated with Rockstar. 
    """

    def __init__(self, input_fname, dt, simname, **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        dt : Numpy dtype object 
            The ``dt`` argument instructs the reader how to interpret the 
            columns stored in the ASCII data. 

        simname : string 
            Nickname of the simulation. Currently supported simulations are 
            ``bolshoi``, ``bolplanck``, ``consuelo`` and ``multidark``. 
            For your own versions of Rockstar run on these or other simulations, 
            use the more general `~halotools.sim_manager.RockstarHlistReader` class. 

        column_indices_to_keep : list, optional 
            List of indices of columns that will be stored in the processed catalog. 
            Default behavior is to keep all columns. 

        row_cuts : list, optional 
            List of tuples used to define which rows of the ASCII data will be kept.
            Default behavior is to make no cuts. 

            The row-cut is determined from a list of tuples as follows. 
            Each element of the ``row_cuts`` list is a three-element tuple. 
            The first tuple element will be interpreted as the index of the column 
            upon which your cut is made. 
            The column-indexing convention is C-like, 
            so that the first column has column-index = 0. 
            The second and third tuple elements will be interpreted 
            as lower and upper bounds on this column, respectively. 

            For example, if you only want to keep halos 
            with :math:`M_{\\rm peak} > 1e10`, 
            and :math:`M_{\\rm peak}` is the tenth column of the ASCII file, 
            then you would set row_cuts = [(9, 1e10, float("inf"))]. 

            For any column-index appearing in ``row_cuts``, this index must also 
            appear in ``column_indices_to_keep``: for purposes of good bookeeping, 
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
        self.simname = simname

        RockstarHlistReader.__init__(self, input_fname, dt, **kwargs)

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
                assert entry[0] in self.column_indices_to_keep
        except KeyError:
            input_row_cuts = []
        except AssertionError:
            msg = ("\nInput ``row_cuts`` must be a list of 3-element tuples. \n"
                "The first entry is an integer that will be interpreted as the \n"
                "column-index upon which a cut is made.\n"
                "All column indices must appear in the input ``column_indices_to_keep``.\n"
                )
            raise HalotoolsError(msg)

        return input_row_cuts



