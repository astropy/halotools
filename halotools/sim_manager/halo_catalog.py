import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import sim_defaults, catalog_manager

from ..utils.array_utils import custom_len
from ..custom_exceptions import *


__all__ = ('HaloCatalog',)


class HaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, simname=sim_defaults.default_simname, 
        halo_finder=sim_defaults.default_halo_finder, 
        redshift = sim_defaults.default_redshift, dz_tol = 0.05, 
        preload_halo_table = False):
        pass




