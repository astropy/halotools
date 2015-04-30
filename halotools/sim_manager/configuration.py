# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
"""

__all__ = ['get_halotools_cache_dir','get_catalogs_dir','list_of_catalogs_in_cache']


import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
import warnings

from . import sim_defaults

def get_halotools_cache_dir():
    """ Find the path to the halotools cache directory. 
    If the directory doesn't exist, make it, then return the path. 

    Returns
    -------
    dir : str
        Path to the halotools cache directory.

    """

    halotools_cache_dir = os.path.join(get_astropy_cache_dir(), 'halotools')

    if not os.path.exists(halotools_cache_dir):
        try:
            os.mkdir(halotools_cache_dir)
        except OSError as e:
            if not os.path.exists(halotools_cache_dir):
                raise IOError("Unable to create a cache directory for Halotools catalogs")
    elif not os.path.isdir(halotools_cache_dir):
        msg = 'Data cache directory {0} is not a directory'
        raise IOError(msg.format(halotools_cache_dir))

    return halotools_cache_dir


def get_catalogs_dir(catalog_type):
    """ Find the path to the halotools cache directory. 
    If the directory doesn't exist, make it, then return the path. 

    Parameters
    ----------
    catalog_type : string, optional
        String giving the type of catalog. 
        Should be 'particles', 'subhalos', or 'raw_halos'. 

    Returns
    -------
    dirname : str
        Path to the halotools directory storing simulation data.

    """
    acceptable_halos_arguments = (
        ['subhalos', 'subhalo', 'halo', 'halos', 
        'halo_catalogs', 'subhalo_catalogs', 'subhalo_catalog', 'halo_catalog',
        'halos_catalogs', 'subhalos_catalogs', 'subhalos_catalog', 'halos_catalog']
        )
    acceptable_particles_arguments = (
        ['particle', 'particles', 'particle_catalog', 'particle_catalogs', 
        'particles_catalog', 'particles_catalogs']
        )
    acceptable_raw_halos_arguments = (
        ['raw_halos', 'raw_subhalos', 'raw_halo', 'raw_subhalo', 
        'raw_halos_catalog', 'raw_subhalos_catalog', 'raw_halo_catalog', 'raw_subhalo_catalog', 
        'raw_halos_catalogs', 'raw_subhalos_catalogs', 'raw_halo_catalogs', 'raw_subhalo_catalogs']
        )

    if catalog_type in acceptable_halos_arguments:
        subdir_name = 'halo_catalogs'
        default_cache_dir = sim_defaults.processed_halocat_cache_dir
    elif catalog_type in acceptable_particles_arguments:
        subdir_name = 'particle_catalogs'
        default_cache_dir = sim_defaults.particles_cache_dir
    elif catalog_type in acceptable_raw_halos_arguments:
        subdir_name = 'raw_halo_catalogs'
        default_cache_dir = sim_defaults.raw_halocat_cache_dir
    else:
        raise TypeError("Input catalog_type must be either 'subhalos' or 'particles'")

    # Check to see whether we are using the package default or user-provided cache directory
    if default_cache_dir != 'pkg_default':
        # Default cache dir has been over-ridden
        # Check to make sure that the provided dirname is actually a directory
        if not os.path.isdir(default_cache_dir):
            errmsg = 'Cache dirname ' + default_cache_dir + 'stored in '
            'sim_defaults module is not a directory'
            raise IOError(errmsg)
        else:
            return default_cache_dir
    else:
        # Use the cache directory provided by the package
        dirname = os.path.join(get_halotools_cache_dir(), subdir_name)
        if not os.path.exists(dirname):
            try:
                os.mkdir(dirname)
            except OSError as e:
                if not os.path.exists(dirname):
                    raise IOError("No path exists for the requested catalog")
        elif not os.path.isdir(dirname):
            msg = 'Data cache directory {0} is not a directory'
            raise IOError(msg.format(dirname))

        return dirname


def list_of_catalogs_in_cache(catalog_type='subhalos'):
    """ Returns a list of strings of filenames pointing to every 
    catalog currently in the cache directory.

    Parameters
    ----------
    catalog_type : string, optional
        String giving the type of catalog. Should be 'particles' or 'subhalos'.

    Returns
    -------
    file_list : array_like
        List of strings. Each entry corresponds to a filename of a catalog in 
        the cache directory.

    """

    from os import listdir
    from os.path import isfile, join

    catalog_path = get_catalogs_dir(catalog_type)

    return [ f.encode('utf-8') for f in listdir(catalog_path) if isfile(join(catalog_path,f)) ]







