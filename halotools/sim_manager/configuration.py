# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
"""

__all__ = (
    ['get_halotools_cache_dir','get_catalogs_dir','infer_simulation_from_fname']
    )


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


def get_catalogs_dir(catalog_type, **kwargs):
    """ Find the path to the halotools cache directory. 
    If the directory doesn't exist, make it, then return the path. 

    Parameters
    ----------
    catalog_type : string, optional
        String giving the type of catalog. 
        Should be 'particles', 'subhalos', or 'raw_halos'. 

    simname : string, optional keyword argument 
        Nickname of the simulation, e.g., `bolshoi`. 

    halo_finder : string, optional keyword argument 
        Nickname of the halo-finder, e.g., `rockstar` or `bdm`. 

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


    def defensively_create_dir(dirname):
        if not os.path.exists(dirname):
            try:
                os.mkdir(dirname)
            except OSError as e:
                if not os.path.exists(dirname):
                    raise IOError("OS Error during creation of path %s" % dirname)
        elif not os.path.isdir(dirname):
            msg = 'Pathname {0} is not a directory'
            raise IOError(msg.format(dirname))


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
        catalog_type_dirname = os.path.join(get_halotools_cache_dir(), subdir_name)
        defensively_create_dir(catalog_type_dirname)
        if 'simname' not in kwargs.keys():
            return catalog_type_dirname
        else:
            simname_dirname = os.path.join(catalog_type_dirname, kwargs['simname'])
            defensively_create_dir(simname_dirname)
            if 'halo_finder' not in kwargs.keys():
                return simname_dirname
            else:
                halo_finder_dirname = os.path.join(simname_dirname, kwargs['halo_finder'])
                defensively_create_dir(halo_finder_dirname)
                return halo_finder_dirname


def infer_simulation_from_fname(fname):

    halocat_path = os.path.dirname(fname)
    halotools_cache_dir = get_halotools_cache_dir()
    subdir_list = [x[0] for x in os.walk(halotools_cache_dir)]
    if halocat_path not in subdir_list:
        print("Input fname is not a subdirectory of "
            "the Halotools cache: \n cannot infer simulations from external sources")
        return None
    else:
        halo_finder = os.path.basename(halocat_path)
        simulation_path = os.path.abspath(os.path.join(halocat_path, os.pardir))
        simname = os.path.basename(simulation_path)
        return simname, halo_finder





