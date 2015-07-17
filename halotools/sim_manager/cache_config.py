# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
"""

__all__ = (
    ['get_halotools_cache_dir','get_catalogs_dir']
    )

import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
import warnings

from . import sim_defaults
from . import supported_sims

def get_halotools_cache_dir():
    """ Find the path to the root halotools cache directory. 

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

def defensively_create_subdir(dirname):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except OSError as e:
            if not os.path.exists(dirname):
                raise IOError("OS Error during creation of path %s" % dirname)
    elif not os.path.isdir(dirname):
        msg = 'Pathname {0} is not a directory'
        raise IOError(msg.format(dirname))

def get_supported_simnames():
    class_list = supported_sims.__all__
    parent_class = supported_sims.NbodySimulation

    supported_simnames = []
    for clname in class_list:
        clobj = getattr(supported_sims, clname)
        if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
            clinst = clobj()
            supported_simnames.append(clinst.simname)

    return list(set(supported_simnames))

def get_supported_halo_finders(input_simname):
    class_list = supported_sims.__all__
    parent_class = supported_sims.HaloCat

    supported_halo_finders = []
    for clname in class_list:
        clobj = getattr(supported_sims, clname)
        if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
            clinst = clobj()
            if clinst.simname == input_simname:
                supported_halo_finders.append(clinst.halo_finder)

    return list(set(supported_halo_finders))

def cache_subdir_for_simulation(parentdir, simname):
    simulation_cache = os.path.join(parentdir, simname)
    if os.path.exists(simulation_cache):
        return simulation_cache
    else:
        supported_sims = get_supported_simnames()
        if simname in supported_sims:
            return simulation_cache
        else:
            raise IOError("It is not permissible to create a subdirectory of "
                "Halotools cache \nfor simulations which have no class defined in "
                "the halotools/sim_manager/supported_sims module. \n")

def cache_subdir_for_halo_finder(parentdir, simname, halo_finder):
    halo_finder_cache = os.path.join(parentdir, halo_finder)
    if os.path.exists(halo_finder_cache):
        return halo_finder_cache
    else:
        supported_halo_finders = get_supported_halo_finders(simname)
        if halo_finder in supported_halo_finders:
            return halo_finder_cache
        else:
            raise IOError("It is not permissible to create a subdirectory of "
                "Halotools cache \nfor a combination of "
                "simulation + halo-finder which has no corresponding class defined in "
                "the halotools/sim_manager/supported_sims module. \n")


def get_catalogs_dir(catalog_type, **kwargs):
    """ Find the path to the subdirectory of the halotools cache directory 
    where `catalog_type` are stored. 
    
    If the directory doesn't exist, make it, then return the path. 

    Parameters
    ----------
    catalog_type : string
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
        raise TypeError("Input catalog_type must be either 'raw_halos', 'halos', or 'particles'")

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
        # Create the directory .astropy/cache/halotools/subdir_name
        catalog_type_dirname = os.path.join(get_halotools_cache_dir(), subdir_name)
        defensively_create_subdir(catalog_type_dirname)

        # Now check to see if there exists a cache subdirectory for simname
        if 'simname' not in kwargs.keys():
            return catalog_type_dirname
        else:
            simname_dirname = cache_subdir_for_simulation(catalog_type_dirname, kwargs['simname'])
            defensively_create_subdir(simname_dirname)

            if 'halo_finder' not in kwargs.keys():
                return simname_dirname
            else:
                halo_finder_dirname = (
                    cache_subdir_for_halo_finder(
                        simname_dirname, kwargs['simname'], kwargs['halo_finder'])
                    )
                defensively_create_subdir(halo_finder_dirname)
                return halo_finder_dirname


def processed_halocats_web_location(**kwargs):
    """ Method returns the web location where pre-processed 
    halo catalog binaries generated by, and for use with, 
    Halotools are stored. 

    Parameters 
    ----------
    simname : string, optional keyword argument 
        Nickname of the simulation, e.g., `bolshoi`. 

    halo_finder : string, optional keyword argument 
        Nickname of the halo-finder, e.g., `rockstar` or `bdm`. 

    Returns 
    -------
    result : string 
        If no arguments are passed, method returns the root web directory 
        of the Halotools binaries, `root_web_dir`. 
        If `simname` is passed, method returns `root_web_dir/simname`. 
        If `simname` and `halo_finder` are both passed, 
        method returns `root_web_dir/simname/halo_finder`. 
    """

    webroot = sim_defaults.processed_halocats_webloc

    if 'simname' not in kwargs.keys():
        return webroot
    else:
        simname = kwargs['simname']
        if 'halo_finder' not in kwargs.keys():
            return os.path.join(webroot, simname)
        else:
            halo_finder = kwargs['halo_finder']
            return os.path.join(os.path.join(webroot, simname), halo_finder)


def enable_cache_access_during_pytest(func):
    """ Decorator used by test suite functions to permit access to the 
    Halotools cache directory. 
    """
    def wrapper(*args, **kwargs):

        xch = os.environ.get('XDG_CACHE_HOME')
        if xch is not None:
            del os.environ['XDG_CACHE_HOME']

        os.environ['XDG_CACHE_HOME'] = get_astropy_cache_dir()
        result = func(*args, **kwargs)
        os.environ['XDG_CACHE_HOME'] = xch

        return result

    return wrapper












