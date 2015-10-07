# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
"""

__all__ = ['get_catalogs_dir']

supported_sim_list = ['bolshoi', 'bolplanck', 'consuelo', 'multidark']

import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
import warnings

from . import sim_defaults

from ..custom_exceptions import UnsupportedSimError, CatalogTypeError

def simname_is_supported(simname):
    """ Method returns a boolean for whether or not the input 
    simname corresponds to a Halotools-supported simulation, as determined by 
    the sub-classes of `~halotools.sim_manager.NbodySimulation`. 

    Parameters 
    ----------
    simname : string, optional  
        Nickname of the simulation. Currently supported simulations are 
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

    Returns 
    -------
    is_supported : bool 

    """
    return simname in supported_sim_list

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


def get_supported_halo_finders(input_simname):

    if input_simname not in supported_sim_list:
        raise UnsupportedSimError(input_simname)
    elif input_simname == 'bolshoi':
        return ['bdm', 'rockstar']
    else:
        return ['rockstar']

def get_catalogs_dir(**kwargs):
    """ Find the path to the subdirectory of the halotools cache directory 
    where `catalog_type` are stored. 
    
    If the directory doesn't exist, make it, then return the path. 

    Parameters
    ----------
    catalog_type : string, optional   
        String giving the type of catalog. 
        Should be 'particles', 'subhalos', or 'raw_halos'. 

    simname : string, optional  
        Nickname of the simulation. Currently supported simulations are 
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

    halo_finder : string, optional  
        Nickname of the halo-finder, e.g., `rockstar` or `bdm`. 

    external_cache_loc : string, optional 
        Absolute path to an alternative source of halo catalogs. 
        Method assumes that ``external_cache_loc`` is organized in the 
        same way that the normal Halotools cache is. Specifically: 

        * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

        * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

        * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

    Returns
    -------
    dirname : str
        Path to the halotools directory storing simulation data.

    """

    # Identify the root cache directory
    if 'external_cache_loc' in kwargs.keys():
        if os.path.isdir(kwargs['external_cache_loc']) is False:
            raise KeyError("Input external_cache_loc directory = %s \n Directory does not exist" % kwargs['external_cache_loc'])
        else:
            halotools_cache_dir = kwargs['external_cache_loc']
    else:
        homedir = _find_home()
        astropy_cache_dir = os.path.join(homedir, '.astropy', 'cache')
        defensively_create_subdir(astropy_cache_dir)

        halotools_cache_dir = os.path.join(astropy_cache_dir, 'halotools')
        defensively_create_subdir(halotools_cache_dir)


    if 'catalog_type' not in kwargs.keys():
        return halotools_cache_dir
    else:
        catalog_type = kwargs['catalog_type']

    acceptable_processed_halos_arguments = (
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

    if catalog_type in acceptable_processed_halos_arguments:
        subdir_name = 'halo_catalogs'
    elif catalog_type in acceptable_particles_arguments:
        subdir_name = 'particle_catalogs'
    elif catalog_type in acceptable_raw_halos_arguments:
        subdir_name = 'raw_halo_catalogs'
    else:
        raise CatalogTypeError(catalog_type)

   # Create the directory .astropy/cache/halotools/subdir_name
    catalog_type_dirname = os.path.join(halotools_cache_dir, subdir_name)
    defensively_create_subdir(catalog_type_dirname)

    # Now check to see if there exists a cache subdirectory for simname
    if 'simname' not in kwargs.keys():
        return catalog_type_dirname
    else:
        if kwargs['simname'] not in supported_sim_list:
            raise UnsupportedSimError(kwargs['simname'])

        simname_dirname = os.path.join(catalog_type_dirname, kwargs['simname'])
        defensively_create_subdir(simname_dirname)

        if 'halo_finder' not in kwargs.keys():
            return simname_dirname
        else:
            halo_finder_dirname = os.path.join(simname_dirname, kwargs['halo_finder'])
            defensively_create_subdir(halo_finder_dirname)
            return halo_finder_dirname


def processed_halo_tables_web_location(**kwargs):
    """ Method returns the web location where pre-processed 
    halo catalog binaries generated by, and for use with, 
    Halotools are stored. 

    Parameters 
    ----------
    simname : string, optional  
        Nickname of the simulation. Currently supported simulations are 
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

    halo_finder : string, optional  
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

    webroot = sim_defaults.processed_halo_tables_webloc

    if 'simname' not in kwargs.keys():
        return webroot
    else:
        simname = kwargs['simname']
        if 'halo_finder' not in kwargs.keys():
            return os.path.join(webroot, simname)
        else:
            halo_finder = kwargs['halo_finder']
            return os.path.join(os.path.join(webroot, simname), halo_finder)













