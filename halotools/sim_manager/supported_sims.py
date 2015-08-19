# -*- coding: utf-8 -*-

import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 

try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except ImportError:
    HAS_SOUP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import posixpath
import urlparse

from . import sim_defaults, catalog_manager

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import custom_len

from ..custom_exceptions import UnsupportedSimError, CatalogTypeError, HalotoolsCacheError

from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.extern import six

from astropy import cosmology
from astropy import units as u
from astropy.table import Table


__all__ = (
    ['NbodySimulation', 'Bolshoi', 'BolPlanck', 'MultiDark', 'Consuelo', 
    'HaloCatalog', 'retrieve_simclass']
    )


######################################################
########## Simulation classes defined below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class NbodySimulation(object):
    """ Abstract base class for any object used as a container for 
    simulation specs. 
    """

    def __init__(self, simname, Lbox, particle_mass, num_ptcl_per_dim, 
        softening_length, initial_redshift, cosmology):
        """
        Parameters 
        -----------
        simname : string 
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        Lbox : float
            Size of the simulated box in Mpc with h=1 units. 

        particle_mass : float
            Mass of the dark matter particles in Msun with h=1 units. 

        num_ptcl_per_dim : int 
            Number of particles per dimension. 

        softening_length : float 
            Softening scale of the particle interactions in kpc with h=1 units. 

        initial_redshift : float 
            Redshift at which the initial conditions were generated. 

        cosmology : object 
            `astropy.cosmology` instance giving the cosmological parameters 
            with which the simulation was run. 

        """
        self.simname = simname
        self.Lbox = Lbox
        self.particle_mass = particle_mass
        self.num_ptcl_per_dim = num_ptcl_per_dim
        self.softening_length = softening_length
        self.initial_redshift = initial_redshift
        self.cosmology = cosmology

        self._attrlist = (
            ['simname', 'Lbox', 'particle_mass', 'num_ptcl_per_dim',
            'softening_length', 'initial_redshift', 'cosmology']
            )

        self._catman = catalog_manager.CatalogManager()

class Bolshoi(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology 
    with Lbox = 250 Mpc/h and particle mass of ~1e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/bolshoi. 
    """

    def __init__(self):

        super(Bolshoi, self).__init__(simname = 'bolshoi', Lbox = 250., 
            particle_mass = 1.35e8, num_ptcl_per_dim = 2048, 
            softening_length = 1., initial_redshift = 80., cosmology = cosmology.WMAP5)

class BolPlanck(NbodySimulation):
    """ Cosmological N-body simulation of Planck 2013 cosmology 
    with Lbox = 250 Mpc/h and 
    particle mass of ~1e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/bolshoip-project/bolshoip/. 
    """

    def __init__(self):

        super(BolPlanck, self).__init__(simname = 'bolplanck', Lbox = 250., 
            particle_mass = 1.35e8, num_ptcl_per_dim = 2048, 
            softening_length = 1., initial_redshift = 80., cosmology = cosmology.Planck13)


class MultiDark(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5 cosmology 
    with Lbox = 1Gpc/h and particle mass of ~1e10 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://www.cosmosim.org/cms/simulations/multidark-project/mdr1. 
    """

    def __init__(self):

        super(MultiDark, self).__init__(simname = 'multidark', Lbox = 1000., 
            particle_mass = 8.721e9, num_ptcl_per_dim = 2048, 
            softening_length = 7., initial_redshift = 65., cosmology = cosmology.WMAP5)

class Consuelo(NbodySimulation):
    """ Cosmological N-body simulation of WMAP5-like cosmology 
    with Lbox = 420 Mpc/h and particle mass of 4e8 Msun/h. 

    For a detailed description of the 
    simulation specs, see http://lss.phy.vanderbilt.edu/lasdamas/simulations.html. 
    """

    def __init__(self):

        super(Consuelo, self).__init__(simname = 'consuelo', Lbox = 420., 
            particle_mass = 1.87e9, num_ptcl_per_dim = 1400, 
            softening_length = 8., initial_redshift = 99., cosmology = cosmology.WMAP5)


def retrieve_simclass(simname):
    """
    Parameters 
    ----------
    simname : string, optional
        Nickname of the simulation. Currently supported simulations are 
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

    Returns 
    -------
    simclass : object
        Appropriate sub-class of `~halotools.sim_manager.NbodySimulation`. 
    """
    if simname == 'bolshoi':
        return Bolshoi 
    elif simname == 'bolplanck':
        return BolPlanck 
    elif simname == 'multidark':
        return MultiDark
    elif simname == 'consuelo':
        return Consuelo 
    else:
        raise UnsupportedSimError(simname)


######################################################
########## Halo Catalog classes defined below ########
######################################################

class HaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, simname=sim_defaults.default_simname, 
        halo_finder=sim_defaults.default_halo_finder, 
        redshift = sim_defaults.default_redshift, dz_tol = 0.05, 
        preload_halo_table = False):
        """
        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar`` or ``bdm``. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        redshift : float, optional
            Redshift of the desired catalog. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.05. 

        preload_halo_table : bool, optional 
            If True, the `HaloCatalog` class will automatically retrieve the halo 
            table from disk upon instantiation. Default is False. 

        Examples 
        ---------
        The default halo catalog can be loaded into memory by calling `HaloCatalog` with no arguments: 

        >>> default_halocat = HaloCatalog() # doctest: +SKIP

        By modifying the `~halotools.sim_manager.sim_defaults` module, 
        you can set your default catalog to whatever simulation, 
        halo-finder and redshift you wish; simply change 
        ``default_simname``, ``default_halo_finder`` and ``default_redshift``, respectively. 

        To access the table of halos bound to your halo catalog:

        >>> halo_table = default_halocat.halo_table # doctest: +SKIP

        As with any `~astropy.table.Table` object, you can access the column data you want 
        via the corresponding field name: 

        >>> mass = halo_table['halo_mvir'] # doctest: +SKIP

        To see what columns are available for a given halo catalog:

        >>> print(halo_table.keys()) # doctest: +SKIP

        Most of the pre-processed `HaloCatalog` objects provided by Halotools 
        also come bundled with a random downsampling of 
        roughly a million dark matter particles. To access the 
        `~astropy.table.Table` storing this data:

        >>> particles = default_halocat.ptcl_table # doctest: +SKIP

        """
        self.catman = catalog_manager.CatalogManager()

        fname, closest_redshift = self._retrieve_closest_halo_table_fname(
            simname, halo_finder, redshift)
        if abs(closest_redshift - redshift) > dz_tol:
            msg = ("Your input cache directory does not contain a halo catalog \n" 
                "within %.3f of your input redshift = %.3f.\n"
                "For the ``%s`` simulation and ``%s`` halo-finder, \n" 
                "the catalog with the closest redshift in your cache has redshift = %.3f.\n"
                "If that is the catalog you want, simply call the HaloCatalog class constructor \n"
                "using the ``redshift`` keyword argument set to %.3f. \nOtherwise, choose a different "
                "halo catalog from your cache,\nor use the CatalogManager to download the catalog you need.\n")
            raise HalotoolsCacheError(msg % (dz_tol, redshift, 
                simname, halo_finder, closest_redshift, closest_redshift))
        else:
            self.processed_halo_table_fname = fname
            simclass = retrieve_simclass(simname)
            simobj = simclass()
            for attr in simobj._attrlist:
                setattr(self, attr, getattr(simobj, attr))
            self.redshift = closest_redshift
            self.halo_finder = halo_finder 
            self.dtype_ascii, self.header_ascii = sim_defaults.return_dtype_and_header(
                self.simname, self.halo_finder)
            self._check_catalog_self_consistency(fname, closest_redshift)

        if preload_halo_table is True:
            self._halo_table = Table.read(self.processed_halo_table_fname, path='data')


    @property 
    def halo_table(self):
        """
        `~astropy.table.Table` object storing a catalog of dark matter halos. 
        """
        if hasattr(self, '_halo_table'):
            return self._halo_table
        else:
            self._halo_table = Table.read(self.processed_halo_table_fname, path='data')
            return self._halo_table

    @property 
    def host_halos(self):
        if hasattr(self, '_halo_table'):
            mask = self._halo_table['halo_hostid'] == self._halo_table['halo_id']
            return self._halo_table[mask]
        else:
            self._halo_table = Table.read(self.processed_halo_table_fname, path='data')
            mask = self._halo_table['halo_hostid'] == self._halo_table['halo_id']
            return self._halo_table[mask]        

    @property 
    def ptcl_table(self):
        """
        `~astropy.table.Table` object storing a catalog of dark matter particles. 
        """
        if hasattr(self, '_ptcl_table'):
            return self._ptcl_table
        else:
            fname, closest_redshift = self._retrieve_closest_ptcl_table_fname()
            if abs(closest_redshift - self.redshift) > 0.01:
                msg = ("Your input cache directory does not contain a particle catalog \n" 
                    "that matches the redshift = %.3f of your halo catalog.\n"
                    "For the ``%s`` simulation, the particle catalog with "
                    "the closest redshift in your cache has z = %.3f.\n"
                    "\nTo see whether a matching ptcl_table is available for download, \n"
                    "use the ``closest_catalog_on_web`` method of the CatalogManager. \n"
                    "If there exists a matching catalog, you can download it with the "
                    "download_ptcl_table method of the CatalogManager.\n")
                raise HalotoolsCacheError(msg % (self.redshift, self.simname, closest_redshift))
            else:
                self.ptcl_table_fname = fname
                self._ptcl_table = Table.read(self.ptcl_table_fname, path='data')
            return self._ptcl_table

        ### Attributes that still need to be implemented: 
        # self.version,self.orig_data_source, etc. 
        # Also should implement some slick way to describe all columns in plain English 

    def _retrieve_closest_halo_table_fname(self, simname, halo_finder, redshift):
        """ Method uses the CatalogManager to return a halo catalog filename. 
        """
        if sim_defaults.default_cache_location == 'pkg_default':
            fname, closest_redshift = self.catman.closest_catalog_in_cache(
                catalog_type = 'halos', 
                simname = simname, 
                halo_finder = halo_finder,
                desired_redshift = redshift)
        else:
            fname, closest_redshift = self.catman.closest_catalog_in_cache(
                catalog_type = 'halos', 
                simname = simname, 
                halo_finder = halo_finder,
                desired_redshift = redshift, 
                external_cache_loc = sim_defaults.default_cache_location)
        
        return fname, closest_redshift

    def _retrieve_closest_ptcl_table_fname(self):
        """ Method uses the CatalogManager to return a particle catalog filename. 
        """
        if sim_defaults.default_cache_location == 'pkg_default':
            fname, closest_redshift = self.catman.closest_catalog_in_cache(
                catalog_type = 'particles', 
                simname = self.simname, 
                desired_redshift = self.redshift)
        else:
            fname, closest_redshift = self.catman.closest_catalog_in_cache(
                catalog_type = 'particles', 
                simname = self.simname, 
                desired_redshift = self.redshift, 
                external_cache_loc = sim_defaults.default_cache_location)

        return fname, closest_redshift

    def _check_catalog_self_consistency(self, fname, closest_redshift):

        msg = ("\nInconsistency between the %s in the metadata of the hdf5 file "
            "and the %s inferred from its filename.\n"
            "This indicates a bug during the generation of the hdf5 file storing the catalog.")

        import h5py
        f = h5py.File(fname)
        if abs(float(f.attrs['redshift']) - closest_redshift) > 0.01:
            raise HalotoolsIOError(msg % ('redshift', 'redshift'))

        if f.attrs['simname'] != self.simname:
            raise HalotoolsIOError(msg % ('simname', 'simname'))

        if f.attrs['halo_finder'] != self.halo_finder:
            raise HalotoolsIOError(msg % ('halo_finder', 'halo_finder'))

        self.cuts_description = f.attrs['cuts_description']

        f.close()


