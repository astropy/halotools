# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
Global scope functions have been modified from the 
paths methods of the astropy config sub-package.
"""
import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir


def get_halotools_catalog_dir():
    """ Find the path to the halotools cache directory. 
    If the directory doesn't exist, make it, then return the path. 

    Returns
    -------
    dir : str
        The path to the halotools cache directory.

    """

    halotools_catalog_dir = os.path.join(get_astropy_cache_dir(), 'halotools_catalogs')

    if not os.path.exists(halotools_catalog_dir):
        try:
            os.mkdir(halotools_catalog_dir)
        except OSError as e:
            if not os.path.exists(halotools_catalog_dir):
                raise
    elif not os.path.isdir(halotools_catalog_dir):
        msg = 'Data cache directory {0} is not a directory'
        raise IOError(msg.format(halotools_catalog_dir))

    return halotools_catalog_dir

def get_local_filename_from_remote_url(remote_url):
    """
    Helper function used to test whether catalog data has already been 
    downloaded and stored in the astropy cache directory.

    Parameters
    ----------
    remote_url : string 
        String giving the url of the catalog to be downloaded.

    Returns
    -------
    local_filename : string
        String giving the filename where the catalog is stored locally.
        The directory will be the default astropy cache download directory.
        If None, then there is no catalog present in the local cache directory 
        that corresponds to the input remote_url.

    """

    url_key = remote_url

    dldir, urlmapfn = get_download_cache_locs()
    with open_shelve(urlmapfn, True) as url2hash:
        if url_key in url2hash:
            local_filename = url2hash[url_key]
        else:
            local_filename = None

    return local_filename

def list_of_catalogs_in_cache():
    """ Returns a list of strings of filenames pointing to every 
    pre-processed halo catalog currently in the cache directory"""

    from os import listdir
    from os.path import isfile, join

    catalog_path = get_halotools_catalog_dir()

    return [ f.encode('utf-8') for f in listdir(catalog_path) if isfile(join(catalog_path,f)) ]


class Config(object):
    """ Configuration object providing standardization of 
    a variety of cross-package settings. """

    def __init__(self):

        self.catalog_pathname = self.getCatalogDir()
        self.hearin_url="http://www.astro.yale.edu/aphearin/Data_files/"

    # Returns the path to this code file
    def getCodeDir(self):
        return os.path.dirname(os.path.realpath(__file__))

    # Returns the path to the directory storing simulation data
    def getCatalogDir(self):
        return os.path.dirname(os.path.realpath(__file__))+'/CATALOGS/'


    def getSimulationFilename(self,simulation_name,scale_factor,halo_finder,use_subhalos):

        if use_subhalos==False:
            fname = (simulation_name+'_a'+
                str(scale_factor)+'_'+halo_finder+'_host_halos.fits' )
        else:
            fname = (simulation_name+'_a'+
                str(scale_factor)+'_'+halo_finder+'_subhalos.fits' )

        return fname

    def getParticleFilename(self,simulation_name,scale_factor,num_ptcl):

        fname = simulation_name+'_'+num_ptcl+'_particles_a'+str(scale_factor)+'.fits'

        return fname













