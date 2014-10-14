# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
Global scope functions have been modified from the 
paths methods of the astropy config sub-package.
"""
import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
import warnings

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
                raise
    elif not os.path.isdir(halotools_cache_dir):
        msg = 'Data cache directory {0} is not a directory'
        raise IOError(msg.format(halotools_cache_dir))

    return halotools_cache_dir


def get_catalogs_dir(catalog_type):
    """ Find the path to the halotools cache directory. 
    If the directory doesn't exist, make it, then return the path. 

    Returns
    -------
    dirname : str
        Path to the halotools directory storing processed halo catalogs.

    """
    if (catalog_type=='halo') or (catalog_type=='halos') or (catalog_type==None):
        subdir_name = 'halo_catalogs'
    elif (catalog_type=='particle') or (catalog_type=='particles'):
        subdir_name = 'particle_catalogs'
    else:
        raise TypeError("Input catalog_type must be either 'halos' or 'particles'")

    dirname = os.path.join(get_halotools_cache_dir(), subdir_name)

    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except OSError as e:
            if not os.path.exists(dirname):
                raise
    elif not os.path.isdir(dirname):
        msg = 'Data cache directory {0} is not a directory'
        raise IOError(msg.format(dirname))

    return dirname

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

def list_of_catalogs_in_cache(catalog_type='halos'):
    """ Returns a list of strings of filenames pointing to every 
    catalog currently in the cache directory"""

    from os import listdir
    from os.path import isfile, join

    catalog_path = get_catalogs_dir(catalog_type)

    return [ f.encode('utf-8') for f in listdir(catalog_path) if isfile(join(catalog_path,f)) ]







