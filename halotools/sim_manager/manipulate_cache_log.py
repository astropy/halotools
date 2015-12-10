""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_halo_table_cache_log_line', 
    'get_halo_table_cache_log_header', 
    'overwrite_halo_table_cache_log', 'read_halo_table_cache_log')

import os, tempfile
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table

import warnings
import datetime

try:
    import h5py
except ImportError:
    warn("\nMost of the functionality of the sim_manager sub-package"
    " requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import sim_defaults

from ..custom_exceptions import HalotoolsError

def get_halo_table_cache_log_fname():
    dirname = os.path.join(get_astropy_cache_dir(), 'halotools')
    return os.path.join(dirname, 'halo_table_cache_log.txt')

def get_halo_table_cache_log_header():
    return '# simname  redshift  halo_finder  version_name  fname  most_recent_use\n'

def get_formatted_halo_table_cache_log_line(simname, redshift, 
    halo_finder, version_name, fname):
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    formatted_line = (
        simname + '  ' + str(redshift) + '  ' + 
        halo_finder + '  ' + version_name + '  ' + fname + '  ' + timenow + '\n'
        )
    return formatted_line

def overwrite_halo_table_cache_log(new_log, **kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    with open(cache_fname, 'w') as f:
        header = get_halo_table_cache_log_header() 
        f.write(header)
        for entry in new_log:
            newline = get_formatted_halo_table_cache_log_line(
                entry['simname'], entry['redshift'], 
                entry['halo_finder'], entry['version_name'], entry['fname'])
            f.write(newline)

def read_halo_table_cache_log(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    return Table.read(cache_fname, format = 'ascii')


def update_halo_table_cache_log(simname, redshift, 
    halo_finder, version_name, fname, **kwargs):
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    pass




def identify_halo_catalog_fname(**kwargs):
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    if 'fname' in kwargs:
        if not os.path.isfile(kwargs['fname']):
            msg = ("\nThe requested filename ``" + kwargs['fname'] + "`` does not exist.\n")
            raise HalotoolsError(msg)
        else:
            update_halo_table_cache_log(kwargs['fname'])
            return kwargs['fname']
    else:
        log = read_cache_memory_log(cache_fname)
        mask = np.ones(len(log), dtype=bool)

        catalog_attrs = ('simname', 'redshift', 'halo_finder', 'version_name')
        for key in catalog_attrs:
            try:
                attr_mask = log[key] == kwargs[key]
                mask *= attr_mask
            except KeyError:
                pass
        matching_catalogs = log[mask]

        if len(matching_catalogs) == 0:
            auto_detect_halo_table(**kwargs)
        elif len(matching_catalogs) == 1:
            return 


def auto_detect_halo_table(**kwargs):
    raise HalotoolsError("The auto_detect_halo_table function is not implemented yet.")

def verify_halo_table_cache_existence(cache_fname):
    """
    """

    if not os.path.isfile(cache_fname):
        msg = ("\nThe file " + cache_fname + "\ndoes not exist. "
            "This file serves as a log for all the halo catalogs you use with Halotools.\n"
            "If you have not yet downloaded the initial halo catalog,\n"
            "you should do so now following the ``Getting Started`` instructions on "
            "http://halotools.readthedocs.org\nIf you have already taken this step,\n"
            "then your halo table cache log has been deleted,\nin which case you should"
            "execute the following script:\n"
            "halotools/scripts/rebuild_halo_table_cache_log.py\n")
        raise HalotoolsError(msg)

def verify_halo_table_cache_header(cache_fname):
    """
    """
    verify_halo_table_cache_existence(cache_fname)

    correct_header = get_halo_table_cache_log_header()
    with open(cache_fname, 'r') as f:
        actual_header = f.readline()

    if correct_header != actual_header:
        msg = ("\nThe file " + cache_fname + 
            "serves as a log for all the halo catalogs you use with Halotools.\n"
            "The correct header that should be in this file is \n"
            + correct_header + "\nThe actual header currently stored in this file is \n"
            + actual_header + "\nTo resolve your error, try opening the log file "
            "with a text editor and replacing the current line with the correct one.\n")
        raise HalotoolsError(msg)

def verify_halo_table_cache_log_columns(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    verify_halo_table_cache_existence(cache_fname = cache_fname)
    verify_halo_table_cache_header(cache_fname = cache_fname)

    try:
        log = kwargs['log']
    except KeyError:
        log = read_halo_table_cache_log(cache_fname = cache_fname)

    correct_header = get_halo_table_cache_log_header()
    expected_key_set = set(correct_header.strip().split()[1:])
    log_key_set = set(log.keys())
    try:
        assert log_key_set == expected_key_set
    except AssertionError:
        cache_fname = get_halo_table_cache_log_fname()
        msg = ("The file " + cache_fname + 
            "\nkeeps track of the halo catalogs"
            "you use with Halotools.\n"
            "This file appears to be corrupted.\n"
            "Please visually inspect this file to ensure it has not been "
            "accidentally overwritten. \n"
            "Then store a backup of this file and execute the following script:\n"
            "halotools/scripts/rebuild_halo_table_cache_log.py\n"
            "If this does not resolve the error you are encountering,\n"
            "and if you have been using halo catalogs stored on some external disk \n"
            "or other non-standard location, you may try manually adding \n"
            "the appropriate lines to the cache log.\n"
            "Please contact the Halotools developers if the issue persists.\n")
        raise HalotoolsError(msg)

def verify_cache_log(**kwargs):

    verify_halo_table_cache_existence(**kwargs)
    verify_halo_table_cache_header(**kwargs)
    verify_halo_table_cache_log_columns(**kwargs)



def check_metadata_consistency(cache_log_entry, **kwargs):
    """
    """
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed in order to use "
            "the Halotools halo catalog cache system.\n")

    halo_table_fname = cache_log_entry['fname']
    f = h5py.File(halo_table_fname)

    for key, requested_attr in kwargs.iteritems():
        try:
            attr_of_cached_catalog = f.attrs[key]
            assert attr_of_cached_catalog == requested_attr
        except KeyError:
            msg = ("\nThe halo table stored in \n``"+halo_table_fname+"\n"
                "does not have metadata stored for the ``"+key+"`` attribute\n"
                "and so some self-consistency checks cannot be performed.\n"
                "If you are seeing this message while attempting to load a \n"
                "halo catalog provided by Halotools, please submit a bug report on GitHub.\n"
                "If you are using your own halo catalog that you have stored \n"
                "in the Halotools cache yourself, you should consider adding this metadata\n"
                "to the hdf5 file as one of the keys of the .attrs file attribute.\n")
            warnings.warn(msg)
        except AssertionError:
            msg = ("\nThe halo table stored in \n``"+halo_table_fname+"\n"
                "has the value ``"+attr_of_cached_catalog+"`` stored as metadata for the \n"
                "``"+key+"`` attribute.\nThis is inconsistent with the "
                "``"+requested_attr+"`` value that you requested.\n"
                "If you are seeing this message while attempting to load a \n"
                "halo catalog provided by Halotools, please submit a bug report on GitHub.\n"
                "If you are using your own halo catalog that you have stored \n"
                "in the Halotools cache yourself, then you have "
                "attempted to access a halo catalog \nby requesting a value for "
                "the ``"+key+"`` attribute that is inconsistent with the stored value.\n")
            raise HalotoolsError(msg)

    f.close()



