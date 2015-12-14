""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_halo_table_cache_log_line', 
    'get_halo_table_cache_log_header', 
    'overwrite_halo_table_cache_log', 'read_halo_table_cache_log')

import os, tempfile, fnmatch
from copy import copy, deepcopy 

from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table

from warnings import warn 
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


def load_cached_halo_table_from_fname(fname, **kwargs):
    """
    """
    # If a cache location is explicitly specified, 
    # use it instead of the standard location. 
    # This facilitate unit-testing
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    try:
        assert os.path.isfile(fname)
    except AssertionError:
        msg = ("\nThe requested filename ``" + fname + "`` does not exist.\n")
        raise HalotoolsError(msg)

    log = read_cache_memory_log(cache_fname)
    mask = log['fname'] == fname
    matching_catalogs = log[mask]

    if len(matching_catalogs) == 0:
        msg = ("The filename you requested ,``"+fname+"`` \nexists but it does not appear"
            "in the halo table cache log,\n"
            +cache_fname+"\nYou can add this catalog to your cache log by calling the\n"
            "``update_halo_table_cache_log`` function.\n")
        warn(msg)
    elif len(matching_catalogs) == 1:
        return Table.read(matching_catalogs['fname'][0], path='data')
    else:
        # There are two or more cache log entries with the same exact filename
        # First try to resolve the problem by 
        # removing any possibly repeated entries from the cache log
        remove_repeated_cache_lines(**kwargs)
        log = read_cache_memory_log(cache_fname)
        mask = log['fname'] == fname
        matching_catalogs = log[mask]
        if len(matching_catalogs) > 1:
            msg = ("The filename you requested ``"+fname+"``\n"
                "appears multiple times in the halo table cache log,\n"
                +cache_fname+"\n, and the metadata between these repeated entries is inconsistent.\n"
                "Use a text editor to open up the log and delete the incorrect lines.\n")
            raise HalotoolsError(msg)



def identify_fname_halo_table(**kwargs):
    """
    """
    # If a cache location is explicitly specified, 
    # use it instead of the standard location. 
    # This facilitate unit-testing
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    # If an input `fname` was passed, check that it exists and return it
    if 'fname' in kwargs:
        if not os.path.isfile(kwargs['fname']):
            msg = ("\nThe requested filename ``" + kwargs['fname'] + "`` does not exist.\n")
            raise HalotoolsError(msg)
        else:
            return kwargs['fname']
    # We need to infer the fname from the metadata and cache log
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
            matching_halo_table_list = auto_detect_halo_table(**kwargs)
            return matching_halo_table_list
        elif len(matching_catalogs) == 1:
            metadata = deepcopy(kwargs)
            try:
                del metadata['cache_fname']
            except KeyError:
                pass
            check_metadata_consistency(matching_catalogs, **metadata)
            return matching_catalogs['fname']
        else:
            msg = ("\nHalotools detected multiple halo catalogs matching "
                "the input arguments.\nThe returned list provides the filenames"
                "of all matching catalogs\n")
            warn(msg)
            return list(matching_catalogs['fname'])


def auto_detect_halo_table(**kwargs):
    """
    """
    matching_halo_table_list = []

    try:
        cache_dirname = kwargs['cache_dirname']
    except KeyError:
        astropy_cache_dir = get_astropy_cache_dir()
        cache_dirname = os.path.join(astropy_cache_dir, 
            'halotools', 'halo_tables')

    fname_pattern = '*.hdf5'
    for path, dirlist, filelist in os.walk(cache_dirname):
        for name in fnmatch.filter(filelist, fname_pattern):
            if file_has_matching_metadata(name, **kwargs) is True:
                matching_halo_table_list.append(name)

    # Now search all directories that appear in the cache log
    # This is not necessarily redundant with the above because 
    # users may have stored halo catalogs on external disks, 
    # which the cache log may be aware of
    verify_cache_log(**kwargs)
    log = read_halo_table_cache_log(**kwargs)
    for entry in log:
        fname_log_entry = entry['fname']
        cache_dirname = os.path.dirname(fname_log_entry)
        for path, dirlist, filelist in os.walk(cache_dirname):
            for name in fnmatch.filter(filelist, fname_pattern):
                if file_has_matching_metadata(name, **kwargs) is True:
                    matching_halo_table_list.append(name)

    matching_halo_table_list = list(set(matching_halo_table_list))
    if len(matching_halo_table_list) == 0:
        msg = ("\nThere are no catalogs in your cache that meet your requested specs.\n"
            "Try supplying an explicit filename instead.\n")
        raise HalotoolsError(msg)
    elif len(matching_halo_table_list) == 1:
        return matching_halo_table_list[0]
    else:
        msg = ("\nHalotools detected multiple halo catalogs matching "
            "the input arguments.\nThe returned list provides the filenames"
            "of all matching catalogs\n")
        warn(msg)
        return matching_halo_table_list



def file_has_matching_metadata(fname_halo_table, dz_tol = 0.05, **kwargs):
    if not os.path.isfile(fname_halo_table):
        raise HalotoolsError("\nThe filename ``"+fname_halo_table+"`` does not exist.\n")
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("Must have h5py package installed "
            "to use the sim_manager sub-package")

    f = h5py.File(fname_halo_table)

    result = True
    metadata_to_check = ('simname', 'halo_finder', 'version_name')
    unavailable_metadata = []
    for metadata_key in metadata_to_check:
        try:
            metadata_in_hdf5_file = f.attrs[metadata_key]
            requested_metadata = kwargs[metadata_key]
            result *= metadata_in_hdf5_file == requested_metadata
        except KeyError:
            unavailable_metadata.append(metadata_key)

    try:
        redshift_of_hdf5_file = f.attrs['redshift']
        requested_redshift = kwargs['redshift']
        result *= abs(redshift_of_hdf5_file - requested_redshift) > dz_tol
    except KeyError:
        unavailable_metadata.append('redshift')

    if len(unavailable_metadata) > 0:
        msg = ("\nThe filename ``"+fname_halo_table+
            "`` has the following missing metadata:\n")
        for metadata_key in unavailable_metadata:
            msg += (metadata_key + "\n")
        msg += ("Thus for this file it is not possible "
            "to verify whether it is a proper match.\n"
            "If this file was provided by Halotools, "
            "please raise an Issue on GitHub.\n"
            "If you provided and/or processed this file yourself, \n"
            "you should use the ``update_metadata`` function on this file.\n")
        warn(msg)

    f.close()
    return result
    

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

def erase_halo_table_cache_log_entry(**kwargs):
    raise HalotoolsError("The erase_halo_table_cache_log_entry function is not implemented yet.")

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
            warn(msg)
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


def check_halo_table_cache_for_nonexistent_files(delete_nonexistent_files=False, **kwargs):
    """ Function searches the halo table cache log for references to files that do not exist and (optionally) deletes them from the log. 
    """

    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    print("Inspecting the halo table cache log located "
        "in the following directory:\ncache_fname")
    verify_cache_log(cache_fname = cache_fname)


    log = read_halo_table_cache_log(cache_fname = cache_fname)
    rows_to_keep = np.ones(len(log), dtype=bool)

    for row, entry in enumerate(log):
        fname = entry['fname']
        if not os.path.isfile(fname):
            missing_file_msg = ("\nLine #"+str(row+1)+" of your halo table cache log \n"
                "stores the following information:\n"
                "simname = " + entry['simname'] + "\n"
                "redshift = " + entry['redshift'] + "\n"
                "halo_finder = " + entry['halo_finder'] + "\n"
                "version_name = " + entry['version_name'] + "\n"
                "fname = " + entry['fname'] + "\n"
                "There is no file stored at that location.\n"
                )

            if delete_nonexistent_files is True:
                missing_file_msg += ("Because you called the ``cleanup_halo_table_cache`` function \n"
                    "with the ``delete_nonexistent_files`` argument set to ``True``, \n"
                    "this entry will now be deleted from your cache.\n")
                print(missing_file_msg)
                rows_to_keep[row] = False
            else:
                missing_file_msg += ("This could simply be because "
                    "the location is an external disk that is not connected.\n"
                    "However, if this file has become obsolete,\n"
                    "then you should delete this entry from your cache.\n"
                    "You can do this in one of two ways:\n"
                    "1. Opening the cache log file stored in \n"+cache_fname+"\n"
                    "and manually deleting line #"+str(row+1)+"\n"
                    "2. Calling the ``cleanup_halo_table_cache`` function again \n"
                    "with the ``delete_nonexistent_files`` argument set to ``True``.\n"
                    )
                print(missing_file_msg)
    log = log[rows_to_keep]
    overwrite_halo_table_cache_log(log, cache_fname = cache_fname)



def remove_repeated_cache_lines(**kwargs):
    """ Method searches the cache for possible repetition of lines and removes them, if present. 
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    verify_cache_log(cache_fname = cache_fname)


    # First we eliminate lines which are exactly repeated
    with open(cache_fname, 'r') as f:
        data_lines = (line for i, line in enumerate(f) if line[0] != '#')
    unique_lines = []
    for line in data_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    # Overwrite the cache with the unique entries
    header = get_halo_table_cache_log_header()
    with open(cache_fname, 'w') as f:
        f.write(header)
        for line in unique_lines:
            f.write(line)
    verify_cache_log(cache_fname = cache_fname)











