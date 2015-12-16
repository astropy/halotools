""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_formatted_halo_table_cache_log_line', 
    'get_halo_table_cache_log_header', 
    'overwrite_halo_table_cache_log', 'read_halo_table_cache_log')

import numpy as np
import os, tempfile, fnmatch
from copy import copy, deepcopy 

from astropy.config.paths import get_cache_dir as get_astropy_cache_dir
from astropy.config.paths import _find_home
from astropy.table import Table
from astropy.table import vstack as table_vstack

from astropy.io import ascii as astropy_ascii

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

def get_redshift_string(redshift):
    return str(np.round(redshift, 4))

def get_halo_table_cache_log_header():
    return 'simname halo_finder redshift version_name fname'

def get_formatted_halo_table_cache_log_line(simname, redshift, 
    halo_finder, version_name, fname):
    redshift_string = get_redshift_string(redshift)
    formatted_line = (
        simname + '  ' + redshift_string + '  ' + 
        halo_finder + '  ' + version_name + '  ' + fname 
        )
    return formatted_line

def overwrite_halo_table_cache_log(new_log, **kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    basename = os.path.dirname(cache_fname)
    try:
        os.makedirs(basename)
    except OSError:
        pass

    new_log.write(cache_fname, format='ascii')

def rebuild_halo_table_cache_log(**kwargs):
    pass

def read_halo_table_cache_log(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    if os.path.isfile(cache_fname):
        return Table.read(cache_fname, format='ascii')
    else:
        msg = ("\nThe Halotools cache log with filename\n``"+cache_fname+"``\n"
            "does not exist. If you have not yet downloaded any of the halo catalogs\n"
            "provided by Halotools, do that now using the ``download_initial_halocat.py`` script,\n"
            "located in ``halotools/scripts``.\n"
            "Otherwise, your cache log may have been accidentally deleted.\n"
            "First verify that your cache directory itself still exists:\n"
            +os.path.dirname(cache_fname) + "\n"
            "Also verify that the ``halo_tables`` sub-directory still contains your halo catalogs.\n"
            "If that checks out, try running the ``rebuild_halo_table_cache_log`` function.\n")
        raise HalotoolsError(msg)


def update_halo_table_cache_log(simname, redshift, 
    halo_finder, version_name, fname, ignore_nearby_redshifts = False, 
    dz_tol = 0.05, **kwargs):
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    if not os.path.isfile(fname):
        msg = ("\nCannot update the cache log with a file named \n" + fname + "\n"
            "because this file does not exist.\n"
            "Be sure you are specifying an absolute path for the fname \n"
            "and verify that the file is located where you think it is.\n")
        raise HalotoolsError(msg)

    new_table_entry = Table(
        {'simname': [simname], 'redshift': [redshift], 
        'halo_finder': [halo_finder], 'version_name': [version_name], 
        'fname': [fname]})
    check_metadata_consistency(new_table_entry)


    remove_repeated_cache_lines(**kwargs)
    log = read_halo_table_cache_log(**kwargs)

    exact_match_mask = np.ones(len(log), dtype = bool)
    exact_match_mask *= simname == log['simname']
    exact_match_mask *= halo_finder == log['halo_finder']
    exact_match_mask *= version_name == log['version_name']

    close_match_mask = copy(exact_match_mask)
    exact_match_mask *= redshift == log['redshift']
    close_match_mask *= abs(redshift - log['redshift']) < dz_tol

    exact_matches = log[exact_match_mask]
    close_matches = log[close_match_mask]

    if len(exact_matches) == 0:
        new_log = table_vstack([log, new_table_entry])
    elif len(exact_matches) == 1:
        if fname == exact_matches['fname']:
            new_log = copy(log)
        else:
            msg = ("\nIn the halo table cache log\n"+cache_fname+"\n"
                "There already exists an entry with metadata that matches "
                "the input arguments to ``update_halo_table_cache_log``\n"
                "However, you requested that the log be updated with the following filename:\n"
                +fname+"\nThis is inconsistent with the filename of the existing log entry:\n"
                +close_matches['fname']+"\nEither delete this line from the log or correct"
                "your input arguments.\n")
            raise HalotoolsError(msg)
    else:
        msg = ("\nIn the halo table cache log\n"+cache_fname+"\n"
            "there are multiple entries with the same metadata but different filenames:\n")
        for entry in exact_matches:
            msg += entry['fname'] + "\n"
        msg += ("This ambiguity can be resolved in one of two ways. \n\n"
            "1. If the duplicate lines are obsolete, simply delete them from the log.\n\n"
            "2. If these lines correctly point to different versions of the same catalog, \n"
            "then you will need to resolve this ambiguity by using different version names \n"
            "To do this, you should first alter the metadata of the hdf5 file as follows:\n\n"
            ">>> f = h5py.File(fname)\n"
            ">>> f.attrs.create('version_name', 'my_new_version_name')\n"
            ">>> f.close()\n\n"
            "After changing the metadata, update the version_name columns in the log.\n")
        raise HalotoolsError(msg)

    if len(exact_matches) == len(close_matches):
        overwrite_halo_table_cache_log(new_log, **kwargs)
    else:
        nearby_redshift_filename = list(set(close_matches['fname']) - set(exact_matches['fname']))[0]
        msg = ("\nThere is already a log entry with a closely matching redshift:\n"
            +nearby_redshift_filename+"\n"
            "You must either use this catalog or set the ``ignore_nearby_redshifts`` to True.\n"
            )
        raise HalotoolsError(msg)


def return_halo_table_fname_after_verification(fname, **kwargs):
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
        msg = ("\nYou tried to load a halo catalog by "
            "passing in the following explicit filename: \n\n" + fname + "\n\nThis file does not exist.\n"
            "First check your spelling, remembering that the full absolute path is required.\n"
            "If your input filename is as intended, the file has either been deleted or \n"
            "is located on an external disk that is not currently plugged in.\n")
        raise HalotoolsError(msg)

    verify_cache_log(**kwargs)
    remove_repeated_cache_lines(cache_fname=cache_fname)
    log = read_halo_table_cache_log(cache_fname=cache_fname)
    mask = log['fname'] == str(fname)
    matching_catalogs = log[mask]

    if len(matching_catalogs) == 0:
        msg = ("The filename you requested ,``"+fname+"`` \nexists but it does not appear"
            "in the halo table cache log,\n"
            +cache_fname+"\nYou can add this catalog to your cache log by calling the\n"
            "``update_halo_table_cache_log`` function.\n")
        raise HalotoolsError(msg)

    elif len(matching_catalogs) == 1:
        idx = np.where(mask == True)[0]
        linenum = idx[0] + 2
        check_metadata_consistency(matching_catalogs[0], linenum = linenum)
        fname = matching_catalogs['fname'][0]
        return fname

    else:
        # There are two or more cache log entries with the same exact filename
        # First try to resolve the problem by 
        # removing any possibly repeated entries from the cache log
        remove_repeated_cache_lines(**kwargs)
        log = read_halo_table_cache_log(cache_fname=cache_fname)
        mask = log['fname'] == fname
        matching_catalogs = log[mask]
        if len(matching_catalogs) == 1:
            check_metadata_consistency(matching_catalogs[0])
            fname = matching_catalogs['fname'][0]
            return fname
        elif len(matching_catalogs) > 1:
            idx = np.where(mask == True)[0] + 1
            msg = ("\nThe filename you requested \n``"+fname+"``\n"
                "appears multiple times in the halo table cache log,\n"
                +"and the metadata stored by these repeated entries is mutually inconsistent.\n"
                "Use a text editor to open up the log and delete the incorrect line(s).\n"
                "The log is stored in the following location:\n"
                +cache_fname+"\n"
                "The offending lines  are #")
            for entry in idx:
                msg += str(entry) + ', '
            msg += "\nwhere the first line of the log file is line #1.\n"
            msg += "\nAlways save a backup version of the log before making manual changes.\n"

            raise HalotoolsError(msg)

def return_halo_table_fname_from_simname_inputs(dz_tol = 0.05, **kwargs):
    """
    """
    if 'fname' in kwargs:
        raise HalotoolsError("\nIf you know the filename of the halo catalog,\n"
            "you should call the ``return_halo_table_fname_after_verification`` function instead.\n")

    try:
        simname = kwargs['simname']
        no_simname_argument = False
    except KeyError:
        simname = sim_defaults.default_simname
        no_simname_argument = True

    try:
        halo_finder = kwargs['halo_finder']
        no_halo_finder_argument = False
    except KeyError:
        halo_finder = sim_defaults.default_halo_finder
        no_halo_finder_argument = True

    try:
        redshift = kwargs['redshift']
        no_redshift_argument = False
    except KeyError:
        redshift = sim_defaults.default_redshift
        no_redshift_argument = True

    try:
        version_name = kwargs['version_name']
        no_version_name_argument = False
    except KeyError:
        version_name = sim_defaults.default_version_name
        no_version_name_argument = True

    # If a cache location is explicitly specified, 
    # use it instead of the standard location. 
    # This facilitate unit-testing
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    remove_repeated_cache_lines(cache_fname=cache_fname)
    log = read_halo_table_cache_log(cache_fname=cache_fname)

    # Search for matching entries in the log
    close_match_mask = np.ones(len(log), dtype=bool)
    close_match_mask *= log['simname'] == simname
    close_match_mask *= log['halo_finder'] == halo_finder
    close_match_mask *= log['version_name'] == version_name

    matches_no_redshift_mask = log[close_match_mask]
    close_match_mask *= abs(log['redshift'] - redshift) < dz_tol
    close_matches = log[close_match_mask]

    # Multiple version check mask
    multiple_version_mask = np.ones(len(log), dtype=bool)
    multiple_version_mask *= log['simname'] == simname
    multiple_version_mask *= log['halo_finder'] == halo_finder
    multiple_version_mask *= abs(log['redshift'] - redshift) < dz_tol
    multi_version_matches = log[multiple_version_mask]

    def add_substring_to_msg(msg):
        if no_simname_argument is True:
            msg += ("simname = ``" + simname + 
                "`` (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + simname + "``\n"

        if no_halo_finder_argument is True:
            msg += ("halo_finder = ``" + halo_finder + 
                "`` (set by sim_defaults.default_halo_finder)\n")
        else:
            msg += "halo_finder = ``" + halo_finder + "``\n"

        if no_redshift_argument is True:
            msg += ("redshift = ``" + str(redshift) + 
                "`` (set by sim_defaults.default_redshift)\n")
        else:
            msg += "redshift = ``" + str(redshift) + "``\n"

        if no_version_name_argument is True:
            msg += ("version_name = ``" + str(version_name) + 
                "`` (set by sim_defaults.default_version_name)\n")
        else:
            msg += "version_name = ``" + str(version_name) + "``\n"
        return msg

    if len(close_matches) == 0:
        if len(matches_no_redshift_mask) == 0:
            msg = ("\nThe Halotools cache log ``"+cache_fname+"``\n"
                "does not contain any entries matching your requested inputs.\n"
                "First, the double-check that your arguments are as intended, including spelling:\n\n")

            msg = add_substring_to_msg(msg)

            msg += ("\nIt is possible that you have spelled everything correctly, \n"
                "but that you just need to add a line to the cache log so that \n"
                "Halotools can remember this simulation in the future.\n"
                "If that is the case, just open up the log, "
                "add a line to it and call this function again.\n"
                "Always save a backup version of the log before making manual changes.\n")
            if len(multi_version_matches) > 0:
                entry = multi_version_matches[0]
                msg += ("\nAlternatively, you do have an alternate version of this catalog\n"
                    "with a closely matching redshift:\n\n")
                msg += "simname = ``" + entry['simname'] + "``\n"
                msg += "halo_finder = ``" + entry['halo_finder'] + "``\n"
                msg += "redshift = ``" + str(entry['redshift']) + "``\n"
                msg += "version_name = ``" + entry['version_name'] + "``\n"
                msg += "fname = ``" + entry['fname'] + "``\n\n"
                msg += ("If this alternate version is the one you want, \n"
                    "just change your function arguments accordingly.")

            raise HalotoolsError(msg)
        else:
            candidate_redshifts = matches_no_redshift_mask['redshift']
            closest_redshift = candidate_redshifts[np.argmin(
                abs(redshift - candidate_redshifts))]
            msg = ("\nThe Halotools cache log ``"+cache_fname+"``\n"
                "does not contain any entries matching your requested inputs.\n"
                "First, the double-check that your arguments are as intended, including spelling:\n\n")

            msg = add_substring_to_msg(msg)

            msg += ("\nFor the cached catalogs matching your \n"
                "``simname``, ``halo_finder`` and ``version_name`` specifications, \n"
                "the closest available redshift is " + str(closest_redshift) + "\n"
                "\nYou should either change your redshift argument \n"
                "or download/process the catalog you need.\n")

            if len(multi_version_matches) > 0:
                entry = multi_version_matches[0]
                msg += ("\nAlternatively, you do have an alternate version of this catalog\n"
                    "with a closely matching redshift:\n\n")
                msg += "simname = ``" + entry['simname'] + "``\n"
                msg += "halo_finder = ``" + entry['halo_finder'] + "``\n"
                msg += "redshift = ``" + str(entry['redshift']) + "``\n"
                msg += "version_name = ``" + entry['version_name'] + "``\n"
                msg += "fname = ``" + entry['fname'] + "``\n\n"
                msg += ("If this alternate version is the one you want, \n"
                    "just change your function arguments accordingly.")

            raise HalotoolsError(msg)

    elif len(close_matches) == 1:
        idx = np.where(close_match_mask == True)[0]
        linenum = idx[0] + 2
        check_metadata_consistency(close_matches[0], linenum = linenum)
        fname = close_matches['fname'][0]

        # Check to make sure that for this filename, 
        # the log does not contain duplicate entries 
        # with mutually inconsistent metadata
        all_fnames_in_log = log['fname']
        fname_mask = fname == all_fnames_in_log
        log_entries_with_matching_fnames = log[fname_mask]
        if len(log_entries_with_matching_fnames) > 1:
            idx = np.where(fname_mask == True)[0] + 1
            msg = ("\nThe filename you requested \n``"+fname+"``\n"
                "appears multiple times in the halo table cache log,\n"
                +"and the metadata stored by these repeated entries is mutually inconsistent.\n"
                "Use a text editor to open up the log and delete the incorrect line(s).\n"
                "The log is stored in the following location:\n"
                +cache_fname+"\n"
                "The offending lines  are #")
            for entry in idx:
                msg += str(entry) + ', '
            msg += "\nwhere the first line of the log file is line #1.\n"
            msg += "\nAlways save a backup version of the log before making manual changes.\n"

            raise HalotoolsError(msg)
            # The log and file are clean, so load the catalog
        else:
            return fname

    else:
        msg = ("\nHalotools detected multiple halo catalogs matching "
            "the input arguments.\nTry decreasing the value of the ``dz_tol`` parameter.\n"
            "Now printing the list of all catalogs matching your requested specifications:\n")
        for entry in close_matches:
            msg += entry['fname'] + "\n"
        raise HalotoolsError(msg)




def auto_detect_matching_halo_tables(**kwargs):
    """
    """
    matching_halo_table_list = []
    close_redshift_halo_table_list = []

    # If a cache location is explicitly specified, 
    # use it instead of the standard location. 
    # This facilitate unit-testing
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    cache_dirname = os.path.dirname(cache_fname)
    halo_table_cache_dirname = os.path.join(cache_dirname, 'halo_tables')

    fname_pattern = '*.hdf5'
    for path, dirlist, filelist in os.walk(halo_table_cache_dirname):
        for name in fnmatch.filter(filelist, fname_pattern):
            exact_match, close_match = halo_table_hdf5_has_matching_metadata(name, **kwargs)
            if exact_match:
                matching_halo_table_list.append(name)
            if close_match:
                close_redshift_halo_table_list.append(name)

    # Now search all directories that appear in the cache log
    # This is not necessarily redundant with the above because 
    # users may have stored halo catalogs on external disks, 
    # which the cache log may be aware of
    verify_cache_log(**kwargs)
    log = read_halo_table_cache_log(**kwargs)
    for entry in log:
        fname_log_entry = entry['fname']
        halo_table_cache_dirname = os.path.dirname(fname_log_entry)
        for path, dirlist, filelist in os.walk(halo_table_cache_dirname):
            for name in fnmatch.filter(filelist, fname_pattern):
                exact_match, close_match = halo_table_hdf5_has_matching_metadata(name, **kwargs)
                if exact_match:
                    matching_halo_table_list.append(name)
                if close_match:
                    close_redshift_halo_table_list.append(name)

    matching_halo_table_list = list(set(matching_halo_table_list))
    if len(matching_halo_table_list) == 0:
        msg = ("\nThere are no catalogs in your cache that meet your requested specs.\n"
            "Either supply more metadata or an explicit filename instead.\n")
        raise HalotoolsError(msg)
    elif len(matching_halo_table_list) == 1:
        return matching_halo_table_list[0]
    else:
        msg = ("\nHalotools detected multiple halo catalogs matching "
            "the input arguments.\nThe returned list provides the filenames"
            "of all matching catalogs\n")
        warn(msg)
        return matching_halo_table_list



def halo_table_hdf5_has_matching_metadata(fname_halo_table, dz_tol = 0.05, **kwargs):
    """
    """
    if not os.path.isfile(fname_halo_table):
        raise HalotoolsError("\nThe filename ``"+fname_halo_table+"`` does not exist.\n")
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("Must have h5py package installed "
            "to use the sim_manager sub-package")

    f = h5py.File(fname_halo_table)

    exact_match = True
    close_redshift_match = True
    metadata_to_check = ('simname', 'halo_finder', 'version_name')
    unavailable_metadata = []
    for metadata_key in metadata_to_check:
        try:
            metadata_in_hdf5_file = f.attrs[metadata_key]
            requested_metadata = kwargs[metadata_key]
            exact_match *= metadata_in_hdf5_file == requested_metadata
            close_redshift_match *= metadata_in_hdf5_file == requested_metadata
        except KeyError:
            unavailable_metadata.append(metadata_key)

    try:
        redshift_of_hdf5_file = np.round(float(f.attrs['redshift']), 4)
        requested_redshift = kwargs['redshift']
        exact_match *= redshift_of_hdf5_file == requested_redshift
        close_redshift_match *= abs(redshift_of_hdf5_file - requested_redshift) > dz_tol
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
    return exact_match, close_redshift_match
    

def verify_halo_table_cache_existence(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    if not os.path.isfile(cache_fname):
        msg = ("\nThe file " + cache_fname + "\ndoes not exist. "
            "This file serves as a log for all the halo catalogs you use with Halotools.\n"
            "If you have not yet downloaded the initial halo catalog,\n"
            "you should do so now following the ``Getting Started`` instructions on "
            "http://halotools.readthedocs.org\nIf you have already taken this step,\n"
            "first verify that your cache directory itself still exists:\n"
            +os.path.dirname(cache_fname) + "\n"
            "Also verify that the ``halo_tables`` sub-directory of the cache \n"
            "still contains your halo catalogs.\n"
            "If that checks out, try running the following script:\n"
            "halotools/scripts/rebuild_halo_table_cache_log.py``\n")
        raise HalotoolsError(msg)

def verify_halo_table_cache_header(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    verify_halo_table_cache_existence(cache_fname=cache_fname)

    correct_header = get_halo_table_cache_log_header()
    correct_list = correct_header.strip().split()

    with open(cache_fname, 'r') as f:
        actual_header = f.readline()
    header_list = actual_header.strip().split()

    if set(correct_list) != set(header_list):
        msg = ("\nThe file " + cache_fname + 
            "\nserves as a log for all the halo catalogs you use with Halotools.\n"
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
        try:
            log = read_halo_table_cache_log(cache_fname = cache_fname)
        except:
            raise HalotoolsError("\nThe log file has become corrupted "
                "and is not readable with astropy.table.Table.read()\n")

    correct_header = get_halo_table_cache_log_header()
    expected_key_set = set(['simname', 'redshift', 'halo_finder', 'fname', 'version_name'])
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

def check_metadata_consistency(cache_log_entry, linenum = None):
    """
    """
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed in order to use "
            "the Halotools halo catalog cache system.\n")

    halo_table_fname = cache_log_entry['fname']
    if os.path.isfile(halo_table_fname):
        f = h5py.File(halo_table_fname)
    else:
        msg = ("\nYou requested to load a halo catalog "
            "with the following filename: \n"+halo_table_fname+"\n"
            "This file does not exist. \n"
            "Either this file has been deleted, or it could just be stored \n"
            "on an external disk that is currently not plugged in.\n")
        raise HalotoolsError(msg)

    # Verify that the columns of the cache log agree with 
    # the metadata stored in the hdf5 file (if present)
    cache_column_names_to_check = ('simname', 'halo_finder', 'version_name', 'redshift')
    for key in cache_column_names_to_check:
        requested_attr = cache_log_entry[key]
        try:
            attr_of_cached_catalog = f.attrs[key]
            if key == 'redshift':
                assert abs(requested_attr - attr_of_cached_catalog) < 0.01
            else:
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
                "has the value ``"+str(attr_of_cached_catalog)+"`` stored as metadata for the "
                "``"+key+"`` attribute.\nThis is inconsistent with the "
                "``"+str(requested_attr)+"`` value that you requested,\n"
                "which is also the value that appears in the log.\n"
                "If you are seeing this message while attempting to load a \n"
                "halo catalog provided by Halotools, please submit a bug report on GitHub.\n"
                "If you are using your own halo catalog that you have stored \n"
                "in the Halotools cache yourself, then you have "
                "attempted to access a halo catalog \nby requesting a value for "
                "the ``"+key+"`` attribute that is inconsistent with the stored value.\n\n"
                "You can rectify this problem in one of two ways:\n\n"
                "1. If the correct value for the ``"+key+
                "`` attribute is ``"+str(attr_of_cached_catalog)+"``,\n"
                "then you should open up the log and change "
                "the ``"+key+"`` column to ``"+str(attr_of_cached_catalog)+"``.\n")
            if linenum is not None:
                msg += "The relevant line to change is line #" + str(linenum) + ",\n"
                msg += "where the first line of the log is line #1.\n"
            else:
                msg += ("The relevant line is the one with the ``fname`` column set to \n"
                    +halo_table_fname+"\n")

            if (type(requested_attr) == str) or (type(requested_attr) == unicode):
                attr_msg = "'"+str(requested_attr)+"'"
            else:
                attr_msg = str(requested_attr)
            msg += ("\n2. If the correct value for the ``"+key+
                "`` attribute is ``"+str(requested_attr)+"``,\n"
                "then your hdf5 file has incorrect metadata that needs to be changed.\n"
                "You can make the correction as follows:\n\n"
                ">>> fname = '"+halo_table_fname+"'\n"
                ">>> f = h5py.File(fname)\n"
                ">>> f.attrs.create('"+key+"', "+attr_msg+")\n"
                ">>> f.close()\n\n"
                "Be sure to use string-valued variables for the following inputs:\n"
                "``simname``, ``halo_finder``, ``version_name`` and ``fname``,\n"
                "and a float for the ``redshift`` input.\n"
                )
            raise HalotoolsError(msg)

    try:
        assert 'Lbox' in f.attrs.keys()
        assert 'ptcl_mass' in f.attrs.keys()
    except AssertionError:
        msg = ("\nAll halo tables must contain metadata storing the "
            "box size and particle mass of the simulation.\n"
            "The halo table stored in the following location is missing this metadata:\n"
            +halo_table_fname+"\n")
        raise HalotoolsError(msg)

    try:
        d = f['data']
        assert 'halo_id' in d.dtype.names
        assert 'halo_x' in d.dtype.names
        assert 'halo_y' in d.dtype.names
        assert 'halo_z' in d.dtype.names
    except AssertionError:
        msg = ("\nAll halo tables must have the following columns:\n"
            "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``\n")
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
        data_lines = [line for i, line in enumerate(f) if line[0] != '#']
    unique_lines = []
    for line in data_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    # Overwrite the cache with the unique entries
    header = get_halo_table_cache_log_header()
    with open(cache_fname, 'w') as f:
        for line in unique_lines:
            f.write(line)
    verify_cache_log(cache_fname = cache_fname)


def store_new_halo_table_in_cache(halo_table, ignore_nearby_redshifts = False, 
    **metadata):
    """
    """
    try:
        assert type(halo_table) is Table
    except AssertionError:
        msg = ("\nThe input ``halo_table`` must be an Astropy Table object.\n")
        raise HalotoolsError(msg)

    try:
        cache_fname = deepcopy(metadata['cache_fname'])
        del metadata['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()

    # Verify that the metadata has all the necessary keys
    try:
        simname = metadata['simname']
        halo_finder = metadata['halo_finder']
        redshift = metadata['redshift']
        version_name = metadata['version_name']
        fname = metadata['fname']
        Lbox = metadata['Lbox']
        ptcl_mass = metadata['ptcl_mass']
    except KeyError:
        msg = ("\nYou tried to create a new halo catalog without passing in\n"
            "a sufficient amount of metadata as keyword arguments.\n"
            "All calls to the `store_new_halo_table_in_cache` function\n"
            "must have the following keyword arguments "
            "that will be interpreted as halo catalog metadata:\n\n"
            "``simname``, ``halo_finder``, ``redshift``, ``version_name``, ``fname``, \n"
            "``Lbox``, ``ptcl_mass``\n")
        raise HalotoolsError(msg)


    try:
        assert str(fname[-5:]) == '.hdf5'
    except AssertionError:
        msg = ("\nThe input ``fname`` must end with the extension ``.hdf5``\n")
        raise HalotoolsError(msg)

    # The filename cannot already exist
    if os.path.isfile(fname):
        raise HalotoolsError("\nYou tried to store a new halo catalog "
            "with the following filename: \n\n"
            +fname+"\n\n"
            "A file at this location already exists. \n"
            "If you want to overwrite an existing halo catalog,\n"
            "you must instead call the `overwrite_existing_halo_table_in_cache` function.\n"
            "Otherwise, you should choose a different filename.\n")

    try:
        verify_halo_table_cache_existence(cache_fname = cache_fname)
        first_halo_table_in_cache = False
    except HalotoolsError:
        # This is the first halo catalog being stored in cache
        first_halo_table_in_cache = True
        new_log = Table()
        new_log['simname'] = [simname]
        new_log['halo_finder'] = [halo_finder]
        new_log['redshift'] = [redshift]
        new_log['version_name'] = [version_name]
        new_log['fname'] = [fname]
        overwrite_halo_table_cache_log(new_log, cache_fname = cache_fname)

    verify_cache_log(cache_fname = cache_fname)
    remove_repeated_cache_lines(cache_fname = cache_fname)
    log = read_halo_table_cache_log(cache_fname = cache_fname)

    # There is no need for any of the following checks if this is the first catalog stored
    if first_halo_table_in_cache is False:

        # Make sure that the filename does not already appear in the log
        exact_match_mask, close_match_mask = (
            search_log_for_possibly_existing_entry(log, fname = fname)
            )
        exactly_matching_entries = log[exact_match_mask]
        if len(exactly_matching_entries) == 0:
            pass
        else:
            msg = ("\nThe filename you are trying to store, \n"
                +fname+"\nappears one or more times in the Halotools cache log,\n"
                "and yet this file does not yet exist.\n"
                "You must first remedy this problem before you can proceed.\n"
                "Use a text editor to open the cache log, "
                "which is stored at the following location:\n\n"
                +cache_fname+"\n\nThen simply delete the line(s) storing incorrect metadata.\n"
                "The offending lines are #")
            idx = np.where(exact_match_mask == True)[0] + 1
            for entry in idx:
                msg += str(entry) + ', '
            msg += "\nwhere the first line of the log file is line #1.\n"
            msg += "\nAlways save a backup version of the log before making manual changes.\n"
            raise HalotoolsError(msg)

        # Now make sure that there is no log entry with the same metadata 
        exact_match_mask, close_match_mask = (
            search_log_for_possibly_existing_entry(log, 
                simname = simname, halo_finder = halo_finder, 
                redshift = redshift, version_name = version_name)
            )
        exactly_matching_entries = log[exact_match_mask]
        closely_matching_entries = log[close_match_mask]
        if len(closely_matching_entries) == 0:
            pass
        else:
            if len(exactly_matching_entries) == 0:
                if ignore_nearby_redshifts == True:
                    pass
                else:
                    msg = ("\nThere already exists a halo catalog in cache \n"
                        "with the same metadata as the catalog you are trying to store, \n"
                        "and a very similar redshift. The closely matching"
                        "halo catalog has the following filename:\n"
                        +closely_matching_entries['fname'][0]+"\n"
                        "If you want to proceed anyway, you must set the \n"
                        "``ignore_nearby_redshifts`` keyword argument to ``True``.\n"
                        )
                    raise HalotoolsError(msg)
            else:
                msg = ("\nThere is already a halo catalog in your cache log with metdata \n"
                    "that exactly matches the metadata of the catalog you are trying to store.\n"
                    "The filename of this matching halo catalog is:\n\n"
                    +exactly_matching_entries['fname'][0]+"\n\n"
                    "If this log entry is spurious, you should open the log \n"
                    "with a text editor and delete the offending line.\n"
                    "The log is stored at the following filename:\n\n"
                    +cache_fname+"\n\n"
                    "If this matching halo catalog is one you want to continue keeping track of, \n"
                    "then you should change the ``version_name`` \nof the halo catalog "
                    "you are trying to store.\n"
                    )
                raise HalotoolsError(msg)


    # At this point, we have ensured that the filename does not already exist 
    # and it is safe to consider it as a new log entry. 
    # Now we must verify the metadata that was passed in 
    # is consistent with the halo table contents. 


    try:
        halo_id = halo_table['halo_id']
        halo_x = halo_table['halo_x']
        halo_y = halo_table['halo_y']
        halo_z = halo_table['halo_z']
    except KeyError:
        msg = ("\nAll halo tables must at least have the following columns:\n"
            "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``\n")
        raise HalotoolsError(msg)

    # Check that Lbox properly bounds the halo positions
    try:
        assert np.all(halo_x >= 0)
        assert np.all(halo_y >= 0)
        assert np.all(halo_z >= 0)
        assert np.all(halo_x <= Lbox)
        assert np.all(halo_y <= Lbox)
        assert np.all(halo_z <= Lbox)
    except AssertionError:
        msg = ("\nThere are points in the input halo table that "
            "lie outside [0, Lbox] in some dimension.\n")
        raise HalotoolsError(msg)

    # Check that halo_id column contains a set of unique entries
    try:
        num_halos = len(halo_table)
        unique_halo_ids = list(set(halo_id))
        num_unique_ids = len(unique_halo_ids)
        assert num_halos == num_unique_ids
    except AssertionError:
        msg = ("\nThe ``halo_id`` column of your halo table must contain a unique integer "
            "for every halo\n")
        raise HalotoolsError(msg)

    # The table appears to be kosher, so we write it to an hdf5 file, 
    # add metadata, and update the log
    halo_table.write(fname, path='data')

    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed "
            "in order to store a new halo catalog.\n")
    f = h5py.File(fname)
    for key, value in metadata.iteritems():
        if type(value) == unicode:
            value = str(value)
        f.attrs.create(key, value)
    f.close()

    if first_halo_table_in_cache is False:
        new_table_entry = Table({'simname': [simname], 
            'halo_finder': [halo_finder], 
            'redshift': [redshift], 
            'version_name': [version_name], 
            'fname': [fname]}
            )

        new_log = table_vstack([log, new_table_entry])
        overwrite_halo_table_cache_log(new_log, cache_fname = cache_fname)
        remove_repeated_cache_lines(cache_fname = cache_fname)



def remove_unique_fname_from_halo_table_cache_log(fname, 
    raise_warning = False, **kwargs):
    """
    """

    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_halo_table_cache_log_fname()
    verify_cache_log(cache_fname = cache_fname)
    remove_repeated_cache_lines(cache_fname = cache_fname)
    log = read_halo_table_cache_log(cache_fname = cache_fname)

    mask = log['fname'] == fname
    matching_entries = log[mask]
    if len(matching_entries) == 0:
        if raise_warning == True:
            msg = ("\nYou requested that the following fname be deleted \n"
                "from the halo table cache log:\n"+fname+"\n"
                "However, this filename does not appear in the log.\n"
                "This is likely harmless, and if you do not wish to see this warning message,\n"
                "just set the `raise_warning` keyword argument to False.\n")
            warn(msg)
        else:
            pass
    elif len(matching_entries) == 1:
        new_log = log[~mask]
        overwrite_halo_table_cache_log(new_log, cache_fname = cache_fname)
    else:
        idx = np.where(mask == True)[0] + 1
        msg = ("\nYou requested that the following fname be deleted \n"
            "from the halo table cache log:\n"+fname+"\n"
            "However, this filename appears more than once in the log,\n"
            "with the different entries having mutually incompatible metadata.\n"
            "Only one set of this metadata can be correct for a given filename.\n"
            "You must first remedy this problem before you can proceed.\n"
            "To do so, use a text editor to open the cache log, "
            "which is stored at the following location:\n"
            +cache_fname+"\nThen simply delete the line(s) storing incorrect metadata"
            "The offending lines are #")
        for entry in idx:
            msg += str(entry) + ', '
        msg += "\nwhere the first line of the log file is line #1.\n"
        msg += "\nAlways save a backup version of the log before making manual changes.\n"
        raise HalotoolsError(msg)



def overwrite_existing_halo_table_in_cache(halo_table, 
    simname, halo_finder, redshift, version_name, fname, 
    **kwargs):
    pass


def verify_file_storing_unrecognized_halo_table(fname):
    """
    """
    if not os.path.isfile(fname):
        msg = ("\nThe input filename \n" + fname + "\ndoes not exist.")
        raise HalotoolsError(msg)

    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed "
            "in order to use the verify_unrecognized_halo_table function.\n")

    try:
        f = h5py.File(fname)
    except:
        msg = ("\nThe input filename \n" + fname + "\nmust be an hdf5 file.\n")
        raise HalotoolsError(msg)

    try:
        simname = f.attrs['simname']
        halo_finder = f.attrs['halo_finder']
        redshift = np.round(float(f.attrs['redshift']), 4)
        version_name = f.attrs['version_name']
        Lbox = f.attrs['Lbox']
        ptcl_mass = f.attrs['ptcl_mass']
        inferred_fname = f.attrs['fname']
    except:
        msg = ("\nThe hdf5 file storing the halos must have the following metadata:\n"
            "``simname``, ``halo_finder``, ``redshift``, ``version_name``, ``fname``, "
            "``Lbox``, ``ptcl_mass``\n"
            "Here is an example of how to add metadata "
            "for hdf5 files can be added using the following syntax:\n\n"
            ">>> f = h5py.File(fname)\n"
            ">>> f.attrs.create('simname', simname)\n"
            ">>> f.close()\n\n"
            "Be sure to use string-valued variables for the following inputs:\n"
            "``simname``, ``halo_finder``, ``version_name`` and ``fname``,\n"
            "and floats for the following inputs:\n"
            "``redshift``, ``Lbox`` (in Mpc/h)  ``ptcl_mass`` (in Msun/h)\n"
            )

        raise HalotoolsError(msg)

    try:
        halo_table = f['data']
    except:
        msg = ("\nThe hdf5 file must have a dset key called `data`\n"
            "so that the halo table is accessible with the following syntax:\n"
            ">>> f = h5py.File(fname)\n"
            ">>> halo_table = f['data']\n")
        raise HalotoolsError(msg)

    try:
        halo_id = halo_table['halo_id']
        halo_x = halo_table['halo_x']
        halo_y = halo_table['halo_y']
        halo_z = halo_table['halo_z']
    except KeyError:
        msg = ("\nAll halo tables must at least have the following columns:\n"
            "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``\n")
        raise HalotoolsError(msg)

    # Check that Lbox properly bounds the halo positions
    try:
        assert np.all(halo_x >= 0)
        assert np.all(halo_y >= 0)
        assert np.all(halo_z >= 0)
        assert np.all(halo_x <= Lbox)
        assert np.all(halo_y <= Lbox)
        assert np.all(halo_z <= Lbox)
    except AssertionError:
        msg = ("\nThere are points in the input halo table that "
            "lie outside [0, Lbox] in some dimension.\n")
        raise HalotoolsError(msg)

    # Check that halo_id column contains a set of unique entries
    try:
        num_halos = len(halo_table)
        unique_halo_ids = list(set(halo_id))
        num_unique_ids = len(unique_halo_ids)
        assert num_halos == num_unique_ids
    except AssertionError:
        msg = ("\nThe ``halo_id`` column of your halo table must contain a unique integer "
            "for every halo\n")
        raise HalotoolsError(msg)

    return fname

def search_log_for_possibly_existing_entry(log, dz_tol = 0.05, **entries_to_check):
    """
    """
    exact_match_mask = np.ones(len(log), dtype = bool)
    close_match_mask = np.ones(len(log), dtype = bool)

    for key, value in entries_to_check.iteritems():
        exact_match_mask *= log[key] == value

    for key, value in entries_to_check.iteritems():
        if key == 'redshift':
            close_match_mask *= abs(log[key] - value) < dz_tol
        else:
            close_match_mask *= log[key] == value

    return exact_match_mask, close_match_mask
















