""" This module contains functions used to read, interpret and 
update the ascii data file that keeps track of N-body simulation data. 
"""

__all__ = ('get_ptcl_table_cache_log_header', 
    'overwrite_ptcl_table_cache_log', 'read_ptcl_table_cache_log')

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

def get_ptcl_table_cache_log_fname():
    dirname = os.path.join(get_astropy_cache_dir(), 'halotools')
    return os.path.join(dirname, 'ptcl_table_cache_log.txt')

def get_ptcl_table_cache_log_header():
    return 'simname redshift version_name fname'

def overwrite_ptcl_table_cache_log(new_log, **kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()

    basename = os.path.dirname(cache_fname)
    try:
        os.makedirs(basename)
    except OSError:
        pass

    new_log.write(cache_fname, format='ascii')

def read_ptcl_table_cache_log(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()

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
            "Also verify that the ``ptcl_tables`` sub-directory still contains your particle catalogs.\n"
            "If that checks out, try running the ``rebuild_ptcl_table_cache_log`` function.\n")
        raise HalotoolsError(msg)

def return_ptcl_table_fname_after_verification(fname, **kwargs):
    """
    """
    # If a cache location is explicitly specified, 
    # use it instead of the standard location. 
    # This facilitate unit-testing
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()

    try:
        assert os.path.isfile(fname)
    except AssertionError:
        msg = ("\nYou tried to load a particle catalog by "
            "passing in the following explicit filename: \n\n" + fname + "\n\nThis file does not exist.\n"
            "First check your spelling, remembering that the full absolute path is required.\n"
            "If your input filename is as intended, the file has either been deleted or \n"
            "is located on an external disk that is not currently plugged in.\n")
        raise HalotoolsError(msg)

    verify_ptcl_table_cache_log(**kwargs)
    remove_repeated_ptcl_table_cache_lines(cache_fname=cache_fname)
    log = read_ptcl_table_cache_log(cache_fname=cache_fname)
    mask = log['fname'] == str(fname)
    matching_catalogs = log[mask]

    if len(matching_catalogs) == 0:
        msg = ("The filename you requested ,``"+fname+"`` \nexists but it does not appear"
            "in the particle table cache log,\n"
            +cache_fname+"\nYou can add this catalog to your cache log by "
            "opening the log file with a text editor\n"
            "and adding the appropriate line that matches the existing pattern.\n")
        raise HalotoolsError(msg)

    elif len(matching_catalogs) == 1:
        idx = np.where(mask == True)[0]
        linenum = idx[0] + 2
        check_ptcl_table_metadata_consistency(matching_catalogs[0], linenum = linenum)
        fname = matching_catalogs['fname'][0]
        return fname

    else:
        # There are two or more cache log entries with the same exact filename
        # First try to resolve the problem by 
        # removing any possibly repeated entries from the cache log
        remove_repeated_ptcl_table_cache_lines(**kwargs)
        log = read_ptcl_table_cache_log(cache_fname=cache_fname)
        mask = log['fname'] == fname
        matching_catalogs = log[mask]
        if len(matching_catalogs) == 1:
            check_ptcl_table_metadata_consistency(matching_catalogs[0])
            fname = matching_catalogs['fname'][0]
            return fname
        elif len(matching_catalogs) > 1:
            idx = np.where(mask == True)[0] + 1
            msg = ("\nThe filename you requested \n``"+fname+"``\n"
                "appears multiple times in the particle table cache log,\n"
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


def return_ptcl_table_fname_from_simname_inputs(dz_tol = 0.05, **kwargs):
    """
    """
    try:
        simname = kwargs['simname']
        no_simname_argument = False
    except KeyError:
        simname = sim_defaults.default_simname
        no_simname_argument = True

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
        cache_fname = get_ptcl_table_cache_log_fname()
    remove_repeated_ptcl_table_cache_lines(cache_fname=cache_fname)
    log = read_ptcl_table_cache_log(cache_fname=cache_fname)

    # Search for matching entries in the log
    close_redshift_match_mask = np.ones(len(log), dtype=bool)
    close_redshift_match_mask *= log['simname'] == simname
    close_redshift_match_mask *= log['version_name'] == version_name
    close_redshift_match_mask *= abs(log['redshift'] - redshift) < dz_tol
    close_matches = log[close_redshift_match_mask]

    simname_only_mask = log['simname'] == simname

    def add_substring_to_msg(msg):
        if no_simname_argument is True:
            msg += ("simname = ``" + simname + 
                "`` (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + simname + "``\n"

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

        msg = ("\nThe Halotools cache log ``"+cache_fname+"``\n"
            "does not contain any entries matching your requested inputs.\n"
            "First, the double-check that your arguments are as intended, including spelling:\n\n")

        msg = add_substring_to_msg(msg)

        msg += ("\nIt is possible that you have spelled everything correctly, \n"
            "but that you just need to add a line to the cache log so that \n"
            "Halotools can remember this simulation in the future.\n"
            "If that is the case, just open up the log, "
            "add a line to it and call this function again.\n"
            "Be sure that the redshift you enter agrees exactly \nwith the "
            "corresponding entry of `halo_table_cache_log.txt`\n"
            "Always save a backup version of the log before making manual changes.\n")
        raise HalotoolsError(msg)

    elif len(close_matches) == 1:
        idx = np.where(close_redshift_match_mask == True)[0]
        linenum = idx[0] + 2
        check_ptcl_table_metadata_consistency(close_matches[0], linenum = linenum)
        fname = close_matches['fname'][0]
        return fname
    else:
        msg = ("\nHalotools detected multiple particle catalogs matching "
            "the input arguments.\n"
            "Now printing the list of all catalogs matching your requested specifications:\n")
        for entry in close_matches:
            msg += entry['fname'] + "\n"
        msg += ("Either delete the erroneous lines from the log \n"
            "or decrease the ``dz_tol`` parameter of the "
            "return_ptcl_table_fname_from_simname_inputs function.\n")
        raise HalotoolsError(msg)

def check_ptcl_table_metadata_consistency(cache_log_entry, linenum = None):
    """
    """
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed in order to use "
            "the Halotools particle catalog cache system.\n")

    ptcl_table_fname = cache_log_entry['fname']
    if os.path.isfile(ptcl_table_fname):
        f = h5py.File(ptcl_table_fname)
    else:
        msg = ("\nYou requested to load a particle catalog "
            "with the following filename: \n"+ptcl_table_fname+"\n"
            "This file does not exist. \n"
            "Either this file has been deleted, or it could just be stored \n"
            "on an external disk that is currently not plugged in.\n")
        raise HalotoolsError(msg)

    # Verify that the columns of the cache log agree with 
    # the metadata stored in the hdf5 file (if present)
    cache_column_names_to_check = ('simname', 'version_name', 'redshift')
    for key in cache_column_names_to_check:
        requested_attr = cache_log_entry[key]
        try:
            attr_of_cached_catalog = f.attrs[key]
            if key == 'redshift':
                assert abs(requested_attr - attr_of_cached_catalog) < 0.01
            else:
                assert attr_of_cached_catalog == requested_attr
        except KeyError:
            msg = ("\nThe particle table stored in \n``"+ptcl_table_fname+"\n"
                "does not have metadata stored for the ``"+key+"`` attribute\n"
                "and so some self-consistency checks cannot be performed.\n"
                "If you are seeing this message while attempting to load a \n"
                "particle catalog provided by Halotools, please submit a bug report on GitHub.\n"
                "If you are using your own particle catalog that you have stored \n"
                "in the Halotools cache yourself, you should consider adding this metadata\n"
                "to the hdf5 file as one of the keys of the .attrs file attribute.\n")
            warn(msg)
        except AssertionError:
            msg = ("\nThe particle table stored in \n``"+ptcl_table_fname+"\n"
                "has the value ``"+str(attr_of_cached_catalog)+"`` stored as metadata for the "
                "``"+key+"`` attribute.\nThis is inconsistent with the "
                "``"+str(requested_attr)+"`` value that you requested,\n"
                "which is also the value that appears in the log.\n"
                "If you are seeing this message while attempting to load a \n"
                "particle catalog provided by Halotools, please submit a bug report on GitHub.\n"
                "If you are using your own particle catalog that you have stored \n"
                "in the Halotools cache yourself, then you have "
                "attempted to access a particle catalog \nby requesting a value for "
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
                    +ptcl_table_fname+"\n")

            if (type(requested_attr) == str) or (type(requested_attr) == unicode):
                attr_msg = "'"+str(requested_attr)+"'"
            else:
                attr_msg = str(requested_attr)
            msg += ("\n2. If the correct value for the ``"+key+
                "`` attribute is ``"+str(requested_attr)+"``,\n"
                "then your hdf5 file has incorrect metadata that needs to be changed.\n"
                "You can make the correction as follows:\n\n"
                ">>> fname = '"+ptcl_table_fname+"'\n"
                ">>> f = h5py.File(fname)\n"
                ">>> f.attrs.create('"+key+"', "+attr_msg+")\n"
                ">>> f.close()\n\n"
                "Be sure to use string-valued variables for the following inputs:\n"
                "``simname``, ``version_name`` and ``fname``,\n"
                "and a float for the ``redshift`` input.\n"
                )
            raise HalotoolsError(msg)

    try:
        assert 'Lbox' in f.attrs.keys()
    except AssertionError:
        msg = ("\nAll particle tables must contain metadata storing the "
            "box size of the simulation.\n"
            "The particle table stored in the following location is missing this metadata:\n"
            +ptcl_table_fname+"\n")
        raise HalotoolsError(msg)

    try:
        d = f['data']
        assert 'x' in d.dtype.names
        assert 'y' in d.dtype.names
        assert 'z' in d.dtype.names
    except AssertionError:
        msg = ("\nAll particle tables must at least have the following columns:\n"
            "``x``, ``y``, ``z``\n")
        raise HalotoolsError(msg)

    f.close()


def remove_repeated_ptcl_table_cache_lines(**kwargs):
    """ Method searches the cache for possible repetition of lines and removes them, if present. 
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()
    verify_ptcl_table_cache_log(cache_fname = cache_fname)


    # First we eliminate lines which are exactly repeated
    with open(cache_fname, 'r') as f:
        data_lines = [line for i, line in enumerate(f) if line[0] != '#']
    unique_lines = []
    for line in data_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    # Overwrite the cache with the unique entries
    header = get_ptcl_table_cache_log_header()
    with open(cache_fname, 'w') as f:
        for line in unique_lines:
            f.write(line)
    verify_ptcl_table_cache_log(cache_fname = cache_fname)


def verify_ptcl_table_cache_log(**kwargs):

    verify_ptcl_table_cache_existence(**kwargs)
    verify_ptcl_table_cache_header(**kwargs)
    verify_ptcl_table_cache_log_columns(**kwargs)


def verify_ptcl_table_cache_existence(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()

    if not os.path.isfile(cache_fname):
        msg = ("\nThe file " + cache_fname + "\ndoes not exist. "
            "This file serves as a log for all the catalogs of randomly selected\n"
            "dark matter particles you use with Halotools.\n"
            "If you have not yet downloaded the initial particle catalog,\n"
            "you should do so now following the ``Getting Started`` instructions on "
            "http://halotools.readthedocs.org\n\nIf you have already taken this step,\n"
            "first verify that your cache directory itself still exists:\n"
            +os.path.dirname(cache_fname) + "\n"
            "Also verify that the ``particle_catalogs`` sub-directory of the cache \n"
            "still contains your particle catalogs.\n"
            "If that checks out, try running the following script:\n"
            "halotools/scripts/rebuild_ptcl_table_cache_log.py``\n")
        raise HalotoolsError(msg)

def verify_ptcl_table_cache_header(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()

    verify_ptcl_table_cache_existence(cache_fname=cache_fname)

    correct_header = get_ptcl_table_cache_log_header()
    correct_list = correct_header.strip().split()

    with open(cache_fname, 'r') as f:
        actual_header = f.readline()
    header_list = actual_header.strip().split()

    if set(correct_list) != set(header_list):
        msg = ("\nThe file " + cache_fname + 
            "\nserves as a log for all the catalogs of randomly selected\n"
            "dark matter particles you use with Halotools.\n"
            "The correct header that should be in this file is \n"
            + correct_header + "\nThe actual header currently stored in this file is \n"
            + actual_header + "\nTo resolve your error, try opening the log file "
            "with a text editor and replacing the current line with the correct one.\n")
        raise HalotoolsError(msg)

def verify_ptcl_table_cache_log_columns(**kwargs):
    """
    """
    try:
        cache_fname = kwargs['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()
    verify_ptcl_table_cache_existence(cache_fname = cache_fname)
    verify_ptcl_table_cache_header(cache_fname = cache_fname)

    try:
        log = kwargs['log']
    except KeyError:
        try:
            log = read_ptcl_table_cache_log(cache_fname = cache_fname)
        except:
            msg = ("The file " + cache_fname + 
                "\nkeeps track of the particle catalogs"
                "you use with Halotools.\n"
                "This file appears to be corrupted.\n"
                "Please visually inspect this file to ensure it has not been "
                "accidentally overwritten. \n"
                "Then store a backup of this file and execute the following script:\n"
                "halotools/scripts/rebuild_ptcl_table_cache_log.py\n"
                "If this does not resolve the error you are encountering,\n"
                "and if you have been using particle catalogs stored on some external disk \n"
                "or other non-standard location, you may try manually adding \n"
                "the appropriate lines to the cache log.\n"
                "Please contact the Halotools developers if the issue persists.\n")
            raise HalotoolsError(msg)

    correct_header = get_ptcl_table_cache_log_header()
    expected_key_set = set(['simname', 'redshift', 'fname', 'version_name'])
    log_key_set = set(log.keys())
    try:
        assert log_key_set == expected_key_set
    except AssertionError:
        cache_fname = get_ptcl_table_cache_log_fname()
        msg = ("The file " + cache_fname + 
            "\nkeeps track of the particle catalogs"
            "you use with Halotools.\n"
            "This file appears to be corrupted.\n"
            "Please visually inspect this file to ensure it has not been "
            "accidentally overwritten. \n"
            "Then store a backup of this file and execute the following script:\n"
            "halotools/scripts/rebuild_ptcl_table_cache_log.py\n"
            "If this does not resolve the error you are encountering,\n"
            "and if you have been using particle catalogs stored on some external disk \n"
            "or other non-standard location, you may try manually adding \n"
            "the appropriate lines to the cache log.\n"
            "Please contact the Halotools developers if the issue persists.\n")
        raise HalotoolsError(msg)



def verify_file_storing_unrecognized_ptcl_table(fname):
    """
    """
    if not os.path.isfile(fname):
        msg = ("\nThe input filename \n" + fname + "\ndoes not exist.")
        raise HalotoolsError(msg)

    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed "
            "in order to use the verify_unrecognized_ptcl_table function.\n")

    try:
        f = h5py.File(fname)
    except:
        msg = ("\nThe input filename \n" + fname + "\nmust be an hdf5 file.\n")
        raise HalotoolsError(msg)

    try:
        simname = f.attrs['simname']
        redshift = np.round(float(f.attrs['redshift']), 4)
        version_name = f.attrs['version_name']
        Lbox = f.attrs['Lbox']
        inferred_fname = f.attrs['fname']
    except:
        msg = ("\nThe hdf5 file storing the particles must have the following metadata:\n"
            "``simname``, ``redshift``, ``version_name``, ``fname``, ``Lbox``. \n"
            "Here is an example of how to add metadata "
            "for hdf5 files can be added using the following syntax:\n\n"
            ">>> f = h5py.File(fname)\n"
            ">>> f.attrs.create('simname', simname)\n"
            ">>> f.close()\n\n"
            "Be sure to use string-valued variables for the following inputs:\n"
            "``simname``, ``version_name`` and ``fname``,\n"
            "and floats for the following inputs:\n"
            "``redshift``, ``Lbox`` (in Mpc/h) \n"
            )

        raise HalotoolsError(msg)

    try:
        ptcl_table = f['data']
    except:
        msg = ("\nThe hdf5 file must have a dset key called `data`\n"
            "so that the particle table is accessible with the following syntax:\n"
            ">>> f = h5py.File(fname)\n"
            ">>> ptcl_table = f['data']\n")
        raise HalotoolsError(msg)

    try:
        ptcl_x = ptcl_table['x']
        ptcl_y = ptcl_table['y']
        ptcl_z = ptcl_table['z']
    except KeyError:
        msg = ("\nAll particle tables must at least have the following columns:\n"
            "``x``, ``y``, ``z``\n")
        raise HalotoolsError(msg)

    # Check that Lbox properly bounds the particle positions
    try:
        assert np.all(ptcl_x >= 0)
        assert np.all(ptcl_y >= 0)
        assert np.all(ptcl_z >= 0)
        assert np.all(ptcl_x <= Lbox)
        assert np.all(ptcl_y <= Lbox)
        assert np.all(ptcl_z <= Lbox)
    except AssertionError:
        msg = ("\nThere are points in the input particle table that "
            "lie outside [0, Lbox] in some dimension.\n")
        raise HalotoolsError(msg)

    return fname


def store_new_ptcl_table_in_cache(ptcl_table, ignore_nearby_redshifts = False, 
    **metadata):
    """
    """
    try:
        assert type(ptcl_table) is Table
    except AssertionError:
        msg = ("\nThe input ``ptcl_table`` must be an Astropy Table object.\n")
        raise HalotoolsError(msg)

    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed "
            "in order to store a new particle catalog.\n")

    # The following two keyword arguments are intentionally absent 
    # from the docstring and are for developer convenience only. 
    # No end-user should ever have recourse for either 
    # cache_fname or overwrite_existing_ptcl_table
    try:
        cache_fname = deepcopy(metadata['cache_fname'])
        del metadata['cache_fname']
    except KeyError:
        cache_fname = get_ptcl_table_cache_log_fname()


    # Verify that the metadata has all the necessary keys
    try:
        simname = metadata['simname']
        redshift = metadata['redshift']
        version_name = metadata['version_name']
        fname = metadata['fname']
        Lbox = metadata['Lbox']
    except KeyError:
        msg = ("\nYou tried to create a new particle catalog without passing in\n"
            "a sufficient amount of metadata as keyword arguments.\n"
            "All calls to the `store_new_ptcl_table_in_cache` function\n"
            "must have the following keyword arguments "
            "that will be interpreted as particle catalog metadata:\n\n"
            "``simname``, ``redshift``, ``version_name``, ``fname``, \n"
            "``Lbox``\n")
        raise HalotoolsError(msg)


    try:
        assert str(fname[-5:]) == '.hdf5'
    except AssertionError:
        msg = ("\nThe input ``fname`` must end with the extension ``.hdf5``\n")
        raise HalotoolsError(msg)

    # The filename cannot already exist
    if os.path.isfile(fname):
        raise HalotoolsError("\nYou tried to store a new particle catalog "
            "with the following filename: \n\n"
            +fname+"\n\n"
            "A file at this location already exists. \n"
            "Either delete it or choose a different filename.\n")

    try:
        verify_ptcl_table_cache_existence(cache_fname = cache_fname)
        first_ptcl_table_in_cache = False
    except HalotoolsError:
        # This is the first particle catalog being stored in cache
        first_ptcl_table_in_cache = True
        new_log = Table()
        new_log['simname'] = [simname]
        new_log['redshift'] = [redshift]
        new_log['version_name'] = [version_name]
        new_log['fname'] = [fname]
        overwrite_ptcl_table_cache_log(new_log, cache_fname = cache_fname)

    verify_cache_log(cache_fname = cache_fname)
    remove_repeated_ptcl_table_cache_lines(cache_fname = cache_fname)
    log = read_ptcl_table_cache_log(cache_fname = cache_fname)

    # There is no need for any of the following checks if this is the first catalog stored
    if first_ptcl_table_in_cache is False:

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
                simname = simname, 
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
                    msg = ("\nThere already exists a particle catalog in cache \n"
                        "with the same metadata as the catalog you are trying to store, \n"
                        "and a very similar redshift. \nThe closely matching "
                        "particle catalog has the following filename:\n\n"
                        +closely_matching_entries['fname'][0]+"\n\n"
                        "If you want to proceed anyway, you must set the \n"
                        "``ignore_nearby_redshifts`` keyword argument to ``True``.\n"
                        )
                    raise HalotoolsError(msg)
            else:
                msg = ("\nThere is already a particle catalog in your cache log with metdata \n"
                    "that exactly matches the metadata of the catalog you are trying to store.\n"
                    "The filename of this matching particle catalog is:\n\n"
                    +exactly_matching_entries['fname'][0]+"\n\n"
                    "If this log entry is spurious, you should open the log \n"
                    "with a text editor and delete the offending line.\n"
                    "The log is stored at the following filename:\n\n"
                    +cache_fname+"\n\n"
                    "If this matching particle catalog is one you want to continue keeping track of, \n"
                    "then you should change the ``version_name`` \nof the particle catalog "
                    "you are trying to store.\n"
                    )
                raise HalotoolsError(msg)


    # At this point, we have ensured that the filename does not already exist 
    # and it is safe to consider it as a new log entry. 
    # Now we must verify the metadata that was passed in 
    # is consistent with the particle table contents. 

    try:
        x = ptcl_table['x']
        y = ptcl_table['y']
        z = ptcl_table['z']
    except KeyError:
        msg = ("\nAll particle tables must at least have the following columns:\n"
            "``x``, ``y``, ``z``\n")
        if first_ptcl_table_in_cache is True:
            # The cache log we created pointed to a 
            # bogus ptcl_table and so needs to be deleted
            os.system('rm ' + cache_fname)
        raise HalotoolsError(msg)

    # Check that Lbox properly bounds the particle positions
    try:
        assert np.all(x >= 0)
        assert np.all(y >= 0)
        assert np.all(z >= 0)
        assert np.all(x <= Lbox)
        assert np.all(y <= Lbox)
        assert np.all(z <= Lbox)
    except AssertionError:
        msg = ("\nThere are points in the input particle table that "
            "lie outside [0, Lbox] in some dimension.\n")
        if first_ptcl_table_in_cache is True:
            # The cache log we created pointed to a 
            # bogus ptcl_table and so needs to be deleted
            os.system('rm ' + cache_fname)
        raise HalotoolsError(msg)

    # The table appears to be kosher, so we write it to an hdf5 file, 
    # add metadata, and update the log
    ptcl_table.write(fname, path='data')

    f = h5py.File(fname)
    for key, value in metadata.iteritems():
        if type(value) == unicode:
            value = str(value)
        f.attrs.create(key, value)
    f.close()

    if first_ptcl_table_in_cache is False:
        new_table_entry = Table({'simname': [simname], 
            'redshift': [redshift], 
            'version_name': [version_name], 
            'fname': [fname]}
            )

        new_log = table_vstack([log, new_table_entry])
        overwrite_ptcl_table_cache_log(new_log, cache_fname = cache_fname)
        remove_repeated_ptcl_table_cache_lines(cache_fname = cache_fname)


def search_log_for_possibly_existing_entry(log, dz_tol = 0.05, **catalog_attrs):
    """
    """
    exact_match_mask = np.ones(len(log), dtype = bool)
    close_match_mask = np.ones(len(log), dtype = bool)

    for key, value in catalog_attrs.iteritems():
        exact_match_mask *= log[key] == value

    for key, value in catalog_attrs.iteritems():
        if key == 'redshift':
            close_match_mask *= abs(log[key] - value) < dz_tol
        else:
            close_match_mask *= log[key] == value

    return exact_match_mask, close_match_mask







