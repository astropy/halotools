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
    exact_match_mask = np.ones(len(log), dtype=bool)
    exact_match_mask *= log['simname'] == simname
    exact_match_mask *= log['version_name'] == version_name
    exact_match_mask *= log['redshift'] == redshift
    exact_matches = log[exact_match_mask]

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

    if len(exact_matches) == 0:

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
            "corresponding entry of `ptcl_table_cache_log.txt`\n"
            "Always save a backup version of the log before making manual changes.\n")
        raise HalotoolsError(msg)

    elif len(exact_matches) == 1:
        idx = np.where(exact_match_mask == True)[0]
        linenum = idx[0] + 2
        check_ptcl_table_metadata_consistency(exact_matches[0], linenum = linenum)
        fname = exact_matches['fname'][0]
        return fname
    else:
        msg = ("\nHalotools detected multiple particle catalogs matching "
            "the input arguments.\n"
            "Now printing the list of all catalogs matching your requested specifications:\n")
        for entry in close_matches:
            msg += entry['fname'] + "\n"
        msg += ("Please delete the erroneous lines from the log before proceeding.\n")
        raise HalotoolsError(msg)

def check_ptcl_table_metadata_consistency(cache_log_entry, linenum = None):
    """
    """
    try:
        import h5py
    except ImportError:
        raise HalotoolsError("\nYou must have h5py installed in order to use "
            "the Halotools halo catalog cache system.\n")

    ptcl_table_fname = cache_log_entry['fname']
    if os.path.isfile(ptcl_table_fname):
        f = h5py.File(ptcl_table_fname)
    else:
        msg = ("\nYou requested to load a halo catalog "
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
            "If you have not yet downloaded the initial halo catalog,\n"
            "you should do so now following the ``Getting Started`` instructions on "
            "http://halotools.readthedocs.org\nIf you have already taken this step,\n"
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
                "\nkeeps track of the halo catalogs"
                "you use with Halotools.\n"
                "This file appears to be corrupted.\n"
                "Please visually inspect this file to ensure it has not been "
                "accidentally overwritten. \n"
                "Then store a backup of this file and execute the following script:\n"
                "halotools/scripts/rebuild_ptcl_table_cache_log.py\n"
                "If this does not resolve the error you are encountering,\n"
                "and if you have been using halo catalogs stored on some external disk \n"
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
            "\nkeeps track of the halo catalogs"
            "you use with Halotools.\n"
            "This file appears to be corrupted.\n"
            "Please visually inspect this file to ensure it has not been "
            "accidentally overwritten. \n"
            "Then store a backup of this file and execute the following script:\n"
            "halotools/scripts/rebuild_ptcl_table_cache_log.py\n"
            "If this does not resolve the error you are encountering,\n"
            "and if you have been using halo catalogs stored on some external disk \n"
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
            "so that the halo table is accessible with the following syntax:\n"
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

    # Check that Lbox properly bounds the halo positions
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









