#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

import sys, os
from halotools.sim_manager import DownloadManager, sim_defaults
from halotools.custom_exceptions import HalotoolsError, UnsupportedSimError

existing_fname_error_msg = ("\n\nThe following filename already exists in your cache directory: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "simply execute this script again but using ``-overwrite`` as a command-line argument.\n\n")

command_line_arg_error_msg = ("\n\nThe only command-line argument recognized by the "
    "download_initial_halocat script is ``-overwrite``.\n"
    "The -overwrite flag should be thrown in case your cache directory already contains the "
    "default processed halo catalog, and you want to overwrite it with a new download.\n\n")

def main(flags):
    """ args is a python list. Element 0 is the name of the module. 
    The remaining elements are the command-line arguments as strings. 
    """

    simname = sim_defaults.default_simname
    halo_finder = sim_defaults.default_halo_finder
    redshift = sim_defaults.default_redshift

    downman = DownloadManager()

    if len(flags) == 1:
        new_halo_log_entry = downman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg)
        new_ptcl_log_entry = downman.download_ptcl_table(simname = simname, 
            redshift = redshift, dz_tol = 0.05, 
            initial_download_script_msg = existing_fname_error_msg)

    elif (len(flags) == 2) & (flags[1] == '-overwrite'):
        new_halo_log_entry = downman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg, 
            overwrite = True)
        new_ptcl_log_entry = downman.download_ptcl_table(simname = simname, 
            redshift = redshift, dz_tol = 0.05, overwrite=True, 
            initial_download_script_msg = existing_fname_error_msg)
    else:
        raise HalotoolsError(command_line_arg_error_msg)

    cache_dirname = str(os.path.dirname(downman.halo_table_cache.cache_log_fname)).strip()
    halo_table_cache_basename = str(os.path.basename(downman.halo_table_cache.cache_log_fname))
    ptcl_table_cache_basename = str(os.path.basename(downman.ptcl_table_cache.cache_log_fname))

    msg = (
        "By running the initial download script, you have set up the Halotools cache \n"
        "in the following location on disk:\n\n" + cache_dirname + "\n\n"
        "The directory now contains the following two cache log files: \n\n" + 
        str(downman.halo_table_cache.cache_log_fname) + "\n" + 
        str(downman.ptcl_table_cache.cache_log_fname) + "\n\n"
        "These two ASCII files maintain a record of the \nhalo and particle catalogs "
        "you use with Halotools.\n"
        "The "+halo_table_cache_basename+" cache log now has a single entry \n"
        "reflecting the default halo catalog you just downloaded; "
        "\nthe halo catalog is now stored in the following location:\n\n"
        + new_halo_log_entry.fname + "\n\n"
        "The "+ptcl_table_cache_basename+" cache log also has a single entry \n"
        "corresponding to a random downsampling of ~1e6 dark matter particles from the same snapshot; "
        "\nthe particle catalog is now stored in the following location:\n\n"
        + new_ptcl_log_entry.fname + "\n\n"
        "Both hdf5 files store an Astropy Table data structure. \n"
        "\nThe Halotools cache system allows you to \n"
        "load these catalogs into memory with the following syntax:\n\n"
        ">>> from halotools.sim_manager import OverhauledHaloCatalog\n"
        ">>> bolshoi_z0 = OverhauledHaloCatalog()\n"
        ">>> halos = bolshoi_z0.halo_table\n"
        ">>> particles = bolshoi_z0.ptcl_table\n\n")


    print(msg)
    print("\a\a")


###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

