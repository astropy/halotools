#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

import sys
from halotools.sim_manager import CatalogManager, sim_defaults
from halotools.halotools_exceptions import HalotoolsError

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

    catman = CatalogManager()

    if len(flags) == 1:
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg)
    elif (len(flags) == 2) & (flags[1] == '-overwrite'):
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg, 
            overwrite = True)
    else:
        raise HalotoolsError(command_line_arg_error_msg)



###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

