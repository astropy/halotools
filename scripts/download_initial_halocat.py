#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

import sys
from halotools.sim_manager import CatalogManager, sim_defaults
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

    catman = CatalogManager()

    if len(flags) == 1:
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg)
        catman.download_ptcl_table(simname = simname, 
            desired_redshift = redshift, dz_tol = 0.05)

    elif (len(flags) == 2) & (flags[1] == '-overwrite'):
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg, 
            overwrite = True)
        catman.download_ptcl_table(simname = simname, 
            desired_redshift = redshift, dz_tol = 0.05, overwrite=True)
    else:
        raise HalotoolsError(command_line_arg_error_msg)

    msg = ("\n\nYour Halotools cache directory now has two hdf5 files, \n"
        "one storing a z = %.2f %s halo catalog for the %s simulation, \n"
        "another storing a random downsampling of ~1e6 dark matter particles from the same snapshot.\n"
        "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
        ">>> from halotools.sim_manager import HaloCatalog\n"
        ">>> bolshoi_z0 = HaloCatalog()\n"
        ">>> halos = bolshoi_z0.halo_table\n"
        ">>> particles = bolshoi_z0.ptcl_table\n\n")

    print(msg % (abs(redshift), halo_finder, simname))


###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

