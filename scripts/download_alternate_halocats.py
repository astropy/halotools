#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line script to download the default halo catalog"""

import sys
from halotools.sim_manager import CatalogManager, sim_defaults
from halotools.halotools_exceptions import HalotoolsError

existing_fname_error_msg = ("\n\nThe following filename already exists in your cache directory: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "simply execute this script again but using ``-overwrite`` as a command-line argument.\n\n")

command_line_arg_error_msg = ("\n\nThe download_alternate_halocat.py script should be called "
    "with three arguments.\n"
    "The first argument gives the simname, for which you have the following options:\n"
    "``bolshoi``, ``bolplanck``, ``multidark``, or ``consuelo``. \n"
    "The second argument gives the redshift of the desired snapshot; you can choose between z=0, 0.5, 1, or 2. \n"
    "The third argument specifies the halo-finder, for which your options are ``bdm`` or ``rockstar``, \n"
    "though presently ``bdm`` halos are only available for the ``bolshoi`` simulation.\n\n"
    "You may also throw flags for several options.\n"
    "The ``-overwrite`` flag can be thrown in case your cache directory already contains the "
    "default processed halo catalog, and you want to overwrite it with a new download.\n"
    "The ``-noptcl`` flag can be thrown in case you do not to download the random downsampling "
    "of dark matter particles that accompony each halo catalog by default.\n"
    "If the ``-help`` flag is thrown, the message you are currently reading will be reproduced and the "
    "script will take no action.\n\n")

def main(flags):
    """ args is a python list. Element 0 is the name of the module. 
    The remaining elements are the command-line arguments as strings. 
    """

    if '-help' in flags:
        print(command_line_arg_error_msg)
        return 

    if len(flags) < 4:
        print("\n\n HalotoolsError: The download_alternate_halocat.py script was called with incorrect arguments.\n\n")
        print(command_line_arg_error_msg)
        return 

    simname = str(flags[1])
    redshift = float(flags[2])
    halo_finder = str(flags[3])

    if '-overwrite' in flags:
        overwrite = True
    else:
        overwrite = False

    catman = CatalogManager()

    catman.download_processed_halo_table(simname = simname, 
        halo_finder = halo_finder, desired_redshift = redshift, 
        initial_download_script_msg = existing_fname_error_msg, 
        overwrite = overwrite)

    if '-noptcl' not in flags:
        catman.download_ptcl_table(simname = simname, 
            desired_redshift = redshift, dz_tol = 0.05)

        msg = ("\n\nYour Halotools cache directory now has two hdf5 files, \n"
            "one storing a z = %.2f %s halo catalog for the %s simulation, \n"
            "another storing a random downsampling of ~1e6 dark matter particles from the same snapshot.\n"
            "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
            ">>> from halotools.sim_manager import HaloCatalog\n"
            ">>> halocat = HaloCatalog(simname = your_chosen_simname, redshift = your_chosen_redshift, halo_finder = your_chosen_halo_finder)\n"
            ">>> halos = halocat.halo_table\n"
            ">>> particles = halocat.ptcl_table\n\n")
    else:
        msg = ("\n\nYour Halotools cache directory a new hdf5 file \n"
            "storing a z = %.2f %s halo catalog for the %s simulation. \n"
            "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
            ">>> from halotools.sim_manager import HaloCatalog\n"
            ">>> halocat = HaloCatalog(simname = your_chosen_simname, redshift = your_chosen_redshift, halo_finder = your_chosen_halo_finder)\n"
            ">>> halos = halocat.halo_table\n")

    print(msg % (abs(redshift), halo_finder, simname))


###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

