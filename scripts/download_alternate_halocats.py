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
    "The ``-halosonly`` flag can be thrown in case you do not to download the random downsampling "
    "of dark matter particles that accompany each halo catalog by default.\n"
    "The ``-ptclsonly`` flag can be thrown in case you only want the particles.\n"
    "If the ``-help`` flag is thrown, the message you are currently reading will be reproduced and the "
    "script will take no action.\n\n")

def main(command_line_args):
    """ args is a python list. Element 0 is the name of the module. 
    The remaining elements are the command-line arguments as strings. 
    """

    args = [arg for arg in command_line_args if (arg[0] != '-') & (arg != 'scripts/download_alternate_halocats.py')]
    opts = [opt for opt in command_line_args if opt[0]=='-']

    if '-help' in opts:
        print(command_line_arg_error_msg)
        return 

    if '-ptclsonly' in opts:
        # for a in args:
        #     print a
        if len(args) != 2:
            msg = ("\n\nHalotoolsError: \nWhen throwing the -ptclsonly flag during a call to the "
            "download_alternate_halocat.py script, \nyou must specify a simname and redshift, "
            "and only those two quantities.\n"
            "Now printing the -help message for further details.\n")
            print(msg)
            print(command_line_arg_error_msg)
            return 
        else:
            try:
                simname = str(args[0])
                redshift = float(args[1])
            except ValueError:
                msg = ("\n\nHalotoolsError: \nWhen throwing the -ptclsonly flag during a call to the "
                "download_alternate_halocat.py script, \nyou must specify a simname and redshift, "
                "and only those two quantities.\n"
                "Now printing the -help message for further details.\n")
                print(msg)
                print(command_line_arg_error_msg)
                return 
            simname = str(args[0])
            redshift = float(args[1])
    else:
        msg = ("\n\nHalotoolsError: \nWhen running the "
        "download_alternate_halocat.py script, \n"
        "you must specify a simname, redshift, and halo-finder, and only those three quantities.\n"
        "Now printing the -help message for further details.\n")
        if len(args) != 3:
            print(msg)   
            print(command_line_arg_error_msg)         
            return 
        else:
            try:
                simname = str(args[0])
                redshift = float(args[1])
                halo_finder = str(args[2])
            except ValueError:
                msg = ("\n\nHalotoolsError: \nWhen running the "
                "download_alternate_halocat.py script, you must specify a simname, redshift, and halo-finder.\n"
                "Now printing the -help message for further details.\n")
                print(msg)
                print(command_line_arg_error_msg)
                return 


    if '-overwrite' in opts:
        overwrite = True
    else:
        overwrite = False

    catman = CatalogManager()

    if '-halosonly' in opts:
        msg = ("\n\nYour Halotools cache directory a new hdf5 file \n"
            "storing a z = %.2f %s halo catalog for the %s simulation. \n"
            "\nHalotools can load this catalog into memory with the following syntax:\n\n"
            ">>> from halotools.sim_manager import HaloCatalog\n"
            ">>> halocat = HaloCatalog(simname = your_chosen_simname, redshift = your_chosen_redshift, halo_finder = your_chosen_halo_finder)\n"
            ">>> halos = halocat.halo_table\n" % (abs(redshift), halo_finder, simname))
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg, 
            overwrite = overwrite, success_msg = msg)
    elif '-ptclsonly' in opts:
        msg = ("\n\nYour Halotools cache directory a new hdf5 file \n"
            "storing a z = %.2f particle catalog for the %s simulation. \n"
            "\nHalotools can load this catalog into memory with the following syntax:\n\n"
            ">>> from halotools.sim_manager import HaloCatalog\n"
            ">>> halocat = HaloCatalog(simname = your_chosen_simname, redshift = your_chosen_redshift)\n"
            ">>> particles = halocat.ptcl_table\n" % (abs(redshift), simname))
        catman.download_ptcl_table(simname = simname, 
            desired_redshift = redshift, dz_tol = 0.05, success_msg = msg, 
            initial_download_script_msg = existing_fname_error_msg)
    else:
        msg = ("\n\nYour Halotools cache directory now has two hdf5 files, \n"
            "one storing a z = %.2f %s halo catalog for the %s simulation, \n"
            "another storing a random downsampling of ~1e6 dark matter particles from the same snapshot.\n"
            "\nHalotools can load these catalogs into memory with the following syntax:\n\n"
            ">>> from halotools.sim_manager import HaloCatalog\n"
            ">>> halocat = HaloCatalog(simname = your_chosen_simname, redshift = your_chosen_redshift, halo_finder = your_chosen_halo_finder)\n"
            ">>> halos = halocat.halo_table\n"
            ">>> particles = halocat.ptcl_table\n\n" % (abs(redshift), halo_finder, simname))
        catman.download_processed_halo_table(simname = simname, 
            halo_finder = halo_finder, desired_redshift = redshift, 
            initial_download_script_msg = existing_fname_error_msg, 
            overwrite = overwrite)
        catman.download_ptcl_table(simname = simname, 
            desired_redshift = redshift, dz_tol = 0.05, 
            success_msg = msg, initial_download_script_msg = existing_fname_error_msg)



###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
        main(sys.argv)

