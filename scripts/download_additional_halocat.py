#!/usr/bin/env python

"""Command-line script to download the default halo catalog"""

import os
from halotools.sim_manager import DownloadManager, sim_defaults
from halotools.custom_exceptions import HalotoolsError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", 
    help="Overwrite the existing halo catalog (if present)", 
    action="store_true")
args = parser.parse_args()

existing_fname_error_msg = ("\n\nThe following filename already exists "
    "in your cache log: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "execute this script again but throw the ``--overwrite`` flag.\n\n")

simname = sim_defaults.default_simname
halo_finder = sim_defaults.default_halo_finder
redshift = sim_defaults.default_redshift
version_name = sim_defaults.default_version_name

downman = DownloadManager()
