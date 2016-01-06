#!/usr/bin/env python

"""Command-line script to download the default halo catalog"""

import os
from halotools.sim_manager import DownloadManager, sim_defaults
from halotools.custom_exceptions import HalotoolsError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-overwrite", 
    help="Overwrite the existing halo catalog (if present)", 
    action="store_true")

parser.add_argument("simname", type = str, 
	choices = ['bolshoi', 'bolplanck', 'multidark', 'consuelo'], 
	help = "Nickname of the simulation")

parser.add_argument("halo_finder", type = str, help = "Nickname of the halo-finder. "
	"The `bdm` option is only available for `bolshoi`. ", 
	choices = ['rockstar', 'bdm'])

parser.add_argument("version_name", type = str, 
	choices = ['halotools_alpha_version1', 'most_recent'])

parser.add_argument("redshift", type = float, help = "Redshift of the snapshot. "
	"Options are 0, 0.5, 1 and 2, with slight variations from simulation to simulation.")

args = parser.parse_args()

existing_fname_error_msg = ("\n\nThe following filename already exists "
    "in your cache log: \n\n%s\n\n"
    "If you really want to overwrite the file, \n"
    "execute this script again but throw the ``-overwrite`` flag.\n\n")

simname = args.simname
halo_finder = args.halo_finder
version_name = args.version_name
redshift = args.redshift
if version == 'most_recent': version = sim_defaults.default_version_name

downman = DownloadManager()
