#!/usr/bin/env python
import argparse
import os
import numpy as np
from halotools.sim_manager import RockstarHlistReader
from halotools.sim_manager import manipulate_cache_log
from halotools.custom_exceptions import HalotoolsError
from astropy.table import Table

parser = argparse.ArgumentParser()

parser.add_argument("input_fname")
parser.add_argument("columns_to_keep_fname")
parser.add_argument("simname")
parser.add_argument("redshift", type=float)
parser.add_argument("version_name")
parser.add_argument("output_fname")

parser.add_argument("--overwrite", 
	help="If an hdf5 file with a matching ``output_fname`` exists, overwrite it.", 
	action="store_true")

args = parser.parse_args()

input_fname = args.input_fname
columns_to_keep_fname = args.columns_to_keep_fname
simname = args.simname
halo_finder = 'rockstar'
redshift = args.redshift
version_name = args.version_name
output_fname = args.output_fname

####################################################################
# The input input_fname must point to an existing file
try:
	assert os.path.isfile(input_fname)
except AssertionError:
	msg = ("\nThe ``input_fname`` argument should be the name of the file storing the ascii hlist.\n"
		"You called the script with the following ``input_fname``:\n"
	+input_fname+"\nThis file does not exist.\n")
	raise HalotoolsError(msg)

####################################################################
# The input columns_to_keep_fname must point to an existing file
try:
	assert os.path.isfile(columns_to_keep_fname)
except AssertionError:
	msg = ("\nThe ``columns_to_keep_fname`` argument should be the name of the file \n"
		"used to determine how the hlist file is processed.\n"
		"If you are unsure of the purpose of this file, run this script again and throw the --help flag.\n"
		"You called the script with the following ``columns_to_keep_fname`` filename:\n"
	+columns_to_keep_fname+"\nThis file does not exist.\n")
	raise HalotoolsError(msg)

####################################################################
# If the output_fname points to an existing file, the --overwrite flag must be thrown
if (os.path.isfile(output_fname)) & (not args.overwrite):
	msg = ("\n\nYou ran this script with the following argument for the ``output_fname``:\n\n"
		+ output_fname + "\n\n"
		"This file already exists."
		"If you want to overwrite this file, \nrun this script again and throw the ``--overwrite`` flag.\n\n"
		)
	raise HalotoolsError(msg)

####################################################################
# Determine whether there is an existing cache log. 
# If present, clean it of the possible presence of exactly duplicate lines 
try: 
	manipulate_cache_log.verify_halo_table_cache_existence()
	manipulate_cache_log.remove_repeated_cache_lines()
	has_existing_cache_log = True
except HalotoolsError:
	has_existing_cache_log = False

####################################################################
### If there is a pre-existing cache log, 
# make sure there are no entries with a filename that 
# exactly matches the input ``output_fname`` argument to the script

if has_existing_cache_log:
	manipulate_cache_log.verify_cache_log()
	log = manipulate_cache_log.read_halo_table_cache_log()

	exact_fname_match_mask, _ = (
		manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
		fname = output_fname)
		)
	exactly_matching_fnames_in_log = log[exact_fname_match_mask]
	if len(exactly_matching_fnames_in_log) == 0:
		pass
	elif len(exactly_matching_fnames_in_log) ==  1:
		matching_fname = exactly_matching_fnames_in_log['fname'][0]
		if args.overwrite:
			manipulate_cache_log.remove_unique_fname_from_halo_table_cache_log(matching_fname)
		else:
			matching_simname = exactly_matching_fnames_in_log['simname'][0]
			matching_halo_finder = exactly_matching_fnames_in_log['halo_finder'][0]
			matching_redshift = exactly_matching_fnames_in_log['redshift'][0]
			matching_version_name = exactly_matching_fnames_in_log['version_name'][0]
			msg = ("\n\nYour Halotools cache log has an existing entry with a filename \n"
				"that matches the ``output_fname`` you chose when executing this script.\n"
				"The existing entry in the log points to a halo catalog with the following attributes:\n\n")
			msg += "simname = " + matching_simname + "\n"
			msg += "halo_finder = " + matching_halo_finder + "\n"
			msg += "redshift = " + str(np.round(matching_redshift, 4)) + "\n"
			msg += "version_name = " + matching_version_name + "\n"
			msg += "fname = " + matching_fname + "\n"

			idx = np.where(exact_fname_match_mask == True)[0]
			linenum = idx[0] + 2
			msg += ("\nIf the existing log entry is still valid, and you want to overwrite the existing file,\n"
				"just call this script again and throw the ``--overwrite`` flag.\n"
				"Alternatively, you can choose a different ``output_fname`` when calling this script.\n"
				"\nHowever, if this log entry is invalid, \n"
				"use a text editor to open the log and delete the entry.\n"
				"The cache log is stored in the following file:\n"
				+str(manipulate_cache_log.get_halo_table_cache_log_fname())+"\n"
				"The relevant line to change is line #" + str(linenum) + ",\n"
				"where the first line of the file is line #1.\n\n"
				)
			raise HalotoolsError(msg)

	else:
		idx = np.where(exact_fname_match_mask == True)[0]
		linenums = idx + 2
		msg = ("\n\nYou ran this script with the following ``output_fname``:\n\n"
			+ output_fname + "\n\n"
		"Your existing Halotools cache log has multiple entries pointing to this file,\n"
		"and the entries in the log have mutually inconsistent metadata.\n"
		"You must address this problem before you can proceed.\n"
		"Use a text editor to open up the cache log, which is stored in the following location:\n"
		+str(manipulate_cache_log.get_halo_table_cache_log_fname())+"\n"
		"The conflicting line numbers are line #")
		for line in linenums:
			msg += str(line) + ", "
		msg += ("where the first line of the file is line #1.\n"
			"At most one of these lines can be storing the correct metadata.\n"
			"After deleting the incorrect line(s), re-run this script.\n\n"
			)
		raise HalotoolsError(msg)

####################################################################
# There are zero entries in the log with an ``fname`` that matches 
# the ``output_fname`` passed to the script. 
# Now address the case where there may be log entries with matching metadata

if has_existing_cache_log:
	manipulate_cache_log.verify_cache_log()
	log = manipulate_cache_log.read_halo_table_cache_log()

	exact_attr_match_mask, close_redshift_match_mask = (
		manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
		simname = simname, halo_finder = halo_finder, 
		redshift = redshift, version_name = version_name)
		)

















# If there is no existing log, create one with a single entry
# try: 
# 	manipulate_cache_log.verify_halo_table_cache_existence()
# except HalotoolsError:
# 	new_log = Table({'simname': [simname], 
# 		'halo_finder': [halo_finder], 
# 		'redshift': [redshift], 
# 		'version_name': [version_name], 
# 		'fname': [output_fname]}
# 		)
# 	manipulate_cache_log.overwrite_halo_table_cache_log(log)





