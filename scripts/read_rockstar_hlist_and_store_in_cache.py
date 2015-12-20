#!/usr/bin/env python
import argparse
import os
import numpy as np
from halotools.sim_manager import RockstarHlistReader
from halotools.sim_manager import manipulate_cache_log
from halotools.custom_exceptions import HalotoolsError
from astropy.table import Table

parser = argparse.ArgumentParser()

input_fname_help_msg = ("Absolute path to the ASCII hlist file storing the halo catalog. "
	"Can be either compressed or uncompressed.")
columns_to_keep_fname_help_msg = ("Absolute path to the "
	"``columns_to_keep_fname`` ASCII file defining "
	"the cuts made on the rows and columns of the hlist file. "
	"As described in the docstring of the RockstarHlistReader class, "
	"the contents of this file are used to select the relevant columns of the hlist file. "
	"The ``columns_to_keep_fname`` file can begin with any number of header lines "
	"beginning with the '#' character, all of which will be ignored. "
	"Besides the header, there should be one row of the ``columns_to_keep_fname`` file "
	"for every column of the hlist file you wish to keep. "
	"Each row of the ``columns_to_keep_fname`` file must have 3 columns. "
	"Column 1: integer providing the index of the column in the hlist file, "
	"where the first column has column-index 0. "
	"Column 2: string providing the name of the column data, e.g., 'halo_id'. "
	"The Halotools convention is for all such strings to begin with 'halo_'. "
	"If you wish store your catalog in the Halotools cache and use it to populate mocks, "
	"you will need to follow this convention; otherwise you may ignore it. "
	"Column 3: string defining the data type stored in the column, e.g., 'f4' for floats,"
	"'f8' for double, 'i4' for int, i8' for long. "
	"The contents of the ``columns_to_keep_fname`` will be stored as metadata of the "
	"resulting hdf5 file, so that you have an exact record of how the hlist file "
	"was reduced into an hdf5 file. See halotools/data/RockstarHlistReader_input_example.dat "
	"for an example ``columns_to_keep_fname``. ")

version_name_msg = ("Nickname used to distinguish between different versions of the same "
	"snapshot. You should use a different version_name for every different version of Rockstar "
	"run on the snapshot, and also for every different time you process the same the halo catalog "
	"with different cuts. ")
parser.add_argument("input_fname", help=input_fname_help_msg)
parser.add_argument("columns_to_keep_fname", help=columns_to_keep_fname_help_msg)
parser.add_argument("simname", help = "Nickname of the simulation. ")
parser.add_argument("redshift", help = "Redshift of the snapshot. ", type=float)
parser.add_argument("version_name", help = version_name_msg)
parser.add_argument("output_fname", help = "Absolute path of the hdf5 file storing the "
	"processed halo catalog.")

parser.add_argument("--overwrite", 
	help="If an hdf5 file with a matching ``output_fname`` exists, overwrite it.", 
	action="store_true")

parser.add_argument("--ignore_nearby_redshifts", action = "store_true", 
	help = "If there are catalogs in your existing cache that match all your "
	"input metadata and have a very similar redshift, you must throw the --ignore_nearby_redshifts "
	"flag in order to use the script to generate a new halo catalog. "
	)



# parser.add_argument("--min_row_value", nargs=2, help = "Use this optional argument to "
# 	"make a cut on some column of the halo catalog on-the-fly as the hlist file is being read. "
# 	"There must be two arguments that follow each appearance of the --min_row_value flag. "
# 	"The first argument is a string giving the column name upon which a cut will be placed. "
# 	"This string must appear in the first column of the ``columns_to_keep_fname`` ascii file: "
# 	"for the sake of good bookkeeping, it is not permissible to "
# 	"place a cut on a column that you do not keep. "
# 	"The second argument defines the lower bound on the data in the column; "
# 	"all halos with a value of the requested column below this cut will be ignored. "
# 	"The --min_row_value flag can appear as many times in the call to the script as you like. "
# 	"Only rows passing all cuts will be accepted.")

# parser.add_argument("--max_row_value", nargs=2, help = "Use this optional argument to "
# 	"make a cut on some column of the halo catalog on-the-fly as the hlist file is being read. "
# 	"There must be two arguments that follow each appearance of the --max_row_value flag. "
# 	"The first argument is a string giving the column name upon which a cut will be placed. "
# 	"This string must appear in the first column of the ``columns_to_keep_fname`` ascii file: "
# 	"for the sake of good bookkeeping, it is not permissible to "
# 	"place a cut on a column that you do not keep. "
# 	"The second argument defines the upper bound on the data in the column; "
# 	"all halos with a value of the requested column above this cut will be ignored. "
# 	"The --max_row_value flag can appear as many times in the call to the script as you like. "
# 	"Only rows passing all cuts will be accepted.")

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
		"This file already exists. "
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
	num_exact_matches = len(exactly_matching_fnames_in_log)

	_, close_redshift_match_mask = (
		manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
		simname = simname, halo_finder = 'rockstar', redshift = redshift, 
		version_name = version_name)
		)
	closely_matching_fnames_in_log = log[close_redshift_match_mask]
	num_close_matches = len(closely_matching_fnames_in_log)

	if num_exact_matches== 0:
		if num_close_matches == 0:
			pass
		else:
			if not args.ignore_nearby_redshifts:
				msg = ("\n\nHalotools detected the following cached halo catalogs that match \n"
					"all your input metadata, and have have a very similar redshift:\n\n")
				for fname in closely_matching_fnames_in_log['fname']:
					msg += fname + "\n"
				msg += ("\nIf you wish to ignore these closely matching catalogs, \n"
					"you must throw the ``--ignore_nearby_redshifts`` flag.\n"
					)
				raise HalotoolsError(msg)

	elif num_exact_matches ==  1:
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





