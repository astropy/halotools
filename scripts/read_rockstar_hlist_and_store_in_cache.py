import argparse
import os
from halotools.sim_manager import RockstarHlistReader
from halotools.sim_manager import manipulate_cache_log
from halotools.custom_exceptions import HalotoolsError
from astropy.table import Table

parser = argparse.ArgumentParser()

parser.add_argument("input_fname")
parser.add_argument("columns_to_keep_fname")
parser.add_argument("simname")
parser.add_argument("redshift")
parser.add_argument("version_name")
parser.add_argument("output_fname")

args = parser.parse_args()

input_fname = args.input_fname
columns_to_keep_fname = args.columns_to_keep_fname
simname = args.simname
halo_finder = 'rockstar'
redshift = args.redshift
version_name = args.version_name
output_fname = args.output_fname

try:
	assert os.path.isfile(input_fname):
except AssertionError:
	msg = ("\nThe ``input_fname`` argument should be the name of the file storing the ascii hlist.\n"
		"You called the script with the following ``input_fname``:\n"
	+input_fname+"\nThis file does not exist.\n")
	raise HalotoolsError(msg)

try:
	assert os.path.isfile(columns_to_keep_fname):
except AssertionError:
	msg = ("\nThe ``columns_to_keep_fname`` argument should be the name of the file \n"
		"used to determine how the hlist file is processed.\n"
		"If you are unsure of the purpose of this file, run this script again and throw the --help flag.\n"
		"You called the script with the following ``columns_to_keep_fname`` filename:\n"
	+columns_to_keep_fname+"\nThis file does not exist.\n")
	raise HalotoolsError(msg)


### If there is a pre-existing cache log, 
# make sure there are no entries matching the input specifications
try: 
	manipulate_cache_log.verify_halo_table_cache_existence()
	has_existing_cache_log = True
except HalotoolsError:
	has_existing_cache_log = False

if has_existing_cache_log:
	manipulate_cache_log.verify_cache_log()
	log = manipulate_cache_log.read_halo_table_cache_log()

	exact_fname_match_mask, _ = (
		manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
		fname = output_fname)
		)
	exactly_matching_fnames_in_log = log[exact_fname_match_mask]
	if len(exactly_matching_fnames_in_log) > 0:
		matching_simname = exactly_matching_fnames_in_log['simname']
		matching_halo_finder = exactly_matching_fnames_in_log['halo_finder']
		matching_redshift = exactly_matching_fnames_in_log['redshift']
		matching_version_name = exactly_matching_fnames_in_log['version_name']
		msg = ("\nYour Halotools cache log has an existing entry with a filename \n"
			"that matches the ``input_fname`` you chose when executing this script.\n"
			"The existing entry in the log points to a halo catalog with the following attributes:\n\n")
		msg += "simname = " + matching_simname + "\n"
		msg += "halo_finder = " + matching_halo_finder + "\n"
		msg += "redshift = " + matching_redshift + "\n"
		msg += "version_name = " + matching_version_name + "\n"

		idx = np.where(exact_fname_match_mask == True)[0]
		linenum = idx[0] + 2
		msg += ("\nIf the existing log entry is still valid, \n"
			"then you can proceed by using a different ``version_name`` when calling this script.\n"
			"If this log entry is invalid, use a text editor to open the log and the entry.\n"
			"The cache log is stored in the following file:\n"
			+str(manipulate_cache_log.get_halo_table_cache_log_fname())+"\n"
			"The relevant line to change is line #" + str(linenum) + ",\n"
			"where the first line of the file is line #1.\n\n"
			)
		raise HalotoolsError(msg)


	# exact_attr_match_mask, close_redshift_match_mask = (
	# 	manipulate_cache_log.search_log_for_possibly_existing_entry(log, 
	# 	simname = simname, halo_finder = halo_finder, 
	# 	redshift = redshift, version_name = version_name)
	# 	)


















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





