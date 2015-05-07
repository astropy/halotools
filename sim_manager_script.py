#!/usr/bin/env python

import numpy as np
import os

from halotools import sim_manager
reload(sim_manager)

catman = sim_manager.CatalogManager()

#############################################################################
####### CHECK THAT WE CAN LOCATE AVAILABLE AND CLOSEST HALO CATALOGS #######

supported_halocats = catman.available_halocats
simname, halo_finder = supported_halocats[0]

location = 'web'
catalog_type = 'raw_halos'
desired_redshift = 1
closest_cat_on_web = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
#print("\n Closest matching catalog on the web = \n%s\n " % closest_cat_on_web[0])
location = 'cache'
closest_cat_in_cache = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
#print("\n Closest matching catalog in cache = \n%s\n " % closest_cat_in_cache[0])

print("\n")
all_cached_files = catman.all_halocats_in_cache('halos')
#for f in all_cached_files: print(f)
#############################################################################


#############################################################################
####### CHECK THAT RAW HALO CATALOG DOWNLOADS GO TO CACHE #######
desired_redshift = 11.5
simname = 'bolshoi'
closest_cat_on_web = catman.closest_halocat('web', 'raw_halos', simname, halo_finder, desired_redshift)
print("\nFor simname = %s and redshift = %.2f, " % (simname, desired_redshift))
print("Closest halocat available for download: \n%s\n" % closest_cat_on_web[0])
is_in_cache = catman.check_for_existing_halocat('cache', closest_cat_on_web[0], 'raw_halos', simname, halo_finder)
print("Is the file already in cache? %r " % is_in_cache)

if is_in_cache == False:
	print("\n... downloading file...\n")
	catman.download_raw_halocat(simname, halo_finder, closest_cat_on_web[1], overwrite = False)
	is_in_cache = catman.check_for_existing_halocat('cache', closest_cat_on_web[0], 'raw_halos', simname, halo_finder)
	print("Is the file now in cache? %r " % is_in_cache)

closest_cat_in_cache = catman.closest_halocat('cache', 'raw_halos', simname, halo_finder, desired_redshift)
if closest_cat_in_cache[0] != is_in_cache:
	print("closest_halocat method failed to detect the following catalog: \n%s\n" % is_in_cache)



#############################################################################
####### CHECK THE HALOCAT_OBJ PROPERTIES #######
halocat_obj = sim_manager.read_nbody.get_halocat_obj(simname, halo_finder)
webloc = halocat_obj.raw_halocat_web_location




#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
print("\n\n\n\n")
