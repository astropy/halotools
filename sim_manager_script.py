#!/usr/bin/env python

import numpy as np
import os
from halotools import sim_manager



catman = sim_manager.CatalogManager()

#############################################################################
####### CHECK THAT WE CAN LOCATE AVAILABLE AND CLOSEST HALO CATALOGS #######

"""
supported_halocats = catman.available_halocats
simname, halo_finder = supported_halocats[0]
catalog_type = 'raw_halos'
desired_redshift = 1


location = 'web'
closest_cat_on_web = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
print("\n Closest matching catalog on the web = \n%s\n " % closest_cat_on_web[0])

location = 'cache'
closest_cat_in_cache = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
print("\n Closest matching catalog in cache = \n%s\n " % closest_cat_in_cache[0])

location='/Volumes/NbodyDisk1/raw_halos/bolshoi/rockstar'
closest_cat_in_cache = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
print("\n Closest matching catalog on external disk = \n%s\n " % closest_cat_in_cache[0])




"""
#############################################################################
# CHECK THE CONVENIENCE FUNCTION ALL_HALOCATS_IN_CACHE
print("\n")
all_cached_files = catman.all_halocats_in_cache('halos')
#for f in all_cached_files: print(f)


#############################################################################
####### CHECK THAT RAW HALO CATALOG DOWNLOADS GO TO CACHE #######
catalog_type = 'raw_halos'
desired_redshift = 12.083
simname = 'bolshoi'
halo_finder = 'rockstar'
disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)

catlist = catman.available_snapshots(disk_location, catalog_type, simname, halo_finder)
print("\n All catalogs on external disk:\n")
for f in catlist:
	print f

### Check the web
closest_cat_on_web = catman.closest_halocat('web', catalog_type, simname, halo_finder, desired_redshift)
print("\nFor simname = %s and redshift = %.2f, " % (simname, desired_redshift))
print("Closest halocat available for download: \n%s\n" % closest_cat_on_web[0])

### Check the cache
is_in_cache = catman.check_for_existing_halocat('cache', closest_cat_on_web[0], catalog_type, simname, halo_finder)
print("Is the file already in cache?\n %r \n" % is_in_cache)

### Check the external disk
is_on_disk = catman.check_for_existing_halocat(disk_location, closest_cat_on_web[0], catalog_type, simname, halo_finder)
print("Is the file stored on disk?\n %r \n" % is_on_disk)
if is_on_disk == False:
	output_fname = catman.download_raw_halocat(simname, halo_finder, closest_cat_on_web[1], 
		overwrite = False, download_loc = disk_location)
	print("\n The following fname was just downloaded: \n%s\n" % output_fname)
	print("\n This fname should agree with: \n%s\n" % closest_cat_on_web[0])
	print("Can we detect this newly downloaded file on disk?\n")
	is_on_disk = catman.check_for_existing_halocat(
		disk_location, closest_cat_on_web[0], catalog_type, simname, halo_finder)
	print is_on_disk
	print ("\n")







if is_in_cache == False:
	print("\n... downloading file...\n")
	catman.download_raw_halocat(simname, halo_finder, closest_cat_on_web[1], overwrite = False)
	is_in_cache = catman.check_for_existing_halocat('cache', closest_cat_on_web[0], catalog_type, simname, halo_finder)
	print("Is the file now in cache? %r " % is_in_cache)

closest_cat_in_cache = catman.closest_halocat('cache', catalog_type, simname, halo_finder, desired_redshift)
if closest_cat_in_cache[0] != is_in_cache:
	print("closest_halocat method failed to detect the following catalog: \n%s\n" % is_in_cache)


#############################################################################
####### CHECK THAT WE CAN PROCESS ANY RAW HALO CATALOG INTO AN ARRAY #######
print("\n\n Processing raw halo catalog found in cache\n\n")
halocat_fname = is_in_cache
arr, reader = catman.process_raw_halocat(halocat_fname, simname, halo_finder)

print("\n\n Processing raw halo catalog found on disk\n\n")
halocat_fname = is_on_disk
arr, reader = catman.process_raw_halocat(halocat_fname, simname, halo_finder)


#############################################################################

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
