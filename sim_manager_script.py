#!/usr/bin/env python

import numpy as np
import os

from halotools import sim_manager
reload(sim_manager)

catman = sim_manager.CatalogManager()

########################################################
####### CHECK THAT RAW HALO CATALOGS GO TO CACHE #######

supported_halocats = catman.available_halocats
simname, halo_finder = supported_halocats[0]
available_cats_for_download = catman.available_snapshots('web', 'raw_halos', simname, halo_finder)
available_cats_in_cache = catman.available_snapshots('cache', 'raw_halos', simname, halo_finder)

location = 'web'
catalog_type = 'raw_halos'
desired_redshift = 1
closest_cat = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
print("\n Closest matching catalog on the web = \n%s\n " % closest_cat[0])
location = 'cache'
closest_cat = catman.closest_halocat(location, catalog_type, simname, halo_finder, desired_redshift)
print("\n Closest matching catalog in cache = \n%s\n " % closest_cat[0])

########################################################


########################################################
####### CHECK THE HALOCAT_OBJ PROPERTIES #######
halocat_obj = sim_manager.read_nbody.get_halocat_obj(simname, halo_finder)
webloc = halocat_obj.raw_halocat_web_location


########################################################

########################################################


########################################################

########################################################


########################################################



