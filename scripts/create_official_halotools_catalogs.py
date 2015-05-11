#!/usr/bin/env python

import numpy as np
import os
from halotools import sim_manager

catman = sim_manager.CatalogManager()


########################################################################
### Create Bolshoi Rockstar z=2 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 2
simname = 'bolshoi'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoi_z2_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################
### Create Bolshoi Rockstar z=1 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 1
simname = 'bolshoi'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoi_z1_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################
### Create Bolshoi Rockstar z=0.5 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 0.5
simname = 'bolshoi'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoi_z05_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################
### Create Bolshoi Rockstar z=0 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 0
simname = 'bolshoi'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoi_z0_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################
### Create Bolshoi BDM z=0 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 0
simname = 'bolshoi'
halo_finder = 'bdm'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoi_z0_bdm = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################
### Create Bolshoi-Pl Rockstar z=0 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 0
simname = 'bolshoipl'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_bolshoipl_z0_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)

########################################################################
### Create MultiDark Rockstar z=0 reduced binary
catalog_type = 'raw_halos'
desired_redshift = 0
simname = 'multidark'
halo_finder = 'rockstar'

disk_location = os.path.join(os.path.join('/Volumes/NbodyDisk1/raw_halos', simname), halo_finder)
closest_fname, z = catman.closest_halocat(disk_location, catalog_type, simname, halo_finder, desired_redshift)
arr, reader = (
	catman.process_raw_halocat(closest_fname, simname, halo_finder)
	)
### Store the catalog
version_name = 'halotools.official.version'
notes = {
	'cuts_description': 
	'The only cut on the original catalog made by ' + 
	' the default_halocat_cut method of RockstarReader' + 
	' is to throw out all (sub)halos with Mpeak < 300 particles'
	}
output_fname_multidark_z0_rockstar = catman.store_processed_halocat(
	arr, reader, version_name, notes = notes, overwrite=True)
########################################################################







