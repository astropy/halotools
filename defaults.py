"""
This module expresses the default values for the halo occupation models.
"""

### Default halo catalog
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
default_simulation_dict = {
	'catalog_filename':default_halo_catalog_filename,
	'Lbox':250.0,
	'scale_factor':1.0003,
	'particle_mass':1.35e8,
	'softening':1.0
}

### Default HOD model
default_hod_dict = {
	'logMmin_cen' : 11.7, # log Mass where < Ncen (M) > = 0.5
	'sigma_logM' : 0.15, # scatter in central galaxy stellar-to-halo mass
	'logMmin_sat' : 11.9, # low-end cutoff in log Mass for a halo to contain a satellite
	'Msat_ratio' : 20.0, # multiplicative factor specifying when a halo contains a satellite
	'alpha_sat' : 1.0 # power law slope of the satellite occupation function
}