"""
This module expresses the default values for the halo occupation models.
"""

class simulation(object):
    ''' Container class for properties of the simulation being used.
    
    Still unused.
    
    
    '''
    
    def __init__(self,simulation_nickname=None):
        
        if simulation_nickname is None:
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        elif simulation_nickname is 'Bolshoi':
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        else:
            pass
        
                    

### Default halo catalog
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
default_simulation_dict = {
	'catalog_filename':default_halo_catalog_filename,
	'Lbox':250.0,
	'scale_factor':1.0003,
	'particle_mass':1.35e8,
	'softening':1.0
}

class hod(object):
    ''' Container class for model parameters determining the HOD.
    
    
    '''
    
    def __init__(self):
        self.hod_dict = {
        'logMmin_cen' : 11.7, # log Mass where < Ncen (M) > = 0.5
        'sigma_logM' : 0.15, # scatter in central galaxy stellar-to-halo mass
        'logMmin_sat' : 11.9, # low-end cutoff in log Mass for a halo to contain a satellite
        'Msat_ratio' : 20.0, # multiplicative factor specifying when a halo contains a satellite
        'alpha_sat' : 1.0, # power law slope of the satellite occupation function
        'fconc' : 0.5 # multiplicative factor used to scale satellite concentrations
        }
        
        self.color_dict = {
        'central_coefficients' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
        'satellite_coefficients' : [0.5,0.75,0.85]        
        }
        self.logM_abcissa = [12,13.5,15] #halo masses at which model quenching fractions are defined


### Default HOD model
default_hod_dict = {
	'logMmin_cen' : 11.7, # log Mass where < Ncen (M) > = 0.5
	'sigma_logM' : 0.15, # scatter in central galaxy stellar-to-halo mass
	'logMmin_sat' : 11.9, # low-end cutoff in log Mass for a halo to contain a satellite
	'Msat_ratio' : 20.0, # multiplicative factor specifying when a halo contains a satellite
	'alpha_sat' : 1.0, # power law slope of the satellite occupation function
    'fconc' : 0.5 # multiplicative factor used to scale satellite concentrations
}

default_color_dict = {
     'central_coefficients' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
     'satellite_coefficients' : [0.5,0.75,0.85], #polynomial coefficients determining quenched fraction of centrals
     'abcissa' : [12,13.5,15]
}


default_NFW_concentration_precision = 0.5











