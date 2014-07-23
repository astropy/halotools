"""
This module expresses the default values for the halo occupation models.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

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


default_luminosity_threshold = -19.5
default_tiny_poisson_fluctuation = 1.e-20

default_Npts_concentration_array = 1000
default_Npts_radius_array = 101


class hod_model(object):
    ''' Container class for model parameters determining the HOD. Not currently in use. Implement later.
    
    
    '''
    
    def __init__(self,model_nickname=None,threshold=None):

        if (model_nickname == None) or (model_nickname == 'zheng_etal07'):
            self.model_nickname = 'zheng_etal07'
            self.publication = 'arXiv:0703457'
            
            #Load some tabulated data from Zheng et al. 2007, Table 1
            logMmin_cen_array = [11.35,11.46,11.6,11.75,12.02,12.3,12.79,13.38,14.22]
            sigma_logM_array = [0.25,0.24,0.26,0.28,0.26,0.21,0.39,0.51,0.77]
            logM0_array = [11.2,10.59,11.49,11.69,11.38,11.84,11.92,13.94,14.0]
            logM1_array = [12.4,12.68,12.83,13.01,13.31,13.58,13.94,13.91,14.69]
            alpha_sat_array = [0.83,0.97,1.02,1.06,1.06,1.12,1.15,1.04,0.87]
            threshold_array = np.arange(-22,-17.5,0.5)
            threshold_array = threshold_array[::-1]

            if threshold == None:
                threshold = default_luminosity_threshold

            threshold_index = np.where(threshold_index==threshold)[0]
            if len(threshold_index)==1:
                self.hod_dict = {
                'logMmin_cen' : logMmin_cen_array[threshold_index[0]],
                'sigma_logM' : sigma_logM_array[threshold_index[0]],
                'logM0' : logM0_array[threshold_index[0]],
                'logM1' : logM1_array[threshold_index[0]],
                'alpha_sat' : alpha_sat_array[threshold_index[0]],
                'fconc' : 1.0 # multiplicative factor used to scale satellite concentrations (not actually a parameter in Zheng+07)
                }
            else:
                threshold_index = [3] #choose Mr19.5 as the fallback. 
                #But the above should be implemented differently, so that an exception is raised.

                self.hod_dict = {
                'logMmin_cen' : logMmin_cen_array[threshold_index[0]],
                'sigma_logM' : sigma_logM_array[threshold_index[0]],
                'logM0' : logM0_array[threshold_index[0]],
                'logM1' : logM1_array[threshold_index[0]],
                'alpha_sat' : alpha_sat_array[threshold_index[0]],
                'fconc' : 1.0 # multiplicative factor used to scale satellite concentrations (not actually a parameter in Zheng+07)
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











