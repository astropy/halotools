"""
Module expressing various default values of the mock-making code.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)



### Default halo catalog
default_halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/presorted_host_halo_catalog.fits'
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


### Default values specifying traditional quenching model
default_quenching_parameters = {
    'quenching_abcissa' : [12,13.5,15],
    'central_quenching_ordinates' : [0.35,0.75,0.95], #polynomial coefficients determining quenched fraction of centrals
    'satellite_quenching_ordinates' : [0.5,0.75,0.85] #polynomial coefficients determining quenched fraction of centrals
    }

default_assembias_parameters = {
	'assembias_abcissa' : [12,14],
	'satellite_assembias_ordinates' : [1.1,1],
	'central_assembias_ordinates' : [1.0,1.1]
	}

default_satcen_parameters = {
	'assembias_abcissa' : [12,13.5,15],
	'satellite_assembias_ordinates' : [1.5,1.25,0.5],
	'central_assembias_ordinates' : [1.0,1.0,1.0]
	}

default_halo_type_split = {
	'halo_type_abcissa' : [13,],
	'halo_type_split' : [0.5,]
	}

def halo_type_function(logM,Mvir_independent_fraction=[0.5,0.5]):
	""" Place-holder method used to assign types to host halos.

	Parameters 
	----------
	logM : array_like
		array of log10(Mvir) of halos in catalog

	Mvir_independent_fraction : array_like
		Value of entry i gives the mass-independent fraction of 
		halos that are assigned ``Type i``.
	"""

	Mvir_independent_fraction = np.array(Mvir_independent_fraction)
	return Mvir_independent_fraction











