"""
Very simple set of sanity checks on mock.py. 
Not even close to a proper test suite yet.
Will be re-written entirely to accommodate the astropy testing suite structure. 

"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import halo_occupation as ho
import numpy as np
import make_mocks
import read_nbody
import timeit
from copy import copy
import observables

def main():

	test_three_dimensional_periodic_distance()
	simulation = test_read_nbody()
	halos = simulation['halos']
	mock = test_make_HOD_mock(simulation)
	mock.populate()
	print(str(mock.num_total_gals)+' galaxies in mock')
	#time_mock()
	#test_satellite_positions(mock)



def test_three_dimensional_periodic_distance():
	""" Use a few known pencil-and-paper answers to check my 3D periodic distance function

	"""

	Lbox = 100.0
	pos1 = np.zeros(3)
	pos2 = pos1 + 1
	pos3 = pos1 + Lbox - 1
	pos4 = pos1 + 1 + (0.5*Lbox)
	pos5 = pos1 + (0.5*Lbox)
	pos1_array = np.tile(pos1,5).reshape(5,3)
	test_points_array = np.append(pos1,[pos2,pos3,pos4,pos5]).reshape(5,3)

	calculated_distances = observables.three_dimensional_periodic_distance(pos1_array,test_points_array,Lbox)

	correct_distance1 = 0.0
	correct_distance2 = np.sqrt(3)
	correct_distance3 = np.sqrt(3)
	correct_distance4 = np.sqrt(3)*(0.5*Lbox - 1)
	correct_distance5 = np.sqrt(3)*(0.5*Lbox)
	correct_distances = [correct_distance1,correct_distance2,correct_distance3,correct_distance4,correct_distance5]

	test_answer =  calculated_distances - correct_distances
	np.testing.assert_allclose(test_answer,np.zeros(len(test_answer)),rtol=1.e-5,atol=1.e-5)


def test_make_HOD_mock(simulation=None):

	if simulation == None:
		simulation = read_nbody.load_bolshoi_host_halos_fits()

	m = make_mocks.HOD_mock(simulation_data = simulation)
	print("")
	print("Mock with all defaults successfully initialized")
	#print("Satellite fraction = "+str(m.satellite_fraction))
	print('')
	m.populate()
	print("Mock with all defaults successfully populated")
	print('')


	return m

"""
	if any(m.galaxies['icen'][0:m.ncens] != 1):
		print("Incorrect bookkeeping on central/satellite counts")
		print("Some galaxy in [0:m.ncens] is not a central")

	if any(m.galaxies['icen'][m.ncens:-1] != 0):
		print("Incorrect bookkeeping on central/satellite counts")
		print("Some galaxy in [m.ncens:-1] is not a satellite")
"""


def test_read_nbody():

	simulation = read_nbody.load_bolshoi_host_halos_fits()

	return simulation


def test_solve_for_quenching_polynomial_coefficients():
	""" 
	Use known pencil-and-paper answer to check 
	that solve_for_quenching_polynomial_coefficients
	is correctly solving the input linear system"""

	x=[0,1,-1,2]
	y=[10,15,11,26]
	coeff = ho.solve_for_quenching_polynomial_coefficients(x,y)
	test_coeff = coeff - np.array([10,2,3,0])
	if any(test_coeff) != 0:
		print("Bad determination of quenching coefficients!")

	if any(m.galaxies['rhalo'][m.galaxies['icen']==1] != 0.0):
		print("Bad assignment of Rvir-scaled halo-centric distance, some centrals have rhalo != 0")

	if any(m.galaxies['rhalo'][m.galaxies['icen']==0] == 0.0):
		print("Bad assignment of Rvir-scaled halo-centric distance, some satellites have rhalo=0")


def time_mock():
	timer_string = "m=make_mocks.HOD_mock(bolshoi_simulation); m(); nhalf = int(m.num_total_gals/2.); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[0:nhalf],m.coords[0:nhalf].copy()); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[0:nhalf],m.coords[nhalf:-1].copy()); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[nhalf:-1],m.coords[nhalf:-1].copy())"
	#timer_string = "m=make_mocks.HOD_mock(bolshoi_simulation); m(); nhalf = int(m.num_total_gals/2.); redcounter = pairs.mr_wpairs.radial_wpairs(None,m[0:nhalf].coords,m[0:nhalf].coords.copy()); bluecounter = pairs.mr_wpairs.radial_wpairs(None,m[nhalf:-1].coords,m[nhalf:-1].coords.copy())"
	setup_string = "import make_mocks; import read_nbody; import copy; import pairs.mr_wpairs; bolshoi_simulation = read_nbody.load_bolshoi_host_halos_fits()"
	t = timeit.Timer(timer_string,setup=setup_string)
	timeit_results =  t.repeat(3,1)
	average_runtime_of_mock_creation = np.mean(timeit_results)
	print("Average number of seconds to create mock:")
	print(average_runtime_of_mock_creation)
	print("")



def test_satellite_positions(mock):
	"""
	Verify that rhalo*rvir gives the true halo-centric distance of all satellites.

	"""

	sats = mock.galaxies[mock.galaxies['icen']==0]
	Lbox = mock.simulation_dict['Lbox']
	actual_distances = observables.three_dimensional_periodic_distance(sats['pos'],sats['hostpos'],Lbox)
	rhalo_derived_distances = sats['rhalo']*sats['rvir']
	



#random_satellite = sats[int(np.floor(np.random.random()*m.nsats))]
#true_host_centric_distance = np.linalg.norm(random_satellite['pos'] - random_satellite['hostpos'])
#catalog_host_centric_distance = random_satellite['rhalo']*random_satellite['rvir']
#print(true_host_centric_distance-catalog_host_centric_distance)

















###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()





