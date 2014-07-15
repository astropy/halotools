"""
Very simple set of sanity checks on mock.py. 
Not even close to a proper test suite yet. 

"""

import halo_occupation as ho
import numpy as np
import mock
import read_nbody
import timeit


m = mock.HOD_mock()
print("")
print("Mock with all defaults successfully created")
print("Satellite fraction = "+str(m.satellite_fraction))
print('')

if any(m.galaxies['icen'][0:m.ncens] != 1):
	print("Incorrect bookkeeping on central/satellite counts")
	print("Some galaxy in [0:m.ncens] is not a central")

if any(m.galaxies['icen'][m.ncens:-1] != 0):
	print("Incorrect bookkeeping on central/satellite counts")
	print("Some galaxy in [m.ncens:-1] is not a satellite")


temp_halos = read_nbody.load_bolshoi_host_halos_fits()

x=[0,1,-1,2]
y=[10,15,11,26]
coeff = ho.solve_for_quenching_polynomial_coefficients(x,y)
test_coeff = coeff - np.array([10,2,3,0])
if any(test_coeff) != 0:
	print("Bad determination of quenching coefficients!")

if any(m.galaxies['rhalo'][m.galaxies['icen']==1] != 0.0):
	print("Bad assignment of halo-centric position, some centrals are non-central")

if any(m.galaxies['rhalo'][m.galaxies['icen']==0] == 0.0):
	print("Bad assignment of halo-centric position, some satellites lie on top of their central")


t=timeit.Timer("m=mock.HOD_mock()","import mock")
timeit_results =  t.repeat(3,1)
average_runtime_of_mock_creation = np.mean(timeit_results)
print("Average number of seconds to create mock:")
print(average_runtime_of_mock_creation)
