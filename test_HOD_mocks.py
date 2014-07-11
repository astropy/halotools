import halo_occupation as ho
import numpy as np
import mock
import read_nbody


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

