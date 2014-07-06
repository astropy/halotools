import halo_occupation as ho
import numpy as np
import mock
import read_nbody


m = mock.HOD_mock()
print("")
print("Mock with all defaults successfully created")
print("Satellite fraction = "+str(m.galaxies['satellite_fraction']))
print('')

temp_halos = read_nbody.load_bolshoi_host_halos_fits()

x=[0,1,-1,2]
y=[10,15,11,26]
coeff = ho.solve_for_quenching_polynomial_coefficients(x,y)
#coeff should be [10,2,3,0]