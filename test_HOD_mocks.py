import halo_occupation as ho
import numpy as np
import mock
import read_nbody


m = mock.HOD_mock()
print("")
print("Mock successfully created")
print("Satellite fraction = "+str(m.galaxies['satellite_fraction']))
print('')

temp_halos = read_nbody.load_bolshoi_host_halos_fits()


