import halo_occupation as ho
import numpy as np
import mock
import read_nbody


m = mock.HOD_mock()
print("")
print("Mock with all defaults successfully created")
print("Satellite fraction = "+str(m.satellite_fraction))
print('')

temp_halos = read_nbody.load_bolshoi_host_halos_fits()

x=[0,1,-1,2]
y=[10,15,11,26]
coeff = ho.solve_for_quenching_polynomial_coefficients(x,y)
#coeff should be [10,2,3,0]

z = np.zeros(3,dtype=[('col1','i8'),('col2','f4'),('col3','3float32')])
z['col1'] = [100000,488888,299999]
z['col2']=[0.4,0.5,0.2]
z['col3']=[(3,4,5),(4,2.,0),(5,7,45)]
