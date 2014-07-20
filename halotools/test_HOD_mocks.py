"""
Very simple set of sanity checks on mock.py. 
Not even close to a proper test suite yet.
Will be re-written entirely to accommodate the astropy testing suite structure. 

"""

import halo_occupation as ho
import numpy as np
import mock
import read_nbody
import timeit
from copy import copy


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
	print("Bad assignment of Rvir-scaled halo-centric distance, some centrals have rhalo != 0")

if any(m.galaxies['rhalo'][m.galaxies['icen']==0] == 0.0):
	print("Bad assignment of Rvir-scaled halo-centric distance, some satellites have rhalo=0")

xdiff = m.galaxies['pos'][:,0] - m.galaxies['hostpos'][:,0]
xtest = (xdiff != 0)
if any(m.galaxies['icen'][xtest] == 1.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[0] != hostpos[0] have icen=1")
	bad_xcens = m.galaxies[(xdiff != 0) & (m.galaxies['icen']==1)]
xtest2 = (xdiff == 0)
if any(m.galaxies['icen'][xtest2] == 0.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[0] = hostpos[0] have icen=0")
	bad_xsats = m.galaxies[(xdiff == 0) & (m.galaxies['icen']==0)]


ydiff = m.galaxies['pos'][:,1] - m.galaxies['hostpos'][:,1]
ytest = (ydiff != 0)
if any(m.galaxies['icen'][ytest] == 1.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[1] != hostpos[1] have icen=1")
	bad_ycens = m.galaxies[(ydiff != 0) & (m.galaxies['icen']==1)]
ytest2 = (ydiff == 0)
if any(m.galaxies['icen'][ytest2] == 0.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[1] = hostpos[1] have icen=0")
	bad_ysats = m.galaxies[(ydiff == 0) & (m.galaxies['icen']==0)]



zdiff = m.galaxies['pos'][:,2] - m.galaxies['hostpos'][:,2]
ztest = (zdiff != 0)
if any(m.galaxies['icen'][ztest] == 1.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[2] != hostpos[2] have icen=1")
	bad_zcens = m.galaxies[(zdiff != 0) & (m.galaxies['icen']==1)]
ztest2 = (zdiff == 0)
if any(m.galaxies['icen'][ztest2] == 0.0):
	#print("Bad assignment of galaxy position: some galaxies with pos[2] = hostpos[2] have icen=0")
	bad_zsats = m.galaxies[(zdiff == 0) & (m.galaxies['icen']==0)]




'''
t=timeit.Timer("m=mock.HOD_mock()","import mock")
timeit_results =  t.repeat(3,1)
average_runtime_of_mock_creation = np.mean(timeit_results)
print("Average number of seconds to create mock:")
print(average_runtime_of_mock_creation)
print("")
'''


sats = m.galaxies[m.galaxies['icen']==0]
cens = m.galaxies[m.galaxies['icen']==1]



random_satellite = sats[int(np.floor(np.random.random()*m.nsats))]
true_host_centric_distance = np.linalg.norm(random_satellite['pos'] - random_satellite['hostpos'])
catalog_host_centric_distance = random_satellite['rhalo']*random_satellite['rvir']
#print(true_host_centric_distance-catalog_host_centric_distance)



#bad_zsats[0]['pos'],bad_zsats[0]['hostpos']





		#self.galaxies['pos'][self.galaxies['icen']==0][:,0] = self.galaxies['hostpos'][self.galaxies['icen']==0][:,0] +  (_generate_random_points_on_unit_sphere(self.nsats)[:,0]*self.galaxies['rhalo'][self.galaxies['icen']==0]*self.galaxies['rvir'][self.galaxies['icen']==0])

'''
print('')
print(sats['pos'][:,0].min())
print(sats['pos'][:,0].max())
print('')
print('')
print(sats['pos'][:,1].min())
print(sats['pos'][:,1].max())
print('')
'''

#sats['rvir']
#mock._generate_random_points_on_unit_sphere(m.nsats)[:,0]

#		self.galaxies['pos'][self.galaxies['icen']==0][:,0] = self.galaxies['hostpos'][self.galaxies['icen']==0][:,0] +  _generate_random_points_on_unit_sphere(self.nsats)[:,0]*self.galaxies['rhalo'][self.galaxies['icen']==0]*self.galaxies['rvir'][self.galaxies['icen']==0]

#sats['pos'][:,0] = sats['hostpos'][:,0] + (mock._generate_random_points_on_unit_sphere(m.nsats)[:,0]*sats['rhalo']*sats['rvir'])



#m.galaxies['pos'][m.galaxies['icen']==0][:,0] = m.galaxies['hostpos'][m.galaxies['icen']==0][:,0] +  (mock._generate_random_points_on_unit_sphere(m.nsats)[:,0]*m.galaxies['rhalo'][m.galaxies['icen']==0]*m.galaxies['rvir'][m.galaxies['icen']==0])
#m.galaxies['pos'][m.galaxies['icen']==0][0,0] = copy(mock._generate_random_points_on_unit_sphere(m.nsats)[0,0])


#rpts = mock._generate_random_points_on_unit_sphere(m.nsats)
#m.galaxies['pos'][m.galaxies['icen']==0][:,0] = copy(rpts[:,0])



