

import read_nbody
import halo_occupation as ho
import numpy as np

class HOD_mock(object):
	'''Base class for any HOD-based mock object'''

	def __init__(self):
		temp_halos = read_nbody.load_bolshoi_host_halos_fits()
		self.halos = {}
		self.halos['logM'] = np.array(np.log10(temp_halos.MVIR))
		self.halos['conc'] = np.array(temp_halos.RVIR/temp_halos.RS)







