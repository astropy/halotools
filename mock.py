

import read_nbody
import halo_occupation


class HOD_mock(object):
	'''Base class for any HOD-based mock object'''

	def __init__(self):
		halos = read_nbody.load_bolshoi_host_halos_fits()
		self.halos = halos






