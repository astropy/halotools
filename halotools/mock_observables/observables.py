""" 
Functions that compute statistics of a mock galaxy catalog in a periodic box. 
Still largely unused in its present form, and needs to be integrated with 
the pair counter and subvolume membership methods.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

__all__=['one_dimensional_periodic_distance','three_dimensional_periodic_distance']

import numpy as np


def one_dimensional_periodic_distance(x1,x2,Lbox):
	""" 
	Find the 1d Euclidean distance between x1 & x2, accounting for box periodicity.

	Args
	----
	x1, x2: 1D array
	Lbox : float

	Returns
	-------
	periodic_distance : array

	"""
	periodic_distance = np.abs(x1 - x2)
	test = periodic_distance > 0.5*Lbox
	periodic_distance[test] = np.abs(periodic_distance[test] - Lbox)
	return periodic_distance

def three_dimensional_periodic_distance(pos1,pos2,Lbox):
	""" 
	Find the 3d Euclidean distance between pos1 & pos2, accounting for box periodicity.

	Args
	----
	x1, x2: 3xN array
	Lbox : float

	Returns
	-------
	periodic_distance : array

	"""

	periodic_xdistance = one_dimensional_periodic_distance(pos1[:,0],pos2[:,0],Lbox)
	periodic_ydistance = one_dimensional_periodic_distance(pos1[:,1],pos2[:,1],Lbox)
	periodic_zdistance = one_dimensional_periodic_distance(pos1[:,2],pos2[:,2],Lbox)

	distances = np.sqrt(periodic_xdistance**2 + periodic_ydistance**2 + periodic_zdistance**2)
	
	return distances 








