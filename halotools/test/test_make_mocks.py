#!/usr/bin/env python

"""
Very simple set of sanity checks on mock.py. 
Still figuring out how to structure this properly.
Will copy and paste my additional tests once I figure out the basic design conventions.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from ..make_mocks import HOD_mock
import timeit
from copy import copy


def test_make_HOD_mock():

	m = HOD_mock()

	assert np.all(m.galaxies['icen'][0:m.ncens] == 1)
#	assert np.all(m.galaxies['icen'][m.ncens:-1] == 0)


"""
def time_mock():
	timer_string = "m=make_mocks.HOD_mock(simulation_data = bolshoi_simulation)"
	setup_string = "import make_mocks; import read_nbody; bolshoi_simulation = read_nbody.load_bolshoi_host_halos_fits()"
	t = timeit.Timer(timer_string,setup=setup_string)
	timeit_results =  t.repeat(3,1)
	average_runtime_of_mock_creation = np.mean(timeit_results)
	print("Average number of seconds to create mock:")
	print(average_runtime_of_mock_creation)
	print("")
"""















###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()





