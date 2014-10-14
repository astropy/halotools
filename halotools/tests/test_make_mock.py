#!/usr/bin/env python

"""
Very simple set of sanity checks on make_mocks module. Highly incomplete.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
import numpy as np
from ..halo_occupation import Zheng07_HOD_Model
from ..make_mocks import HOD_mock
from ..read_nbody import Catalog_Manager
from ..read_nbody import processed_snapshot
import os

"""
def test_Zheng07_mock():
    #relative_filepath = '../CATALOGS/'
    #catalog_dirname = os.path.join(os.path.dirname(__file__),relative_filepath)
    #hard_coded_catalog_dirname='/Users/aphearin/work/repositories/python/halotools/halotools/CATALOGS/'
    #bolshoi = simulation(manual_dirname=hard_coded_catalog_dirname)

    #particle_data = particles(manual_dirname=hard_coded_catalog_dirname)
    mock = HOD_mock()

    #mock = HOD_mock(simulation_data=bolshoi,halo_occupation_model=Zheng07_HOD_Model,
    #    simulation_particle_data = particle_data)
    mock.populate()

    reasonable_ngal_boolean = (mock.num_total_gals > 5.e4) and (mock.num_total_gals < 1.e5)
    assert reasonable_ngal_boolean == True

    satellite_fraction = mock.num_total_sats/float(mock.num_total_gals)
    reasonable_satellite_fraction_boolean = (satellite_fraction > 0.1) and (satellite_fraction < 0.3)
    assert reasonable_satellite_fraction_boolean == True
"""



"""

def time_mock():
	timer_string = "m.populate()"
#	timer_string = "m=make_mocks.HOD_mock(bolshoi_simulation,zheng07_model)"
#	timer_string = "m=make_mocks.HOD_mock(bolshoi_simulation); m(); nhalf = int(m.num_total_gals/2.); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[0:nhalf],m.coords[0:nhalf].copy()); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[0:nhalf],m.coords[nhalf:-1].copy()); counter = pairs.mr_wpairs.radial_wpairs(None,m.coords[nhalf:-1],m.coords[nhalf:-1].copy())"
	#timer_string = "m=make_mocks.HOD_mock(bolshoi_simulation); m(); nhalf = int(m.num_total_gals/2.); redcounter = pairs.mr_wpairs.radial_wpairs(None,m[0:nhalf].coords,m[0:nhalf].coords.copy()); bluecounter = pairs.mr_wpairs.radial_wpairs(None,m[nhalf:-1].coords,m[nhalf:-1].coords.copy())"
	setup_string = "import make_mocks; import halo_occupation as ho; import read_nbody; import copy; import pairs.mr_wpairs; bolshoi_simulation = read_nbody.load_bolshoi_host_halos_fits(); hod_model = ho.Zheng07_HOD_Model(threshold=-20.5); m=make_mocks.HOD_mock(bolshoi_simulation,hod_model)"
	t = timeit.Timer(timer_string,setup=setup_string)
	timeit_results =  t.repeat(5,1)
	average_runtime_of_mock_creation = np.mean(timeit_results)
	print("Average number of seconds to create mock:")
	print(average_runtime_of_mock_creation)
	print("")
"""

