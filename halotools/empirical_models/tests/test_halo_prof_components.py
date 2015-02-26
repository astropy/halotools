#!/usr/bin/env python
from .. import halo_prof_components as hpc

def test_nfw_instance():
	nfw = hpc.NFWProfile()
	assert nfw._conc_parname == 'NFWmodel_conc'