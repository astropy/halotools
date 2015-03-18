#!/usr/bin/env python
from .. import occupation_helpers as occuhelp 
import numpy as np 

def test_enforce_periodicity_of_box():

	box_length = 250
	Npts = 1e5
	coords = np.random.uniform(0,box_length,Npts*3).reshape(Npts,3)

	perturbation_size = box_length/10.
	coord_perturbations = np.random.uniform(
		-perturbation_size,perturbation_size,Npts*3).reshape(Npts,3)

	coords += coord_perturbations

	newcoords = occuhelp.enforce_periodicity_of_box(coords, box_length)
	assert np.all(newcoords >= 0)
	assert np.all(newcoords <= box_length)