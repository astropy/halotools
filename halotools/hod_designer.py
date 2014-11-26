# -*- coding: utf-8 -*-
"""

The sole purpose of this module is to 
provide an interface between `halotools.hod_components`
and `halotools.hod_factory`. 
`halotools.hod_designer` creates a set of instructions 
that is passed to `halotools.hod_factory`, 
in the form of a dictionary. The dictionary provides 
a blueprint telling the factory 
how to build an HOD model from the HOD components. 

"""

import numpy as np


class HOD_Model_Designer(object):
	""" The constructor of this class takes 
	a set of HOD model components as input, 
	and composes their behavior in the appropriate fashion 
	for the HOD factory to understand. 
	In particular, the prime function of this class 
	is to bundle a set of input component models into a 
	component_model_dict, which is a dictionary 
	containing a set of instructions to pass to the HOD factory. 
	"""



	def __init__(self, *args, **kwargs):


		
		pass





