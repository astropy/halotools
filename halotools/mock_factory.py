# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a processed snapshot 
and composite model object. 
Currently only the HOD factory is supported. 

"""

import numpy as np

class HOD_Mock_Factory(object):
	""" The constructor of this class takes 
	a snapshot and a composite model as input, 
	and returns a Monte Carlo realization of the composite model 
	painted onto the input snapshot. 
	"""



	def __init__(self, snapshot, composite_model, bundle_into_table=True):
		pass































