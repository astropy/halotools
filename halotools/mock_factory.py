# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a processed snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np

class HodMockFactory(object):
	""" The constructor of this class takes 
	a snapshot and a composite model as input, 
	and returns a Monte Carlo realization of the composite model 
	painted onto the input snapshot. 
	"""



	def __init__(self, snapshot, composite_model, bundle_into_table=True):

		self.snapshot = snapshot
		self.halos = snapshot.halos
		self.particles = snapshot.particles
		self.model = composite_model

		self.prim_haloprop_key = composite_model.prim_haloprop_key
		self.prim_haloprop = self.halos[self.prim_haloprop_key]
		if hasattr(composite_model,'sec_haloprop_key'): 
			self.sec_haloprop_key = composite_model.sec_haloprop_key
			self.sec_haloprop = self.halos[self.sec_haloprop_key]



	def populate(self):

		pass

	def _allocate_memory(self):




































