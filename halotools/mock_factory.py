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

        # Set the gal_types attribute, sorted so that bounded populations appear first
        self._occupation_bound = []
        for gal_type in self.model.gal_types:
            self._occupation_bound.extend(self.model.occupation_bound[gal_type])
        self.gal_types = np.array(self.model.gal_types[np.argsort(self._occupation_bound)])
        self._occupation_bound = np.array(self._occupation_bound[np.argsort(self._occupation_bound)])

        self.prim_haloprop_key = composite_model.prim_haloprop_key
        if hasattr(composite_model,'sec_haloprop_key'): 
            self.sec_haloprop_key = composite_model.sec_haloprop_key


    def populate(self):
        # Assign properties to bounded populations first.
        bounded_populations = self.gal_types[self._occupation_bound <= 1]
        unbounded_populations = self.gal_types[self._occupation_bound > 1]

        for gal_type in bounded_populations:
            first_index = self._gal_type_indices[gal_type][0]
            last_index = self._gal_type_indices[gal_type][1]

            # First assign the trivial properties
            self.gal_type[first_index:last_index] = gal_type
            self.haloID[first_index:last_index] = (
                self.halos['ID'][self._occupation[gal_type]==1])
            self.prim_haloprop[first_index:last_index]= (
                self.halos[self.prim_haloprop_key][self._occupation[gal_type]==1])
            if hasattr(self.model,'sec_haloprop_key'):
                self.sec_haloprop[first_index:last_index] = (
                    self.halos[self.sec_haloprop_key][self._occupation[gal_type]==1])

            # Now call the phase space model
            self.coords[first_index:last_index] = (
                self.halos['POS'][self._occupation[gal_type]==1])
            self.coordshost[first_index:last_index] = (
                self.halos['POS'][self._occupation[gal_type]==1])
            self.vel[first_index:last_index] = (
                self.halos['VEL'][self._occupation[gal_type]==1])
            self.velhost[first_index:last_index] = (
                self.halos['VEL'][self._occupation[gal_type]==1])




    def _allocate_memory(self):
        self._occupation = {}
        self._total_abundance = {}
        self._gal_type_indices = {}
        first_galaxy_index = 0
        for gal_type in self.gal_types:
            if hasattr(self.model,'sec_haloprop_key'):
                self._occupation[gal_type] = (
                    self.model.mc_occupation(
                        gal_type, 
                        self.halos[self.prim_haloprop_key], 
                        self.halos[self.sec_haloprop_key])
                    )
            else:
                self._occupation[gal_type] = (
                    self.model.mc_occupation(
                        gal_type, 
                        self.halos[self.prim_haloprop_key])
                    )
            self._total_abundance[gal_type] = (
                self._occupation[gal_type].sum()
                )
            last_galaxy_index = first_galaxy_index + self._total_abundance[gal_type]
            self._gal_type_indices[gal_type] = (
                first_galaxy_index, last_galaxy_index)
            first_galaxy_index = last_galaxy_index

        self.Ngals = np.sum(self._total_abundance.values())

        self.coords = np.empty((self.Ngals,3),dtype='f8')
        self.coordshost = np.empty((self.Ngals,3),dtype='f8')
        self.vel = np.empty((self.Ngals,3),dtype='f8')
        self.velhost= np.empty((self.Ngals,3),dtype='f8')
        self.gal_type = np.empty(self.Ngals,dtype=object)
        self.haloID = np.empty(self.Ngals,dtype='i8')
        self.prim_haloprop = np.empty(self.Ngals,dtype='f8')
        if hasattr(self.model,'sec_haloprop_key'):
            self.sec_haloprop = np.empty(self.Ngals,dtype='f8')


        # Still not sure how the composite model keeps track of  
        # what features have been compiled (definitely not as follows, though)
        # if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        self.quiescent = np.empty(self.Ngals,dtype=object)




































