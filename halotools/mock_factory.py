# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a processed snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np
import occupation_helpers as occuhelp

class HodMockFactory(object):
    """ The constructor of this class takes 
    a snapshot and a composite model as input, 
    and returns a Monte Carlo realization of the composite model 
    painted onto the input snapshot. 
    """

    def __init__(self, snapshot, composite_model, bundle_into_table=True):

        # Bind the inputs to the mock object
        self.snapshot = snapshot
        self.halos = snapshot.halos
        self.particles = snapshot.particles
        self.model = composite_model

        self._set_gal_types()
        self.process_halo_catalog()


    def process_halo_catalog(self):

        self.prim_haloprop_key = self.model.prim_haloprop_key
        if hasattr(self.model,'sec_haloprop_key'): 
            self.sec_haloprop_key = self.model.sec_haloprop_key

        # At least need profile parameters for every halo
        # May also wish to add data for prim_haloprop and sec_haloprop

        halo_prof_param_keys = []
        prim_haloprop = self.halos[self.prim_haloprop_key]

        halo_prof_param_model_dict = self.model.halo_prof_param_model.param_func_dict
        for key, prof_param_func in halo_prof_param_model_dict.iteritems():
            new_key = 'prof_model_'+key
            halo_prof_param_keys.extend([new_key])
            self.halos[new_key] = prof_param_func(prim_haloprop)

        setattr(self.halos, 'halo_prof_param_keys', halo_prof_param_keys)

    def _set_gal_types(self):
        """ Internal bookkeeping method used to conveniently bind the gal_types of a 
        composite model, and their occupation bounds, to the mock object. 
        """

        # Set the gal_types attribute, sorted so that bounded populations appear first
        self._occupation_bound = []
        for gal_type in self.model.gal_types:
            self._occupation_bound.extend(self.model.occupation_bound[gal_type])
        self.gal_types = np.array(self.model.gal_types[np.argsort(self._occupation_bound)])
        self._occupation_bound = np.array(self._occupation_bound[np.argsort(self._occupation_bound)])
        if (set(self._occupation_bound) != {1, float("inf")}):
            raise ValueError("The only supported finite occupation bound is unity,"
                " otherwise it must be set to infinity")

    def populate_bounded(self,gal_type):
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
        self.coordshost[first_index:last_index] = (
            self.halos['POS'][self._occupation[gal_type]==1])
        self.velhost[first_index:last_index] = (
            self.halos['VEL'][self._occupation[gal_type]==1])

        # Now call the phase space model
        # Note that this call to mc_coords will eventually need to be modified 
        # to accommodate profile models that depend on two halo properties
        occupations = self._occupation[gal_type][self._occupation[gal_type]>0]
        virial_radii = self.halos['RVIR'][self._occupation[gal_type]==1]

        ### Note that self.model does not yet correctly interface with the following API
        self.coords[first_index:last_index] = (
            self.model.mc_coords(gal_type, 
                self.coords[first_index:last_index], occupations, 
                self.coordshost[first_index:last_index], virial_radii, 
                self.prim_haloprop[first_index:last_index]
                )
            )
        # Velocities are still unmodeled for now
        self.vel[first_index:last_index] = (
            self.halos['VEL'][self._occupation[gal_type]==1])


    def populate_unbounded(self,gal_type):
        first_index = self._gal_type_indices[gal_type][0]
        last_index = self._gal_type_indices[gal_type][1]

        sat_prof_model = self.model.component_model_dict[gal_type]['profile_model']

        host_prof_param_dict = {}
        for prof_param_key in self.halos.halo_prof_param_keys:
            host_prof_param_dict[prof_param_key] = (
                self.halos[prof_param_key][self._occupation[gal_type]>0])

        satsys_prof_param_dict = host_prof_param_dict
        # Profile modulating functions allow satellites to be biased tracers 
        # of the halo profile. Not implemented yet. 
        if hasattr(sat_prof_model, 'prof_param_modfunc'):
            pass 

        inv_cumu_prof_funcs = self.model.halo_prof_model.digitized_inv_cumu_profs(
            satsys_prof_param_dict)

        # Define some convenient shorthands for the properties 
        # that will be assigned to the satellites
        occupations = self._occupation[gal_type][self._occupation[gal_type]>0]
        host_centers = self.halos['POS'][self._occupation[gal_type]>0]
        host_vels = self.halos['VEL'][self._occupation[gal_type]>0]
        host_Rvirs = self.halos['RVIR'][self._occupation[gal_type]>0]/1000.
        host_IDs = self.halos['ID'][self._occupation[gal_type]>0]
        host_prim_haloprops = self.halos[self.prim_haloprop_key][self._occupation[gal_type]>0]
        if hasattr(self.model,'sec_haloprop_key'):
            host_sec_haloprops = (
                self.halos[self.sec_haloprop_key][self._occupation[gal_type]>0])


        satsys_first_index = first_index
        for host_index, Nsatsys in enumerate(occupations):
            satsys_coords = self.coords[satsys_first_index:satsys_first_index+Nsatsys]
            host_center = host_centers[host_index]
            host_vel = host_vels[host_index]
            host_Rvir = host_Rvirs[host_index]

            inv_cumu_prof_func = inv_cumu_prof_funcs[host_index]

            satsys_coords = (self.model.mc_coords(
                satsys_coords, inv_cumu_prof_func, host_center, host_Rvir)
                )
            self.coordshost[satsys_first_index:satsys_first_index+Nsatsys]=host_center
            self.velhost[satsys_first_index:satsys_first_index+Nsatsys]=host_vel
            self.haloID[satsys_first_index:satsys_first_index+Nsatsys]=host_IDs[host_index]
            self.prim_haloprop[satsys_first_index:satsys_first_index+Nsatsys] = (
                host_prim_haloprops[host_index])
            if hasattr(self.model,'sec_haloprop_key'):
                self.sec_haloprop[satsys_first_index:satsys_first_index+Nsatsys] = (
                    host_sec_haloprops[host_index])

            satsys_first_index += Nsatsys


    def populate(self):

        self._allocate_memory()
        # Assign properties to bounded populations
        unity_bounded_populations = self.gal_types[self._occupation_bound == 1]
        for gal_type in unity_bounded_populations:
            self.populate_bounded(gal_type)

        unbounded_populations = self.gal_types[self._occupation_bound == float("inf")]
        for gal_type in unbounded_populations:
            self.populate_unbounded(gal_type)

        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions of the simulation box
        self.coords = occuhelp.enforce_periodicity_of_box(self.coords, self.snapshot.Lbox)


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

        self.coords = np.zeros((self.Ngals,3),dtype='f8')
        self.coordshost = np.zeros((self.Ngals,3),dtype='f8')
        self.vel = np.zeros((self.Ngals,3),dtype='f8')
        self.velhost= np.zeros((self.Ngals,3),dtype='f8')
        self.gal_type = np.zeros(self.Ngals,dtype=object)
        self.haloID = np.zeros(self.Ngals,dtype='i8')
        self.prim_haloprop = np.zeros(self.Ngals,dtype='f8')
        if hasattr(self.model,'sec_haloprop_key'):
            self.sec_haloprop = np.zeros(self.Ngals,dtype='f8')


        # Still not sure how the composite model keeps track of  
        # what features have been compiled (definitely not as follows, though)
        # if 'quenching_abcissa' in self.halo_occupation_model.parameter_dict.keys():
        self.quiescent = np.zeros(self.Ngals,dtype=object)




































