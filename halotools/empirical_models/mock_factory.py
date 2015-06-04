# -*- coding: utf-8 -*-
"""

Module used to construct mock galaxy populations. 
Each mock factory only has knowledge of a simulation snapshot 
and composite model object. 
Currently only composite HOD models are supported. 

"""

import numpy as np

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from astropy.table import Table 

from . import occupation_helpers as occuhelp
from . import model_defaults

from ..sim_manager import sim_defaults

__all__ = ["HodMockFactory"]
__author__ = ['Andrew Hearin']

@six.add_metaclass(ABCMeta)
class MockFactory(object):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies.

    """
    def __init__(self, snapshot, composite_model, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object 
            Object containing the halo catalog and its metadata, 
            produced by `~halotools.sim_manager.read_nbody.ProcessedSnapshot`

        composite_model : object 
            Any HOD-style model built by `~halotools.empirical_models.HodModelFactory`. 
            Whatever the features of the model, the ``composite_model`` object 
            created by the HOD model factory contains all the instructions 
            needed to produce a Monte Carlo realization of the model with `HodMockFactory`. 

        additional_haloprops : list of strings, optional 
            Each entry in this list must be a column key of the halo catalog. 
            For each entry, mock galaxies will have an attribute storing this 
            property of their host halo. The corresponding mock galaxy attribute name 
            will be pre-pended by ``halo_``. 

        halocut_funcobj : function object, optional 
            Function object used to place a cut on the input subhalo catalog. 
            Input must be a length-Nsubhalos structured numpy array or astropy table; 
            output must be a length-Nsubhalos boolean array that will be used as a mask. 

        Notes 
        -----
        Docs for the test suite for mocks made from 
        any pre-loaded HOD-style models can be seen at 
        `~halotools.empirical_models.test_empirical_models.test_preloaded_hod_mocks`. 

        """

        # Bind the inputs to the mock object
        self.snapshot = snapshot
        self.halos = snapshot.halos
        self.particles = snapshot.particles
        self.model = composite_model
        self.gal_types = self.model.gal_types

        self.additional_haloprops = model_defaults.haloprop_list
        self.additional_haloprops.extend(self.model.haloprop_list)
        if 'additional_haloprops' in kwargs.keys():
            if kwargs['additional_haloprops'] == 'all':
                self.additional_haloprops.extend(self.halos.keys())
            else:
                self.additional_haloprops.extend(kwargs['additional_haloprops'])
        self.additional_haloprops = list(set(self.additional_haloprops))

        if 'halocut_funcobj' in kwargs.keys():
            self.halocut_funcobj = kwargs['halocut_funcobj']

        self.galaxy_table = Table() 

    @abstractmethod
    def populate(self, **kwargs):
        """ Method populating halos with mock galaxies. 

        The `populate` method of `MockFactory` 
        has no implementation, it is simply a placeholder used for standardization. 
        """
        raise NotImplementedError("All subclasses of MockFactory"
        " must include a populate method")


class HodMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies. 

    Can be thought of as a factory that takes a model  
    and simulation snapshot as input, 
    and generates a mock galaxy population. 
    The returned collection of galaxies possesses whatever 
    attributes were requested by the model, such as 3-D position,  
    central/satellite designation, star-formation rate, etc. 

    """

    def __init__(self, snapshot, composite_model, populate=True, **kwargs):

        super(HodMockFactory, self).__init__(snapshot, composite_model, **kwargs)

        occupation_bound = np.array([self.model.occupation_bound[gal_type] 
            for gal_type in self.model.gal_types])
        self._occupation_bounds = occupation_bound

        self.process_halo_catalog()

        self.galaxy_table = Table()

        if populate is True:
            self.populate()        


    def process_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This processing includes identifying the 
        catalog columns that will be used by the model to create the mock, 
        building lookup tables associated with the halo profile, 
        and possibly creating new halo properties. 
        """

        #Make cuts on halo catalog
        # select host halos only, since this is an HOD-style model
        host_halo_cut = (self.halos['upid']==-1)
        self.halos = self.halos[host_halo_cut]
        # make mvir completeness cut
        cutoff_mvir = sim_defaults.Num_ptcl_requirement*self.snapshot.particle_mass
        mass_cut = (self.halos['mvir'] > cutoff_mvir)
        self.halos = self.halos[mass_cut]

        # Make any additional cut requested by the composite model
        if hasattr(self.model, 'halocut_funcobj'):
            self.halos = self.model.halocut_funcobj(halos=self.halos)

        ### Create new columns of the halo catalog, if applicable
        for new_haloprop_key, new_haloprop_func in self.model.new_haloprop_func_dict.iteritems():
            if new_haloprop_key in self.halos.keys():
                raise KeyError("There already exists a halo property with the name %s\n" 
                    "However, the composite model attempted to create a new halo property "
                    "with that name by using the new_haloprop_func_dict" % new_haloprop_key)
            self.halos[new_haloprop_key] = new_haloprop_func(halos=self.halos)
            self.additional_haloprops.append(new_haloprop_key)

        # Create new columns for the halo catalog associated with each 
        # parameter of each halo profile model, e.g., 'NFWmodel_conc'. 
        # New column names are the keys of the halo_prof_func_dict dictionary; 
        # new column values are computed by the function objects in halo_prof_func_dict 
        function_dict = self.model.halo_prof_func_dict
        for new_haloprop_key, prof_param_func in function_dict.iteritems():
            self.halos[new_haloprop_key] = prof_param_func(halos=self.halos)
            self.additional_haloprops.append(new_haloprop_key)

        self.build_halo_prof_lookup_tables()

    def build_halo_prof_lookup_tables(self, **kwargs):
        """ Method calling the `~halotools.empirical_models.HaloProfileModel` 
        component models to build lookup tables of halo profiles. 

        Each ``gal_type`` galaxy has its own associated 
        `~halotools.empirical_models.HaloProfileModel` governing the profile if 
        its underlying dark matter halo. The `build_halo_prof_lookup_tables` 
        method calls each of those component models one by one, requesting 
        each of them to build their own lookup table. Care is taken to ensure 
        that each lookup table spans the necessary range of parameters required 
        by the halo catalog being populated. 

        Parameters 
        ----------
        input_prof_param_table_dict : dict, optional 
            Each dict key of ``input_prof_param_table_dict`` should be 
            a profile parameter name, e.g., ``NFWmodel_conc``. 
            Each dict value is a 3-element tuple; 
            the tuple entries provide, respectively, the min, max, and linear 
            spacing used to discretize the profile parameter. 
            This discretization is used by the 
            `~halotools.empirical_models.HaloProfModel.build_inv_cumu_lookup_table` 
            method of the  `~halotools.empirical_models.HaloProfModel` class 
            to create a lookup table associated with the profile parameter.
            If no ``input_prof_param_table_dict`` is passed, the component 
            models will determine how their parameters are discretized. 
        """
        if 'input_prof_param_table_dict' in kwargs.keys():
            input_prof_param_table_dict = kwargs['input_prof_param_table_dict']
        else:
            input_prof_param_table_dict = {}

        prof_param_table_dict={}

        for gal_type in self.gal_types:
            gal_prof_model = self.model.model_blueprint[gal_type]['profile']

            for key in gal_prof_model.halo_prof_func_dict.keys():

                model_parmin = gal_prof_model.prof_param_table_dict[key][0]
                model_parmax = gal_prof_model.prof_param_table_dict[key][1]
                dpar = gal_prof_model.prof_param_table_dict[key][2]

                halocat_parmin = self.halos[key].min() - dpar
                halocat_parmax = self.halos[key].max() + dpar

                parmin = np.min([halocat_parmin,model_parmin])
                parmax = np.max([halocat_parmax,model_parmax])

                prof_param_table_dict[key] = (parmin, parmax, dpar)

        # Now over-write prof_param_table_dict with 
        # input_prof_param_table_dict, if applicable
        for key, value in input_prof_param_table_dict.iteritems():
            prof_param_table_dict[key] = value

        # Parameter discretization choices have been made. Now build the tables. 
        self.model.build_halo_prof_lookup_tables(
            prof_param_table_dict=prof_param_table_dict)

    def populate(self, **kwargs):
        """ Method populating halos with mock galaxies. 
        """
        self.allocate_memory()

        # Loop over all gal_types in the model 
        for gal_type in self.gal_types:

            # Retrieve the indices of our pre-allocated arrays 
            # that store the info pertaining to gal_type galaxies
            gal_type_slice = self._gal_type_indices[gal_type]
            # gal_type_slice is a slice object

            # For the gal_type_slice indices of 
            # the pre-allocated array self.gal_type, 
            # set each string-type entry equal to the gal_type string
            self.galaxy_table['gal_type'][gal_type_slice] = (
                np.repeat(gal_type, self._total_abundance[gal_type],axis=0))

            # Store all other relevant host halo properties into their 
            # appropriate pre-allocated array 
            for halocatkey in self.additional_haloprops:
                galpropkey = model_defaults.host_haloprop_prefix+halocatkey
                self.galaxy_table[galpropkey][gal_type_slice] = np.repeat(
                    self.halos[halocatkey], self._occupation[gal_type], axis=0)

            # Call the galaxy profile components
            for gal_prof_param_key in self.model.gal_prof_param_list:
                self.galaxy_table[gal_prof_param_key][gal_type_slice] = (
                    getattr(self.model, gal_prof_param_key)(
                        gal_type=gal_type, 
                        galaxy_table=self.galaxy_table[gal_type_slice])
                    )

            # Assign positions 
            # This function is called differently than other galaxy properties, 
            # since 'x', 'y', and 'z' is an attribute of any galaxy-halo model
            # and any gal_type, without exception
            pos_method_name = 'pos_'+gal_type

            self.galaxy_table['x'][gal_type_slice], \
            self.galaxy_table['y'][gal_type_slice], \
            self.galaxy_table['z'][gal_type_slice] = (
                getattr(self.model, pos_method_name)(
                    self, gal_type = gal_type)
                )
                
        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions for all populations at once
        self.galaxy_table['x'] = occuhelp.enforce_periodicity_of_box(
            self.galaxy_table['x'], self.snapshot.Lbox)
        self.galaxy_table['y'] = occuhelp.enforce_periodicity_of_box(
            self.galaxy_table['y'], self.snapshot.Lbox)
        self.galaxy_table['z'] = occuhelp.enforce_periodicity_of_box(
            self.galaxy_table['z'], self.snapshot.Lbox)

    def allocate_memory(self):
        """ Method allocates the memory for all the numpy arrays 
        that will store the information about the mock. 
        These arrays are bound directly to the mock object. 

        The main bookkeeping devices generated by this method are 
        ``_occupation`` and ``_gal_type_indices``. 

        """
        self._occupation = {}
        self._total_abundance = {}
        self._gal_type_indices = {}

        first_galaxy_index = 0
        for gal_type in self.gal_types:
            #print("Working on gal_type %s" % gal_type)
            #
            occupation_func_name = 'mc_occupation_'+gal_type
            occupation_func = getattr(self.model, occupation_func_name)
            # Call the component model to get a MC 
            # realization of the abundance of gal_type galaxies
            self._occupation[gal_type] = occupation_func(halos=self.halos)

            # Now use the above result to set up the indexing scheme
            self._total_abundance[gal_type] = (
                self._occupation[gal_type].sum()
                )
            last_galaxy_index = first_galaxy_index + self._total_abundance[gal_type]
            # Build a bookkeeping device to keep track of 
            # which array elements pertain to which gal_type. 
            self._gal_type_indices[gal_type] = slice(
                first_galaxy_index, last_galaxy_index)
            first_galaxy_index = last_galaxy_index

        self.Ngals = np.sum(self._total_abundance.values())

        # Allocate memory for all additional halo properties, 
        # including profile parameters of the halos such as 'halo_NFWmodel_conc'
        for halocatkey in self.additional_haloprops:
            galpropkey = model_defaults.host_haloprop_prefix+halocatkey
            self.galaxy_table[galpropkey] = np.zeros(self.Ngals, 
                dtype = self.halos[halocatkey].dtype)

        # Separately allocate memory for the values of the (possibly biased)
        # galaxy profile parameters such as 'gal_NFWmodel_conc'
        for galcatkey in self.model.gal_prof_param_list:
            self.galaxy_table[galcatkey] = np.zeros(self.Ngals, dtype = 'f4')

        self.galaxy_table['gal_type'] = np.zeros(self.Ngals, dtype=object)

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for key in phase_space_keys:
            self.galaxy_table[key] = np.zeros(self.Ngals, dtype = 'f4')


class SubhaloMockFactory(object):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies.

    """

    def __init__(self, snapshot, composite_model, populate=True, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object 
            Object containing the halo catalog and its metadata, 
            produced by `~halotools.sim_manager.read_nbody.ProcessedSnapshot`

        composite_model : object 
            Any HOD-style model built by `~halotools.empirical_models.HodModelFactory`. 
            Whatever the features of the model, the ``composite_model`` object 
            created by the HOD model factory contains all the instructions 
            needed to produce a Monte Carlo realization of the model with `HodMockFactory`. 

        additional_haloprops : list of strings, optional 
            Each entry in this list must be a column key of the halo catalog. 
            For each entry, mock galaxies will have an attribute storing this 
            property of their host halo. The corresponding mock galaxy attribute name 
            will be pre-pended by ``halo_``. 

        subhalo_cut_funcobj : function object, optional 
            Function object used to place a cut on the input subhalo catalog. 
            Input must be a length-Nsubhalos structured numpy array or astropy table; 
            output must be a length-Nsubhalos boolean array that will be used as a mask. 

        Notes 
        -----
        Docs for the test suite for mocks made from 
        any pre-loaded HOD-style models can be seen at 
        `~halotools.empirical_models.test_empirical_models.test_preloaded_hod_mocks`. 

        """

        super(SubhaloMockFactory, self).__init__(snapshot, composite_model, **kwargs)

        if 'halocut_funcobj' in kwargs.keys():
            self.halocut_funcobj = kwargs['halocut_funcobj']

        self.galaxy_table = Table() 

        # Pre-compute any additional halo properties required by the model
        self.process_halo_catalog()
        self.precompute_galprops()

        if populate is True:
            self.populate()

    def process_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. 
        """

        #Make cuts on halo catalog
        if hasattr(self, 'subhalo_cut_funcobj'):
            mask = self.subhalo_cut_funcobj(self.halos)
            self.halos = self.halos[mask]

        ### Create new columns of the halo catalog, if applicable
        for new_haloprop_key, new_haloprop_func in self.new_haloprop_func_dict.iteritems():
            self.halos[new_haloprop_key] = new_haloprop_func(self.halos)
            self.additional_haloprops.append(new_haloprop_key)


    def precompute_galprops(self):

        self.galaxy_table = Table()
        for key in self.additional_haloprops:
            newkey = model_defaults.host_haloprop_prefix + key
            self.galaxy_table[newkey] = self.halos[key]

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for newkey in phase_space_keys:
            oldkey = model_defaults.host_haloprop_prefix + newkey
            self.galaxy_table[newkey] = self.galaxy_table[oldkey]


    def populate(self, **kwargs):
        """ Method populating halos with mock galaxies. 

        Workhorse method of `SubhaloMockFactory`. First, 
        `allocate_memory` is called to bind empty arrays to ``self``, 
        into which mock galaxy properties will be stored. 
        For every ``gal_type``, each of its component models are called 
        to assign properties to the galaxies; assignment proceeds by 
        filling the empty arrays created by `allocate_memory`. 
        Optionally, the resulting collection of arrays 
        can be bundled into an Astropy Table, for convenience; 
        for MCMC applications, this bundling may impact performance, 
        and is not recommended. 

        Parameters 
        ----------
        create_astropy_table : boolean, optional 
            If True, the `bundle_into_table` method will be called 
            at the end of executing `populate`. If False, 
            `bundle_into_table` method will not be called. 
            If ``create_astropy_table`` is not passed at all, 
            the behavior will be determined by ``self.create_astropy_table``, 
            set during instantation by the class constructor. 
        """
        for galprop_key in self.model.galprop_keys:
            
            model_func_name = galprop_key + 'model_func'
            model_func = getattr(self.model, model_func_name)
            self.galaxy_table[galprop_key] = model_func(mock_galaxies=self.galaxy_table)



