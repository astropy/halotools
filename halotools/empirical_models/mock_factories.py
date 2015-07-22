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

from . import model_helpers as model_helpers
from . import model_defaults

from ..sim_manager import sim_defaults

__all__ = ['MockFactory', 'HodMockFactory', 'SubhaloMockFactory']
__author__ = ['Andrew Hearin']

@six.add_metaclass(ABCMeta)
class MockFactory(object):
    """ Abstract base class responsible for populating a simulation 
    with a synthetic galaxy population.

    `MockFactory` is an abstract base class, and cannot be instantiated. 
    Concrete sub-classes of `MockFactory` such as `HodMockFactory` and 
    `SubhaloMockFactory` are the objects used 
    to populate simulations with galaxies. 
    """
    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object 
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object 
            A model built by a sub-class of `~halotools.empirical_models.ModelFactory`. 

        additional_haloprops : list of strings, optional keyword argument  
            Each entry in this list must be a column key of ``snapshot.halos``. 
            For each entry of ``additional_haloprops``, each member of the mock galaxy population 
            will have an attribute storing this property of its host halo. 
            The corresponding mock galaxy attribute name will be pre-pended by ``halo_``. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        halocut_funcobj : function object, optional keyword argument  
            Function object used to place a cut on the input ``snapshot.halos`` table. 
            Default behavior depends on the sub-class of `MockFactory`. 
            If the ``halocut_funcobj`` keyword argument is passed, 
            the input to the function must be a length-Nsubhalos structured numpy array or Astropy table; 
            the function output must be a length-Nsubhalos boolean array that will be used as a mask. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Dictionary of function objects used to create additional halo properties 
            by `preprocess_halo_catalog`. Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 
        """

        required_kwargs = ['snapshot', 'model']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        self.halos = self.snapshot.halos
        self.particles = self.snapshot.particles
        if hasattr(self.model, 'gal_types'):
            self.gal_types = self.model.gal_types

        self._build_additional_haloprops_list(**kwargs)
        self._build_new_haloprop_func_dict(**kwargs)

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

    def _build_additional_haloprops_list(self, **kwargs):

        # Create a list of halo properties that will be inherited by the mock galaxies
        self.additional_haloprops = model_defaults.haloprop_list
        if hasattr(self.model, '_haloprop_list'):
            self.additional_haloprops.extend(self.model._haloprop_list)
        if 'additional_haloprops' in kwargs.keys():
            if kwargs['additional_haloprops'] == 'all':
                self.additional_haloprops.extend(self.halos.keys())
            else:
                self.additional_haloprops.extend(kwargs['additional_haloprops'])
        self.additional_haloprops = list(set(self.additional_haloprops))


    def _build_new_haloprop_func_dict(self, **kwargs):
        """ Private method tests self-consistency of the new_haloprop_func_dict 
        dictionaries with the halo catalog. 

        """
        # Test consistency of the keyword argument new_haloprop_func_dict
        if 'new_haloprop_func_dict' in kwargs.keys():
            kwargs_input_haloprop_func_dict = kwargs['new_haloprop_func_dict']

            # Test that kwargs_input_haloprop_func_dict does not have keys that 
            # overlap with the halo catalog.  
            for key in kwargs_input_haloprop_func_dict.keys():
                if key in self.halos.keys():
                    raise KeyError("There already exists a halo property "
                        "with the name %s.\n However, the keyword argument  "
                        "new_haloprop_func_dict contains this key.\n"
                        "If the %s column is already the column you need, "
                        "then you should delete the corresponding entry of new_haloprop_func_dict.\n"
                        "Otherwise, you should rename the key "
                        "that you are using new_haloprop_func_dict to create." % (key, key))
        else:
            kwargs_input_haloprop_func_dict = {}

        # Test consistency of the new_haloprop_func_dict bound to the composite model
        if hasattr(self.model, 'new_haloprop_func_dict'):
            model_haloprop_func_dict = self.model.new_haloprop_func_dict

            # Test that model_haloprop_func_dict does not have keys that 
            # overlap with the halo catalog.  
            for key in model_haloprop_func_dict.keys():
                if key in self.halos.keys():
                    raise KeyError("There already exists a halo property "
                        "with the name %s.\n However, the composite model's "
                        "new_haloprop_func_dict contains this key.\n"
                        "If the %s column is already the column you need, "
                        "then you should delete the corresponding entry of new_haloprop_func_dict.\n"
                        "Otherwise, you should rename the key "
                        "that you are using new_haloprop_func_dict to create." % (key, key))
        else:
            model_haloprop_func_dict = {}

        kwargs_input_haloprop_func_set = set(kwargs_input_haloprop_func_dict)
        model_haloprop_func_set = set(model_haloprop_func_dict)
        intersection = kwargs_input_haloprop_func_set.intersection(model_haloprop_func_set)
        if intersection == set():
            composite_haloprop_func_dict = dict(
                kwargs_input_haloprop_func_dict.items() + 
                model_haloprop_func_dict.items()
                )
        else:
            repeated_key = list(intersection)[0]
            raise KeyError("The dict key %s appears both in "
                " the new_haloprop_func_dict passed to "
                "the mock factory as a keyword argument, "
                "and also appears in the new_haloprop_func_dict"
                "bound to the model. "
                "You must disambiguate either by providing a new key name, "
                "or by deleting this entry from one of the dictionaries. " % repeated_key)

        if composite_haloprop_func_dict != {}:
            self.new_haloprop_func_dict = composite_haloprop_func_dict


class HodMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies based on an HOD-style model. 

    Can be thought of as a factory that takes a model  
    and simulation snapshot as input, 
    and generates a mock galaxy population. 
    The returned collection of galaxies possesses whatever 
    attributes were requested by the model, such as xyz position,  
    central/satellite designation, star-formation rate, etc. 

    """

    def __init__(self, populate=True, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object, keyword argument
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.HodModelFactory`. 

        additional_haloprops : list of strings, optional keyword argument  
            Each entry in this list must be a column key of ``snapshot.halos``. 
            For each entry of ``additional_haloprops``, each member of the mock galaxy population 
            will have an attribute storing this property of its host halo. 
            The corresponding mock galaxy attribute name will be pre-pended by ``halo_``. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        halocut_funcobj : function object, optional keyword argument  
            Function object used to place a cut on the input ``snapshot.halos`` table. 
            Default behavior depends on the sub-class of `MockFactory`. 
            If the ``halocut_funcobj`` keyword argument is passed, 
            the input to the function must be a length-Nsubhalos structured numpy array or Astropy table; 
            the function output must be a length-Nsubhalos boolean array that will be used as a mask. 

        populate : boolean, optional keyword argument  
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Dictionary of function objects used to create additional halo properties 
            by `preprocess_halo_catalog`. Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `HodMockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 
        """

        super(HodMockFactory, self).__init__(populate=populate, **kwargs)

        self.preprocess_halo_catalog()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self, **kwargs):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. This pre-processing includes identifying the 
        catalog columns that will be used by the model to create the mock, 
        building lookup tables associated with the halo profile, 
        and possibly creating new halo properties. 

        Parameters 
        ----------
        logrmin : float, optional 
            Minimum radius used to build the lookup table for the halo profile. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        logrmax : float, optional 
            Maximum radius used to build the lookup table for the halo profile. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Npts_radius_table : int, optional 
            Number of control points used in the lookup table for the halo profile.
            Default is set in `~halotools.empirical_models.model_defaults`. 

        """

        ################ Make cuts on halo catalog ################
        # Select host halos only, since this is an HOD-style model
        host_halo_cut = (self.halos['halo_upid']==-1)
        self.halos = self.halos[host_halo_cut]

        # make a conservative mvir completeness cut 
        # This can be relaxed by changing sim_defaults.Num_ptcl_requirement
        cutoff_mvir = sim_defaults.Num_ptcl_requirement*self.snapshot.particle_mass
        mass_cut = (self.halos['halo_mvir'] > cutoff_mvir)
        self.halos = self.halos[mass_cut]

        # Make any additional cuts requested by the composite model
        if hasattr(self.model, 'halocut_funcobj'):
            self.halos = self.model.halocut_funcobj(halos=self.halos)
        ############################################################

        ### Create new columns of the halo catalog, if applicable
        if hasattr(self, 'new_haloprop_func_dict'):
            for new_haloprop_key, new_haloprop_func in self.new_haloprop_func_dict.iteritems():
                self.halos[new_haloprop_key] = new_haloprop_func(halos=self.halos)
                self.additional_haloprops.append(new_haloprop_key)

        # Create new columns for the halo catalog associated with each 
        # parameter of each halo profile model, e.g., 'NFWmodel_conc'. 
        # New column names are the keys of the halo_prof_func_dict dictionary; 
        # new column values are computed by the function objects in halo_prof_func_dict 
        for halo_prof_param_key in self.model.prof_param_keys:
            method_name = halo_prof_param_key + '_halos'
            method_behavior = getattr(self.model, method_name)
            self.halos[halo_prof_param_key] = method_behavior(halos=self.halos)
            self.additional_haloprops.append(halo_prof_param_key)

        self.model.build_halo_prof_lookup_tables(**kwargs)

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
            for prof_param_key in self.model.prof_param_keys:
                method_name = prof_param_key + '_' + gal_type
                method_behavior = getattr(self.model, method_name)
                self.galaxy_table[prof_param_key][gal_type_slice] = (
                    method_behavior(galaxy_table = self.galaxy_table[gal_type_slice])
                    )

            # Assign positions 
            pos_method_name = 'pos_'+gal_type

            self.galaxy_table['x'][gal_type_slice], \
            self.galaxy_table['y'][gal_type_slice], \
            self.galaxy_table['z'][gal_type_slice] = (
                getattr(self.model, pos_method_name)(
                    galaxy_table=self.galaxy_table[gal_type_slice])
                )
                
        # Positions are now assigned to all populations. 
        # Now enforce the periodic boundary conditions for all populations at once
        self.galaxy_table['x'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['x'], self.snapshot.Lbox)
        self.galaxy_table['y'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['y'], self.snapshot.Lbox)
        self.galaxy_table['z'] = model_helpers.enforce_periodicity_of_box(
            self.galaxy_table['z'], self.snapshot.Lbox)

        if hasattr(self.model, 'galaxy_selection_func'):
            mask = self.model.galaxy_selection_func(self.galaxy_table)
            self.galaxy_table = self.galaxy_table[mask]

    def allocate_memory(self):
        """ Method allocates the memory for all the numpy arrays 
        that will store the information about the mock. 
        These arrays are bound directly to the mock object. 

        The main bookkeeping devices generated by this method are 
        ``_occupation`` and ``_gal_type_indices``. 

        """

        self.galaxy_table = Table() 

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
        for galcatkey in self.model.prof_param_keys:
            self.galaxy_table[galcatkey] = np.zeros(self.Ngals, dtype = 'f4')

        self.galaxy_table['gal_type'] = np.zeros(self.Ngals, dtype=object)

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for key in phase_space_keys:
            self.galaxy_table[key] = np.zeros(self.Ngals, dtype = 'f4')


class SubhaloMockFactory(MockFactory):
    """ Class responsible for populating a simulation with a 
    population of mock galaxies.

    """

    def __init__(self, populate=True, **kwargs):
        """
        Parameters 
        ----------
        snapshot : object, keyword argument 
            Object containing the halo catalog and other associated data.  
            Produced by `~halotools.sim_manager.supported_sims.HaloCatalog`

        model : object, keyword argument
            A model built by a sub-class of `~halotools.empirical_models.SubhaloModelFactory`. 

        additional_haloprops : list of strings, optional keyword argument  
            Each entry in this list must be a column key of ``snapshot.halos``. 
            For each entry of ``additional_haloprops``, each member of the mock galaxy population 
            will have an attribute storing this property of its host halo. 
            The corresponding mock galaxy attribute name will be pre-pended by ``halo_``. 
            If ``additional_haloprops`` is set to the string value ``all``, 
            the galaxy table will inherit every halo property in the catalog. Default is None. 

        halocut_funcobj : function object, optional keyword argument  
            Function object used to place a cut on the input ``snapshot.halos`` table. 
            Default behavior depends on the sub-class of `MockFactory`. 
            If the ``halocut_funcobj`` keyword argument is passed, 
            the input to the function must be a length-Nsubhalos structured numpy array or Astropy table; 
            the function output must be a length-Nsubhalos boolean array that will be used as a mask. 

        populate : boolean, optional keyword argument  
            If set to ``False``, the class will perform all pre-processing tasks 
            but will not call the ``model`` to populate the ``galaxy_table`` 
            with mock galaxies and their observable properties. Default is ``True``. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Dictionary of function objects used to create additional halo properties 
            by `preprocess_halo_catalog`. Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `HodMockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 
        """

        super(SubhaloMockFactory, self).__init__(populate=populate, **kwargs)

        # Pre-compute any additional halo properties required by the model
        self.preprocess_halo_catalog()
        self.precompute_galprops()

        if populate is True:
            self.populate()

    def preprocess_halo_catalog(self):
        """ Method to pre-process a halo catalog upon instantiation of 
        the mock object. 
        """

        # Make any cuts on the halo catalog requested by the composite model
        if hasattr(self.model, 'halocut_funcobj'):
            self.halos = self.model.halocut_funcobj(halos=self.halos)

        ### Create new columns of the halo catalog, if applicable
        if hasattr(self.model, 'new_haloprop_func_dict'):
            for new_haloprop_key, new_haloprop_func in self.model.new_haloprop_func_dict.iteritems():
                self.halos[new_haloprop_key] = new_haloprop_func(halos=self.halos)
                self.additional_haloprops.append(new_haloprop_key)


    def precompute_galprops(self):
        """ Method pre-processes the input subhalo catalog, and pre-computes 
        all halo properties that will be inherited by the ``galaxy_table``. 
        """

        for key in self.additional_haloprops:
            newkey = model_defaults.host_haloprop_prefix + key
            self.galaxy_table[newkey] = self.halos[key]

        phase_space_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for newkey in phase_space_keys:
            oldkey = model_defaults.host_haloprop_prefix + newkey
            self.galaxy_table[newkey] = self.galaxy_table[oldkey]

        self.galaxy_table['galid'] = np.arange(len(self.galaxy_table))

        for galprop in self.model.galprop_list:
            component_model = self.model.model_blueprint[galprop]
            if hasattr(component_model, 'gal_type_func'):
                newkey = galprop + '_gal_type'
                self.galaxy_table[newkey] = (
                    component_model.gal_type_func(galaxy_table=self.galaxy_table)
                    )

    def populate(self):
        """ Method populating subhalos with mock galaxies. 
        """
        for galprop_key in self.model.galprop_list:
            
            model_func_name = 'mc_'+galprop_key
            model_func = getattr(self.model, model_func_name)
            self.galaxy_table[galprop_key] = model_func(galaxy_table=self.galaxy_table)

        if hasattr(self.model, 'galaxy_selection_func'):
            mask = self.model.galaxy_selection_func(self.galaxy_table)
            self.galaxy_table = self.galaxy_table[mask]


