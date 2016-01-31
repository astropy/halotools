.. _hod_modeling_tutorial5:

****************************************************************
Example 5: An HOD model with cross-component dependencies
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the :ref:`hod_modeling_tutorial0`, 
illustrates an example of a component model that 
depends on the results of some other component model. 

Overview of the Example 5 HOD model 
====================================

The model we'll build will only have central galaxies, no satellites. 
This is perfectly permissible: Halotools places no restrictions on the 
number or kind of subpopulations of galaxies that must be present 
in the universe you create. Our population of centrals will 
be based on the models used in the ``leauthaud11`` HOD model. 
Additionally, we'll add two new component models: 
one governing galaxy shape, a second governing galaxy size. 

In terms of implementation details, 
the new feature to focus on here is this: the model for galaxy 
size will have an explicit dependence on galaxy size, even though 
these models are controlled by independently defined components. 
Again, our model will not be physically motivated, 
and the purpose will be to teach you how to 
build a model with inter-dependence between different components.  

Briefly, in our model the shape of a galaxy will be randomly selected 
to be either a disk or an elliptical. The size of disk galaxies 
will be whatever the spin of the halo is, and the size of elliptical 
galaxies will be the value of a custom-defined halo property 
computed in a pre-processing phase. To streamline the presentation, 
we will omit the features described in the previous example 
and focus on just the new features introduced in this example. 

Source code for the new model 
=================================
.. code:: python

    class Shape(object):
        
        def __init__(self, gal_type):

            self.gal_type = gal_type
            self._mock_generation_calling_sequence = ['assign_shape']
            self._galprop_dtypes_to_allocate = np.dtype([('shape', object)])

        def assign_shape(self, **kwargs):
            table = kwargs['table']
            randomizer = np.random.random(len(table))
            table['shape'][:] = np.where(randomizer > 0.5, 'elliptical', 'disk')

    class Size(object):

        def __init__(self, gal_type):

            self.gal_type = gal_type
            self._mock_generation_calling_sequence = ['assign_size']
            self._galprop_dtypes_to_allocate = np.dtype([('galsize', 'f4')])

            self.new_haloprop_func_dict = {'halo_custom_size': self.calculate_halo_size}

        def assign_size(self, **kwargs):
            table = kwargs['table']
            disk_mask = table['shape'] == 'disk'
            table['galsize'][disk_mask] = table['halo_spin'][disk_mask]
            table['galsize'][~disk_mask] = table['halo_custom_size'][~disk_mask]

        def calculate_halo_size(self, **kwargs):
            table = kwargs['table']
            return 0.5*table['halo_rvir']*table['halo_rs']


Now we'll build our composite model using the ``model_feature_calling_sequence``, 
a new keyword introduced in this tutorial:

.. code:: python

    from halotools.empirical_models import Leauthaud11Cens, TrivialPhaseSpace
    cen_occupation = Leauthaud11Cens()
    cen_profile = TrivialPhaseSpace(gal_type = 'centrals')
    cen_shape = Shape(gal_type = 'centrals')
    cen_size = Size(gal_type = 'centrals')

    from halotools.empirical_models import HodModelFactory
    model = HodModelFactory(
        centrals_occupation = cen_occupation, 
        centrals_profile = cen_profile, 
        centrals_shape = cen_shape, 
        centrals_size = cen_size, 
        model_feature_calling_sequence = ('centrals_occupation', 
            'centrals_profile', 'centrals_shape', 'centrals_size')
        )























