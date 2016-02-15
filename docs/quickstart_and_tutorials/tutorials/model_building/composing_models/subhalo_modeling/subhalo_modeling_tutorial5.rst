.. _subhalo_modeling_tutorial5:

*********************************************************************
Example 5: A subhalo-based model with cross-component dependencies
*********************************************************************

.. currentmodule:: halotools.empirical_models

This section of the :ref:`subhalo_modeling_tutorial0`, 
illustrates an example of a component model that 
depends on the results of some other, independently defined component model. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/hod_modeling/subhalo_modeling_tutorial5.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 

Overview of the Example 5 model 
====================================

The model we'll build will be based on the ``behroozi10`` model. 
Additionally, we'll add two new component models: 
one governing galaxy shape, a second governing galaxy size. 

In terms of implementation details, 
the new feature to focus on here is this: the model for galaxy 
size will have an explicit dependence on galaxy shape, even though 
these models are controlled by independently defined components. 
Again, our model will not be physically motivated: 
the purpose is to teach you how to 
build a model with inter-dependence between different components.  

Briefly, in our model the shape of a galaxy will be randomly selected 
to be either a disk or an elliptical. The size of disk galaxies 
will be whatever the spin of the halo is, and the size of elliptical 
galaxies will be the value of a custom-defined halo property 
computed in a pre-processing phase. To streamline the presentation, 
we will omit the features described in the previous example 
and focus on just the new features introduced in this example. 

Source code for the new model
-----------------------------

.. code:: python

    class Shape(object):
    
        def __init__(self):
    
            self._mock_generation_calling_sequence = ['assign_shape']
            self._galprop_dtypes_to_allocate = np.dtype([('shape', object)])
    
        def assign_shape(self, **kwargs):
            table = kwargs['table']
            randomizer = np.random.random(len(table))
            table['shape'][:] = np.where(randomizer > 0.5, 'elliptical', 'disk')
    
    class Size(object):
    
        def __init__(self):
    
            self._mock_generation_calling_sequence = ['assign_size']
            self._galprop_dtypes_to_allocate = np.dtype([('galsize', 'f4')])
            self.list_of_haloprops_needed = ['halo_spin']
            
            self.new_haloprop_func_dict = {'halo_custom_size': self.calculate_halo_size}
    
        def assign_size(self, **kwargs):
            table = kwargs['table']
            disk_mask = table['shape'] == 'disk'
            table['galsize'][disk_mask] = table['halo_spin'][disk_mask]
            table['galsize'][~disk_mask] = table['halo_custom_size'][~disk_mask]
    
        def calculate_halo_size(self, **kwargs):
            table = kwargs['table']
            return 2*table['halo_rs']

Now we'll build our composite model using the ``model_feature_calling_sequence``, 
a new keyword introduced in this tutorial:

.. code:: python

    from halotools.empirical_models import Behroozi10SmHm
    mstar_model = Behroozi10SmHm(redshift = 0)
    shape_model = Shape()
    size_model = Size()
    
    from halotools.empirical_models import SubhaloModelFactory
    model = SubhaloModelFactory(
        stellar_mass = mstar_model, 
        size = size_model, 
        shape = shape_model, 
        model_feature_calling_sequence = ('stellar_mass','shape', 'size')
        )



The **__init__** method of the component models 
===========================================================================

All features in the constructor of the *Shape* class have been covered 
previously in this tutorial. The only thing that may be worth noting here
is that one of the physics functions assigns a string: 
**assign_shape** writes either ``elliptical`` or ``disk``. 
In such a case, when declaring the ``_galprop_dtypes_to_allocate``, 
the most robust way to handle this is to use a Python object as the `numpy.dtype`. 

The role of ``new_haloprop_func_dict`` 
------------------------------------------
The *Size* component model illustrates the use of the 
``new_haloprop_func_dict`` feature. As described in the 
:ref:`new_haloprop_func_dict_mechanism` section of the 
:ref:`composite_model_constructor_bookkeeping_mechanisms` documentation page, 
this feature allows you to add new columns to the ``halo_table`` in a 
pre-processing phase of mock-making. Here we use this mechanism 
to add a new column to the halo catalog called ``halo_custom_size``, 
which in this case is twice the NFW scale radius. 
This mechanism is necessary because the **assign_size** method expects the 
``halo_custom_size`` column to be present in the ``table`` passed to it. 
The way the ``new_haloprop_func_dict`` mechanism works is this: 
it stores a dictionary whose key(s) is the name of the new halo column 
that will be created, and the value bound to that key is a function 
object that computes this quantity. 
See :ref:`new_haloprop_func_dict_mechanism` for further discussion. 

The "physics functions" of the component models 
===========================================================================

The physics function in the *Size* class differs from those covered previously 
in a subtle but critical detail: the **assign_size** method requires that 
the ``galaxy_table`` has a column called ``shape`` that has already been 
assigned sensible values, but yet this assignment is not carried out 
by the *Size* class itself, it is carried out by the *Shape* class. 
This means that we need to make sure that during the process of mock generation, 
the physics functions in the *Shape* class are called before the physics 
functions of the *Size* class. 

The order in which the component models are called is controllable by the 
``model_feature_calling_sequence``. Previously, we did not use this keyword. 
When this keyword is not supplied, the default behavior is for the 
``stellar_mass`` component to be called first (if it is present), and all other 
features to be called in a random order. By explicitly listing the features 
of your model in the ``model_feature_calling_sequence`` keyword, 
you override this default behavior with your own calling sequence. 

Concluding comments 
=====================
This example concludes the tutorial on subhalo-based model building. 
If you have further questions on how to build models, 
please contact the Halotools developers and/or raise an Issue on GitHub. 
























