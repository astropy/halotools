.. _subhalo_modeling_tutorial4:


****************************************************************
Example 4: A more complex subhalo-based component model 
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the :ref:`subhalo_modeling_tutorial0`, 
illustrates a more complex example of a component model that 
that you have written yourself. What follows is basically just 
a more full-featured version of the previous 
:ref:`subhalo_modeling_tutorial3` that illustrates a few more tricks. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/hod_modeling/subhalo_modeling_tutorial4.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 


.. _overview_subhalo_model_tutorial_example4: 

Overview of the Example 4 subhalo-based model 
===============================================

The model we'll build will be based on the ``behroozi10`` model, 
and we will use the ``baseline_model_instance`` feature 
described in the :ref:`baseline_model_instance_mechanism_subhalo_model_building` 
section of the documentation. 
In addition to the basic ``behroozi10`` features, we'll add 
a component model that governs galaxy shape. 
Again, our model will not be physically motivated, but we will 
introduce some implementation complexity to teach you how to 
build models with sophisticated features.  

In this simple model, galaxy shape is characterized by two properties:
an *axis_ratio*, and whether or not the galaxy is *disrupted.* 
Galaxies living in halos above some critical mass are never disrupted; 
galaxies living in halos below this mass have a random chance of being disrupted. 
Disrupted galaxies are assigned a random *axis_ratio*; 
non-disrupted galaxies all have *axis_ratio = 0.3*. 
Both the critical mass and the random disruption chance 
are continuously variable parameters of the model. 
The model applies to both centrals and satellites, 
for which the disruption fraction parameters 
are independently specified.  

Source code for the new component models
----------------------------------------

.. code:: python

    class Shape(object):
    
        def __init__(self, prim_haloprop_key):
    
            self._mock_generation_calling_sequence = (
                ['assign_disrupted', 'assign_axis_ratio'])
            self._galprop_dtypes_to_allocate = np.dtype(
                [('axis_ratio', 'f4'), ('disrupted', bool)])
            self.list_of_haloprops_needed = ['halo_spin', 'halo_upid']
    
            self.prim_haloprop_key = prim_haloprop_key
            self._methods_to_inherit = (
                ['assign_disrupted', 'assign_axis_ratio', 
                'disrupted_fraction_vs_halo_mass_centrals', 
                'disrupted_fraction_vs_halo_mass_satellites'])
            self.param_dict = ({
                'max_disruption_mass': 1e13, 
                'disrupted_fraction_centrals': 0.25, 
                'disrupted_fraction_satellites': 0.35
                        })
    
        def assign_disrupted(self, **kwargs):
            
            table = kwargs['table']
            upid = table['halo_upid']
            halo_mass = table['halo_mvir']
    
            disrupted_fraction = np.empty_like(halo_mass)
            central_mask = upid == -1
            disrupted_fraction[central_mask] = (
                self.disrupted_fraction_vs_halo_mass_centrals(halo_mass[central_mask]))
            disrupted_fraction[~central_mask] = (
                self.disrupted_fraction_vs_halo_mass_satellites(halo_mass[~central_mask]))
            
            randomizer = np.random.uniform(0, 1, len(halo_mass))
            is_disrupted = randomizer < disrupted_fraction
    
            if 'table' in kwargs.keys():
                table['disrupted'][:] = is_disrupted
            else:
                return is_disrupted
    
        def assign_axis_ratio(self, **kwargs):
            
            table = kwargs['table']
            mask = table['disrupted'] == True
            num_disrupted = len(table['disrupted'][mask])
            table['axis_ratio'][mask] = np.random.random(num_disrupted)
            table['axis_ratio'][~mask] = 0.3
    
        def disrupted_fraction_vs_halo_mass_centrals(self, mass):
            
            bool_mask = mass > self.param_dict['max_disruption_mass']
            val = self.param_dict['disrupted_fraction_centrals']
            return np.where(bool_mask == True, 0, val)
    
        def disrupted_fraction_vs_halo_mass_satellites(self, mass):
            
            bool_mask = mass > self.param_dict['max_disruption_mass']
            val = self.param_dict['disrupted_fraction_satellites']
            return np.where(bool_mask == True, 0, val)

You incorporate this new component into a composite model in the same way as before:

.. code:: python

    galaxy_shape = Shape('halo_mvir')
    from halotools.empirical_models import PrebuiltSubhaloModelFactory, SubhaloModelFactory
    behroozi_model = PrebuiltSubhaloModelFactory('behroozi10')
    new_model = SubhaloModelFactory(baseline_model_instance = behroozi_model, 
        shape = galaxy_shape)
    
The **__init__** method of the component model 
===========================================================================

The first four lines in the **__init__** method should be familiar 
from :ref:`subhalo_modeling_tutorial3`. In this example, there will be two 
different physics functions called and two different galaxy properties 
assigned. The **assign_disrupted** method must be called before the 
**assign_axis_ratio** method, because the *axis_ratio* has an explicit 
dependence upon disruption designation, as described in :ref:`overview_subhalo_model_tutorial_example4`. 

The role of ``prim_haloprop_key``
--------------------------------------

Next, by allowing you to build a model by passing in *prim_haloprop_key*, 
you open up the possibility of building models based on different 
choices for the halo mass definition. This trick is used ubiquitously throughout the 
Halotools code base. When the `SubhaloModelFactory` detects that a 
*prim_haloprop_key* attribute is present in a component model, the string 
bound to that attribute is automatically added to the ``list_of_haloprops_needed``. 

The role of ``_methods_to_inherit``
--------------------------------------

The next new feature that appears in the **__init__** method is the 
``_methods_to_inherit`` attribute. This list controls what methods the composite 
model will inherit from the component model. If you do not specify this list 
(we did not specify it in the previous example), then the `SubhaloModelFactory` 
will assume that the only methods you want your composite model to inherit 
are the methods appearing in ``_mock_generation_calling_sequence``. 
However, our *Shape* model has interesting ancillary functions 
**disrupted_fraction_vs_halo_mass_centrals** and **disrupted_fraction_vs_halo_mass_satellites**; 
we may wish to study these functions on their own, even if only to make plots 
(see the accompanying IPython Notebook for a demonstration). 
This is enabled by adding these method names to the 
``_methods_to_inherit`` list. Note that if you do choose to define this list 
inside **__init__**, it is required that every method name appearing in 
the ``_mock_generation_calling_sequence`` also appears in ``_methods_to_inherit``, 
or Halotools will raise an exception. 

The role of ``param_dict``
----------------------------

As described in the :ref:`param_dict_mechanism` section of the 
:ref:`composite_model_constructor_bookkeeping_mechanisms` documentation page, 
the ``param_dict`` mechanism allows you to control the behavior 
of your model with tunable parameters, as is done, for example, 
in an MCMC-type likelihood analysis. By defining our physics functions 
to depend on the values stored in the component model ``param_dict``, 
we can modify the behavior of our component model instance by 
changing the values stored in this dictionary. 

One detail to pay special attention to is the naming of the keys in your 
component model dictionary. 
Galaxy shape is controlled by an instance of the *Shape* component model. 
And the composite model ``param_dict`` is built simply 
by concatenating the key:value pairs of each component model ``param_dict``. 
So when the composite model is built, if there is not some way to differentiate 
between the parameter names belonging to the shape component of centrals vs. 
satellites, then there is no way to independently modify one set of parameters 
vs. the other. 
By defining distinctive names for the keys of the component model ``param_dict`` 
that are specific to the feature, as we have done here, we protect against the 
possibility that some other component model has a ``param_dict`` key with the same name. 
In cases where there is such duplication, Halotools will always raise 
a warning so that you can assess whether the key repetition is intended. 



The "physics functions" of the component model 
===========================================================================

The physics functions in the *Shape* class differ from the one covered in the 
*Size* class of the previous example in one important respect: 
the *disrupted* column of the ``galaxy_table`` must be assigned before 
the **assign_axis_ratio** method is called in order to get sensible results. 
This is guaranteed by proper use of the ``_mock_generation_calling_sequence``, 
as described above. 

There is another, less important difference to notice: the 
**disrupted_fraction_vs_halo_mass_centrals** and 
**disrupted_fraction_vs_halo_mass_satellites**
methods accept either a  plain Numpy array positional argument 
rather than a ``table`` keyword argument. As discussed in the previous example, 
it is compulsory for any physics function that appears in ``_mock_generation_calling_sequence``
to accept a ``table`` keyword argument, 
and for the appropriate columns of the input ``table`` to be overwritten as necessary. 
That is because the `SubhaloMockFactory` always passes a ``galaxy_table`` to each physics 
function via the ``table`` keyword argument. However, any internal function you write 
can accept whatever arguments you find most convenient to work with. 

This tutorial continues with :ref:`subhalo_modeling_tutorial5`. 

