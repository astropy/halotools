.. _hod_modeling_tutorial4:

****************************************************************
Example 4: A more complex HOD component model 
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the :ref:`hod_modeling_tutorial0`, 
illustrates a more complex example of a component model that 
that you have written yourself. What follows is basically just 
a more full-features version of the previous 
:ref:`hod_modeling_tutorial3` that illustrates a few more tricks. 

.. _overview_hod_tutorial_example4: 

Overview of the Example 4 HOD model 
====================================

The model we'll build will be based on the ``zheng07`` HOD, 
and we will use the ``baseline_model_instance`` feature 
described in the :ref:`baseline_model_instance_mechanism_hod_building` 
section of the documentation. 
In addition to the basic ``zheng07`` features, we'll add 
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
for which these parameters are independently specified.  


Source code for the new model 
=================================
.. code:: python

    class Shape(object):
        
        def __init__(self, gal_type, prim_haloprop_key):

            self.gal_type = gal_type
            self._mock_generation_calling_sequence = (
                ['assign_disrupted', 'assign_axis_ratio'])
            self._galprop_dtypes_to_allocate = np.dtype(
                [('axis_ratio', 'f4'), ('disrupted', bool)])
            self.list_of_haloprops_needed = ['halo_spin']

            self.prim_haloprop_key = prim_haloprop_key
            self._methods_to_inherit = (
                ['assign_disrupted', 'assign_axis_ratio', 
                'disrupted_fraction_vs_halo_mass'])
            self.param_dict = ({
                'max_disruption_mass_'+self.gal_type: 1e12, 
                'disrupted_fraction_'+self.gal_type: 0.25})

        def assign_disrupted(self, **kwargs):
            if 'table' in kwargs.keys():
                table = kwargs['table']
                halo_mass = table[self.prim_haloprop_key]
            else:
                halo_mass = kwargs['prim_haloprop']

            disrupted_fraction = self.disrupted_fraction_vs_halo_mass(halo_mass)
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
            table['axis_ratio'][mask][:] = np.random.random(num_disrupted)
            table['axis_ratio'][~mask][:] = 0.3

        def disrupted_fraction_vs_halo_mass(self, mass):
            bool_mask = mass > self.param_dict['max_disruption_mass_'+self.gal_type]
            val = self.param_dict['disrupted_fraction_'+self.gal_type]
            return np.where(bool_mask == True, 0, val)

You incorporate this new component into a composite model in the same way as before:

.. code:: python 

    cen_shape = Shape('centrals', 'halo_mvir')
    sat_shape = Shape('satellites', 'halo_m200b')
    from halotools.empirical_models import PrebuiltHodModelFactory, HodModelFactory
    zheng_model = PrebuiltHodModelFactory('zheng07')
    new_model = HodModelFactory(baseline_model_instance = zheng_model, 
        centrals_shape = cen_shape, satellites_shape = sat_shape)


The **__init__** method of the component model 
===========================================================================

The first four lines in the **__init__** method should be familiar 
from :ref:`hod_modeling_tutorial3`. In this example, there will be two 
different physics functions called and two different galaxy properties 
assigned. The **assign_disrupted** method must be called before the 
**assign_axis_ratio** method, because the *axis_ratio* has an explicit 
dependence upon disruption designation, as described in :ref:`overview_hod_tutorial_example4`. 

The role of ``prim_haloprop_key``
--------------------------------------

Next, by allowing you to build a model by passing in *prim_haloprop_key*, 
you open up the possibility of building models based on different 
choices for the halo mass definition. This trick is used ubiquitously throughout the 
Halotools code base. When the `HodModelFactory` detects that a 
*prim_haloprop_key* attribute is present in a component model, the string 
bound to that attribute is automatically added to the ``list_of_haloprops_needed``. 

The role of ``_methods_to_inherit``
--------------------------------------

The next new feature that appears in the **__init__** method is the 
``_methods_to_inherit`` attribute. This list controls what methods the composite 
model will inherit from the component model. If you do not specify this list 
(we did not specify it in the previous example), then the `HodModelFactory` 
will assume that the only methods you want your composite model to inherit 
are the methods appearing in ``_mock_generation_calling_sequence``. 
However, our *Shape* model has an interesting ancillary function 
**disrupted_fraction_vs_halo_mass** that we may wish to study on its own, 
even if only to make plots. This is enabled by adding this method name to the 
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

One detail to pay special attention to is how the keys of this dictionary 
are defined in this example. Note that each key contains a substring that 
is set by the ``gal_type``. In this example, we do this for a specific reason. 
The shape of both centrals and satellites are controlled by instances of the 
*Shape* component model. And the composite model ``param_dict`` is built simply 
by concatenating the key:value pairs of each component model ``param_dict``. 
So when the composite model is built, if there is not some way to differentiate 
between the parameter names belonging to the shape component of centrals vs. 
satellites, then there is no way to independently modify one set of parameters 
vs. the other. By defining names for the keys of the component model ``param_dict`` 
that vary with the ``gal_type``, as we have done here, this issue is resolved 
because the composite model will have four parameters with unambiguous interpretations: 
``max_disruption_mass_centrals``, ``max_disruption_mass_satellites``, 
``disrupted_fraction_centrals`` and ``disrupted_fraction_satellites``. 
If we were using entirely separate classes to define shapes of centrals and satellites, 
this precaution would be unnecessary. 

The "physics functions" of the component model 
===========================================================================

The physics functions in the *Shape* class differ from the one covered in the 
*Size* class of the previous example in one important respect: 
the *disrupted* column of the ``galaxy_table`` must be assigned before 
the **assign_axis_ratio** method is called in order to get sensible results. 
This is guaranteed by proper use of the ``_mock_generation_calling_sequence``, 
as described above. 

There is another, less important difference to notice: the 
**assign_disrupted** method accepts either a ``table`` keyword argument 
or a ``prim_haloprop`` argument. As discussed in the previous example, 
it is compulsory for any physics function to accept a ``table`` keyword argument, 
and for the appropriate columns of the input ``table`` to be overwritten as necessary. 
That is because the `HodMockFactory` always passes a ``galaxy_table`` to each physics 
function via the ``table`` keyword argument. However, nothing stops you from 
adding functionality to your physics functions so that they will behave differently 
in different circumstances. In this example, your composite model will have the freedom 
to pass in an array of halo masses and the **assign_disrupted** method will 
return the array of values it *would* have assigned had the method been called during 
mock generation. This functionality can be useful for making plots, tracking down bugs, 
or just generally studying the behavior of your component model. 
All Halotools-provided component models support passing in arrays of the relevant 
quantities for exactly this purpose. In Halotools, the convention is that these arrays 
are passed in via the ``prim_haloprop`` keyword, though this you need not follow 
this convention if you find some other syntax more convenient or intuitive. 











            
