:orphan:

.. _writing_your_own_hod_occupation_component:

***********************************************************************
Writing your own HOD occupation component
***********************************************************************

As discussed in :ref:`hod_modeling_tutorial1`, the component model governing the 
abundance of galaxies in a halo is a required component in all HOD-style models. 
This section of the documentation describes how you can write your own 
component model governing HOD-style occupation statistics in a way 
that will interface properly with the `HodMockFactory`. 

Preliminary comments 
=======================

Unlike writing component models for other kinds of features of the galaxy population, 
when writing an occupation component you must subclass from the `OccupationComponent` 
class. The reason for this requirement is simple. 
The occupation component is plays a special role in the Halotools implementation 
of HOD-style mock making because the result returned by this component determine how 
much memory should be allocated for the galaxy population being modeled. 
This is unlike typical component model features, which simply add new columns to an 
existing ``galaxy_table`` whose length has already been determined. Thus 
there are a few special methods and attributes that the `HodMockFactory` 
assumes will be present of any occupation component. 

The way Halotools enforces the presence of these special methods and attributes is as follows. 
When building an HOD-style model, 
for every ``gal_type`` in the composite model you are required to pass in 
a keyword of the form ``gal_type_occupation``. Any time the `HodModelFactory` detects 
a keyword that concludes in ``_occupation``, the model factory requires that the 
object bound to this keyword be an instance of a sub-class of `OccupationComponent`. 
If you are able to write your occupation component model and successfully instantiate it, 
then the internals of the `OccupationComponent` guarantee that you have successfully 
met the necessary requirements. 

In the next section, we'll dive into the source code of a minimal example of source code 
for a working occupation component model and dissect each necessary ingredient 
one by one. 

Source code for a minimal example of an occupation component model
=====================================================================

.. code:: python

	from halotools.empirical_models import OccupationComponent 

	class MyCentralOccupation(OccupationComponent):

		def __init__(self, gal_type, threshold):

			OccupationComponent.__init__(self, 
				gal_type = gal_type, 
				threshold = threshold, 
				upper_occupation_bound = 1)

		def mean_occupation(self, **kwargs):
			table = kwargs['table']
			return np.zeros(len(table)) + 0.1

Unpacking the source code 
=============================

The ``MyCentralOccupation`` class does just three things:

	1. Sub-classes from OccupationComponent

	2. Calls **__init__** of the `OccupationComponent` parent class from within its own constructor. 

	3. Defines a ``mean_occupation`` method. 

So long as your occupation component model does these three things, 
then your model will work with the `HodMockFactory`. 

Defining the ``mean_occupation`` method 
=========================================

As described in the :ref:`physics_function_hod_explanation` section of the 
:ref:`hod_modeling_tutorial3` documentation page, your ``mean_occupation`` method 
must accept a ``table`` keyword argument. The value it returns should be an array 
of the same length as the input table, and it should be bounded by the 
``upper_occupation_bound`` (more about this later in this page). 
So long as your ``mean_occupation`` function conforms to these requirements, 
you have written an acceptable function. 

Defining the ``__init__`` constructor 
========================================
You are free to write your own **__init__** function however you like, 
so long as you call the **__init__** function of the ``OccupationComponent`` class 
somewhere within it. The `OccupationComponent` constructor has the following required 
keyword arguments: ``gal_type``, ``threshold`` and ``upper_occupation_bound``. 
The ``gal_type`` can be any string, 
though this should be chosen to match the name of the population you are modeling 
in the composite model. 
The ``threshold`` argument refers to the minimum value of the primary galaxy property, 
e.g., -20 if the galaxy sample were selected to be brighter than some limiting 
r-band absolute magnitude, or 10.5 for :math:`{\rm log}_{10}M_{\ast}`. 
The value you choose for ``threshold`` need not necessarily have any impact on the 
behavior of your model, it just needs to be present. 

Setting the value of the ``upper_occupation_bound``
------------------------------------------------------
The value you choose for ``upper_occupation_bound`` has important significance 
for the behavior of your model. When defining an occupation component, 
the actual function called during the mock population sequence is ``mc_occupation``, 
which is defined in the `OccupationComponent` super-class. 
If you set ``upper_occupation_bound`` to 1, then the distribution of galaxy abundances 
in your model will be assumed to be a nearest-integer distribution, as is the case for a
central galaxy population. If you set ``upper_occupation_bound`` to ``np.inf``, 
then your distribution will be assumed to be a Poisson, as is the case for typical 
satellite galaxy occupation models. 
If you choose some other value, then you must provide your own implementation of the 
``mc_occupation`` method that over-rides the method in the super-class. 


Source code for an occupation component model with a custom ``mc_occupation`` method
======================================================================================

In case you wish to study occupation statistics that deviate from nearest-integer or Poisson 
statistics, the example below shows how to model a satellite population with a 
physically-silly-but-distinct ``mc_occupation`` method in which a halo has either 0 or 5 satellites. 
By matching this pattern you can write our own occupation component and have total control over 
the occupation statistics in your model. 

.. code:: python

	from halotools.empirical_models import OccupationComponent 

	class MySatelliteOccupation(OccupationComponent):

		def __init__(self, threshold):

			OccupationComponent.__init__(self, 
				gal_type = 'satellites', 
				threshold = threshold, 
				upper_occupation_bound = 5)

		def mean_occupation(self, **kwargs):
			table = kwargs['table']
			return np.zeros(len(table)) + 2.5

		def mc_occupation(self, **kwargs):
			table = kwargs['table']
			meanocc = self.mean_occupation(**kwargs)
			result = np.where(meanocc < 2.5, 0, 5)
			table['halo_num_satellites'] = result
			return result














