.. _subhalo_modeling_tutorial3:

*********************************************************************
Example 3: A subhalo-based model with a feature of your own creation
*********************************************************************
.. currentmodule:: halotools.empirical_models

In this section of the :ref:`subhalo_modeling_tutorial0`, 
we'll build a composite model that includes a component model that 
is not part of Halotools, but that you yourself have written. 

TThere is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/subhalo_modeling/subhalo_modeling_tutorial3.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 

Overview of the new model
=============================

The model we'll build will be based on the ``behroozi10`` model, 
and we will use the ``baseline_model_instance`` feature 
described in the :ref:`baseline_model_instance_mechanism_subhalo_model_building` 
section of the documentation. 
In addition to the stellar mass modeled in ``behroozi10``, we'll add 
a component model that governs galaxy size. 
Our model for size will have no physical motivation whatsoever. That 
part is up to you. This tutorial just teaches you the mechanics of 
incorporating a new feature into the factory. 

Building the new component model
==================================

All component models are instances of python classes, 
so we will define a new class for our new feature. 
You can always brush up on python classes by reading the  
`Python documentation on classes <https://docs.python.org/2/tutorial/classes.html>`_. 
But for our purposes there is really only one basic thing to know. 
If you define a **__init__** method inside your class, then this 
method is what gets called whenever your class is instantiated. This means that 
any data that gets bound to *self* inside **__init__** will be bound to any 
instance of your class. 

The example source code below shows the basic pattern 
you need to match when writing your own component model. 
We'll unpack each line of this code in the discussion that follows.
However, for now, while reading this code take note of the big picture. 

    1. You need to provide a few pieces of boilerplate data in the **__init__** method so that the Halotools factory knows how to interface with your model. 

    2. You need to write the "physics function" that is responsible for the behavior of the model (**assign_size,** in this case). 

.. code:: python

    class Size(object):
    
        def __init__(self):
    
            self._mock_generation_calling_sequence = ['assign_size']
            self._galprop_dtypes_to_allocate = np.dtype([('galsize', 'f4')])
            self.list_of_haloprops_needed = ['halo_spin']
    
        def assign_size(self, **kwargs):
            table = kwargs['table']
            table['galsize'][:] = table['halo_spin']/5.

Now we'll build an instance of the *Size* component model  
and incorporate this feature into a composite model:

.. code:: python

    galaxy_size = Size()

    from halotools.empirical_models import PrebuiltSubhaloModelFactory, SubhaloModelFactory
    behroozi10_model = PrebuiltSubhaloModelFactory('behroozi10')
    new_model = SubhaloModelFactory(baseline_model_instance = behroozi10_model, 
                                size = galaxy_size)

    # Your new model can generate a mock in the same way as always
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname = 'bolshoi')
    new_model.populate_mock(halocat)

The **__init__** method of your component model 
===========================================================================

There are three lines of code here, and each of them binds some new data 
to the class instance. Thus the *component_model_instance* above 
will have three attributes: ``_mock_generation_calling_sequence``, 
``_galprop_dtypes_to_allocate`` and ``list_of_haloprops_needed``. 
Each of these attributes plays an important role in structuring the interface 
between your model and the `SubhaloModelFactory`, so we'll now discuss 
them one by one. 

.. _role_of_subhalo_mock_generation_calling_sequence:

The role of the `_mock_generation_calling_sequence`
-----------------------------------------------------

During the generation of a mock catalog, the `SubhaloMockFactory` calls upon 
the component models one-by-one to assign their properties to the ``galaxy_table``. 
When each component is called upon, every method whose name appears in 
the component model's ``_mock_generation_calling_sequence`` gets passed 
the ``galaxy_table``. These methods are called in the order they appear 
in the ``_mock_generation_calling_sequence`` list. So the purpose of 
the ``_mock_generation_calling_sequence`` list is to inform the 
`SubhaloModelFactory` what to do when it comes time for the 
component model to play its role in creating the mock galaxy distribution. 

See the :ref:`mock_generation_calling_sequence_mechanism` section of the 
:ref:`composite_model_constructor_bookkeeping_mechanisms` page 
for further discussion. 

The role of the `_galprop_dtypes_to_allocate`
-----------------------------------------------------
One of the tasks handled by the `SubhaloMockFactory` is the allocation of 
the appropriate memory that will be stored in your ``galaxy_table``. 
For every galaxy property in a composite model, there needs to be a 
corresponding column of the ``galaxy_table`` of the appropriate data type. 
The ``_galprop_dtypes_to_allocate`` ensures that this is the case. 

The way this works is that every component model must declare a 
`numpy.dtype` object and bind it to the 
``_galprop_dtypes_to_allocate`` attribute of the composite model instance. 
You can read more about Numpy `~numpy.dtype` objects in the Numpy documentation, 
but the basic syntax is illustrated in the source code above: 
our new column will be named ``galsize``, and each row stores a float. 

You can see how to alter the syntax for the case of a component model that assigns 
more than one galaxy property in the next example of this tutorial. 
See the :ref:`galprop_dtypes_to_allocate_mechanism` section of the 
:ref:`composite_model_constructor_bookkeeping_mechanisms` documentation page 
for further discussion. 

The role of the `list_of_haloprops_needed`
-----------------------------------------------------

This attribute provides a list of all the keys of the ``halo_table`` that 
the functions appearing ``_mock_generation_calling_sequence`` will need to access 
during mock population. For example, the **assign_size** method requires 
access to the ``halo_spin`` column, and so the ``halo_spin`` string appears in 
``list_of_haloprops_needed``. This is fairly self-explanatory, but you can 
read more about the under-the-hood details in the 
:ref:`list_of_haloprops_needed_mechanism` section of the 
:ref:`composite_model_constructor_bookkeeping_mechanisms` documentation page. 

The "physics function" of your component model 
==================================================

In the example above, there is just one function responsible for 
the physics underlying this model: **assign_size.** 
The behavior of this simple function is pretty trivial: 
whatever the spin of the halo is, divide it by five and call the result 
the size of the galaxy. Obviously this is physically silly, but the 
calling signature illustrates how to write your more physically realistic model. 

Any method in your ``_mock_generation_calling_sequence`` must accept a ``table`` keyword 
argument. That is because when the `MockFactory` calls your component model 
methods, it will pass the ``galaxy_table`` that is being built to each of your 
physics functions via a ``table`` keyword argument. 
The simplest way to handle this in a way that will be compatible with future Halotools 
updates is to have your physics functions use the Python built-in kwargs mechanism, 
as demonstrated in the source code above. 

In order for your model's underlying physics to propagate into the properties 
of the mock galaxy population, your physics function(s) must write 
values to the appropriate column of the ``table``. In this case, 
we wrote to the ``galsize`` column. The **[:]** syntax is not strictly necessary, 
but it is good practice to use it because it ensures that an exception will be raised 
if you attempt to write to column that does not exist in the ``galaxy_table``; 
you can omit the **[:]** if you want to eschew this safety mechanism, 
but there is no difference in performance and so this is syntax is recommended 
as a sanity check on all the bookkeeping. 

This tutorial continues with :ref:`subhalo_modeling_tutorial4`. 

