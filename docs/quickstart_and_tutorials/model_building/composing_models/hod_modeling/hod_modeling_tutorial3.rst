.. _hod_modeling_tutorial3:

*******************************************************************
Example 3: An HOD-style model with a feature of your own creation
*******************************************************************

.. currentmodule:: halotools.empirical_models

In this section of the :ref:`hod_modeling_tutorial0`, 
we'll build a composite model that includes a component model that 
is not part of Halotools, but that you yourself have written. 

Overview of the new model
=============================

The model we'll build will be based on the ``zheng07`` HOD, 
and we will use the ``baseline_model_instance`` feature 
described in the :ref:`baseline_model_instance_mechanism_hod_building` 
section of the documentation. 
In addition to the basic ``zheng07`` features, we'll add 
a component model that governs galaxy size. 
Our model for size will have no physical motivation whatsoever. That 
part is up to you. This tutorial just teaches you the mechanics of 
incorporating a new feature into the factory. 

Building the new component model
==================================

All component models are instances of python classes, 
so we will define a new class for our new feature. 
The example source code below shows the basic pattern 
you need to match when writing your own component model. 
We'll unpack each line of this code in the discussion that follows.

However, for now, while reading this code take note of the big picture. 

    1. You need to provide a few pieces of boilerplate data in the **__init__** method so that the Halotools factory knows how to interface with your model. 

    2. You need to write the "physics function" that is responsible for the behavior of the model (**assign_size,** in this case). 

.. code:: python

    class Size(object):
        
        def __init__(self, gal_type):

            self.gal_type = gal_type
            self._mock_generation_calling_sequence = ['assign_size']
            self._galprop_dtypes_to_allocate = np.dtype([('galsize', 'f4')])
            self.list_of_haloprops_needed = ['halo_spin']
            
        def assign_size(self, table):
            
            table['galsize'][:] = table['halo_spin']/5.
            























