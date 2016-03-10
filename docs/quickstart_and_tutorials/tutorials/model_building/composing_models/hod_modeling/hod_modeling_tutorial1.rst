.. _hod_modeling_tutorial1:

****************************************************************
Example 1: Building a simple HOD-style model
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes how you can use the 
`HodModelFactory` to build one of the simplest and most widely 
used galaxy-halo models in the literature: the HOD based on 
`Zheng et al 2007 <http://arxiv.org/abs/astro-ph/0703457>`_. 
By following this example, you will learn the basic 
calling sequence for building HOD models. 
We will build an exact replica of the ``zheng07`` pre-built model, 
so you will also learn how the `PrebuiltHodModelFactory` is really just 
"syntax candy" for the `HodModelFactory`. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


	**halotools/docs/notebooks/hod_modeling/hod_modeling_tutorial1.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 


Preliminary comments
=====================================================

In the HOD, the galaxy population is decomposed into distinct types: 
*centrals* and *satellites*. Models for the properties of these populations 
are typically very different, and so Halotools requires you to use 
independently defined component models for these two sub-populations. 
As described below, this is accomplished by your choices for the 
keyword arguments you supply to the `HodModelFactory`. 

As sketched in :ref:`factory_design_diagram`, 
to build any composite model you pass a 
set of component model instances to the `HodModelFactory`. 
These instances are all passed in the form of keyword arguments. 
The keywords you choose must be formatted in a specific way as these strings 
contain information that the factory uses to interpret each feature. 
In particular, the strings you use for keywords will be parsed in a way 
that informs the `HodModelFactory` of the name of galaxy type 
(e.g., 'centrals' or 'satellites'), as well as the kind of feature 
supplied by the component model. 

For the sake of concreteness, let's now dive in and 
write down the source code for how to call the `HodModelFactory`. 

.. _hod_factory_tutorial_zheng07_source_code:

Source code for the ``zheng07`` model
=====================================================

.. code:: python

	from halotools.empirical_models import HodModelFactory

	from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
	cens_occ_model =  Zheng07Cens()
	cens_prof_model = TrivialPhaseSpace()

	from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
	sats_occ_model =  Zheng07Sats()
	sats_prof_model = NFWPhaseSpace()

	model_instance = HodModelFactory(
		centrals_occupation = cens_occ_model, 
		centrals_profile = cens_prof_model, 
		satellites_occupation = sats_occ_model, 
		satellites_profile = sats_prof_model)

	# The model_instance is a composite model 
	# All composite models can directly populate N-body simulations 
	# with mock galaxy catalogs using the populate_mock method:

    from halotools.sim_manager import FakeSim
    halocat = FakeSim()
	model_instance.populate_mock(halocat)

	# Setting simname to 'fake' populates a mock into a fake halo catalog 
	# that is generated on-the-fly, but you can use the populate_mock 
	# method with any Halotools-formatted catalog 

Unpacking the ``zheng07`` source code
=====================================================

First notice how the `HodModelFactory` was instantiated 
with a set of keyword arguments. 
Each of the strings we used for the keys were formatted in a specific way:
the name of the galaxy type, followed by an underscore, 
followed by a nickname for the feature being passed in. 
From the set of keywords alone, the `HodModelFactory` learns  
1. there are two galaxy populations: ``centrals`` and ``satellites``, 
and 2. the list features of each of these galaxy populations are 
``occupation`` and ``profile``. More about this in the next section. 

Next notice what value we bound to each keyword. In all cases, 
we passed in a *component model instance*. These particular component models 
were defined by Halotools, but as we will see in more complex examples, 
you can also include component models that are entirely of your devising. 

We will not cover any of the details about the component model 
features collected together by the ``zheng07`` composite model. 
You can read about that in the :ref:`zheng07_composite_model` 
section of the documentation, 
and also by reading the docstring of each component model class. 

Required features of any HOD-style composite model
=====================================================

As mentioned above and elsewhere in the documentation, there is no 
limitation on the number of features you can include in a composite model. 
However, for any HOD-style model the ``occupation`` and ``profile`` features 
are compulsory for each galaxy type in your model. 

``occupation`` component 
--------------------------

The `HodModelFactory` needs to know ``occupation`` in order to create 
the appropriate number of galaxies in each halo. 
As described in :ref:`writing_your_own_hod_occupation_component` and 
in a more complex worked example in a later part of this tutorial, 
all classes used to define an occupation feature must 
sub-class from `OccupationComponent`. 
For the purpose of this example, 
we will stick with `OccupationComponent` sub-classes that already 
appear in the Halotools code base. 

``profile`` component 
--------------------------

The `HodModelFactory` needs ``profile`` in order to know how to 
distribute the galaxies within their parent halo. 
The process for defining your own profile model is described in 
:ref:`writing_your_own_hod_profile_component`. Again, 
for the purpose of this example, 
we will stick with profile models that already 
appear in the Halotools code base. 

The `PrebuiltHodModelFactory` is just syntax candy
=====================================================

If you followed the :ref:`preloaded_models_overview` section of the documentation, 
or if you have just played around a bit with the code, 
you are already familiar with the `PrebuiltHodModelFactory` class. 
The calling signature for this class takes a string that serves as a 
nickname of a model, e.g., ``zheng07``. Under the hood, the 
`PrebuiltHodModelFactory` class has virtually no functionality of its own. 
The only task that the `PrebuiltHodModelFactory` does is to translate a 
string into the appropriate set of arguments 
to pass on to the `HodModelFactory`, that's it. 
So the code that appears in the :ref:`hod_factory_tutorial_zheng07_source_code` 
section above actually produces an identical composite model as the following:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model_instance = PrebuiltHodModelFactory('zheng07')

Final example: source code for the ``leauthaud11`` model
==========================================================

We'll cement our understanding of how to build simple HOD-style models 
by providing just one more example of a bare bones composite model. 
Compare the source code below to the code that appears in the 
:ref:`hod_factory_tutorial_zheng07_source_code` to convince yourself 
that you see the pattern and are ready to move on to building more complex models. 

.. code:: python

	from halotools.empirical_models import HodModelFactory

	from halotools.empirical_models import TrivialPhaseSpace, Leauthaud11Cens
	another_cens_occ_model =  Leauthaud11Cens()
	another_cens_prof_model = TrivialPhaseSpace()

	from halotools.empirical_models import NFWPhaseSpace, Leauthaud11Sats
	another_sats_occ_model =  Leauthaud11Sats()
	another_sats_prof_model = NFWPhaseSpace()

	another_sats_occ_model._suppress_repeated_param_warning = True

	model_instance = HodModelFactory(
		centrals_occupation = another_cens_occ_model, 
		centrals_profile = another_cens_prof_model, 
		satellites_occupation = another_sats_occ_model, 
		satellites_profile = another_sats_prof_model)

The only line in the above code that may be unfamiliar is setting 
``_suppress_repeated_param_warning`` to True. This is not strictly necessary, 
and is only there to prevent Halotools from raising a warning message which 
in this case is harmless. Briefly, the parameter dictionaries of 
`Leauthaud11Cens` and `Leauthaud11Sats` share several parameters in common 
as both models make use of the `Behroozi10SmHm` model. In general, Halotools 
raises a warning if multiple component models have parameters with the exact 
same name as this can lead to buggy behavior if two parameters with the same 
name have different meanings. In this case, it is harmless so we can simply 
``_suppress_repeated_param_warning`` to True and not be bothered with a warning. 
You can read more about this bookkeeping device at the end of the 
:ref:`param_dict_mechanism` section of the :ref:`composite_model_constructor_bookkeeping_mechanisms` 
page of the documentation. 


See the :ref:`leauthaud11_composite_model` section of the documentation 
for more more information about this model. 

This tutorial continues with :ref:`hod_modeling_tutorial2`. 














