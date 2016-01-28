.. _hod_modeling_tutorial1:

****************************************************************
Example 1: Building a simple HOD-style model
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes how you can use the 
`HodModelFactory` to build one of the simplest and most widely 
used galaxy-halo models in the literature: the HOD based on 
Zheng et al. (2007), arXiv:0703457. 

A concrete example of a composite HOD model dictionary
---------------------------------------------------------------------------------------------------------------------

With the above picture in mind, let's now look at a specific example of how a composite model dictionary is built. For our example, we’ll use the particularly simple `~halotools.empirical_models.Zheng07` composite model, one of the fully-functional composite models that comes pre-built with Halotools in the `~halotools.empirical_models` sub-package. 

All the pre-built dictionary functions in the `~halotools.empirical_models` sub-package do the same thing: they build a composite model dictionary, pass the dictionary to the relevant factory, and return an instance of a composite model, as diagrammed in :ref:`factory_design_diagram`. To help understand this example, have a look at the source code for the `~halotools.empirical_models.Zheng07` composite model while you read this section of the documentation. 


For definiteness, let’s look at how the Zheng07 composite model is built. The first chunk of code builds a dictionary called `subpopulation_dictionary_centrals`:

.. code:: python

    ### Build subpopulation dictionary for centrals
    subpopulation_dictionary_centrals = {}

    # Build the `occupation` feature
    occupation_feature_centrals = zheng07_components.Zheng07Cens(threshold = threshold, **kwargs)
    subpopulation_dictionary_centrals['occupation'] = occupation_feature_centrals

    # Build the `profile` feature
    profile_feature_centrals = TrivialPhaseSpace(**kwargs)
    subpopulation_dictionary_centrals['profile'] = profile_feature_centrals


The first key added to this subpopulation dictionary is `occupation`, and the value bound to this key is an instance of `~halotools.empirical_models.Zheng07Cens`. Note how the arguments passed to the Zheng07 function are in turn passed on to `~halotools.empirical_models.Zheng07Cens`, allowing you to customize the behavior of the central occupation statistics via the keyword arguments you pass to `~halotools.empirical_models.Zheng07`. The basic behavior that `~halotools.empirical_models.Zheng07Cens` controls is the mean number of central galaxies found in a halo; see the docstring of `~halotools.empirical_models.Zheng07Cens` for specific details of its options and implementation. 

The second key added to `subpopulation_dictionary_centrals` is `profile`, and the value bound to this key is an instance of the `~halotools.empirical_models.TrivialPhaseSpace` class. The behavior of `~halotools.empirical_models.TrivialPhaseSpace` is simple: our population of `centrals` will reside at the exact center of its host halo and will be at rest in the frame of the halo.

In principle, the keys to a subpopulation dictionary can be any string that you like. For the sake of consistency, the Halotools convention for HOD-style model dictionarys is to use the string `occupation` to contain the instructions for the occupation statistics of a given population, and the string `profile` to contain the instructions for modeling the intra-halo phase space distribution.

At this point, we have finished building the subpopulation dictionary for centrals and we move on to satellites. The process is exactly the same as before, only now we build `subpopulation_dictionary_satellits`, and use `~halotools.empirical_models.Zheng07Sats` and `~halotools.empirical_models.NFWPhaseSpace` as our component models:

.. code:: python

    ### Build subpopulation dictionary for satellites
    subpopulation_dictionary_centrals = {}

    # Build the occupation model
    occupation_feature_satellites = zheng07_components.Zheng07Sats(threshold = threshold, **kwargs)
    occupation_feature_satellites._suppress_repeated_param_warning = True
    subpopulation_dictionary_satellites['occupation'] = occupation_feature_satellites

    # Build the profile model
    profile_feature_satellites = NFWPhaseSpace(**kwargs)    
    subpopulation_dictionary_satellites['profile'] = profile_feature_satellites

In a `~halotools.empirical_models.Zheng07` universe, galaxies are either `centrals` or `satellites`, and the only attributes they have are position and velocity. So the above two dictionaries are all we need to build a composite model dictionary. This building process is simple: we just create a new dictionary with one key for `centrals` and another for `satellites`, and bind the subpopulation dictioaries to these keys:

.. code:: python

    ### Compose subpopulation dictionarys together into a composite dictionary
    composite_model_dictionary = {
        'centrals' : subpopulation_dictionary_centrals,
        'satellites' : subpopulation_dictionary_satellites 
        }


The final line of code in the `~halotools.empirical_models.Zheng07` function is to pass this composite model dictionary to the `~halotools.empirical_models.HodModelFactory`, which now has all the information necessary to build an instance of a composite model. 

.. code:: python 

    composite_model = factories.HodModelFactory(composite_model_dictionary)
    return composite_model



















