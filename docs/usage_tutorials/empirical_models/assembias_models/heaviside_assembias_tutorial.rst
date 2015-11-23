:orphan:

.. _heaviside_assembias_tutorial:

*********************************************
Tutorial on the HeavisideAssembias model
*********************************************

This tutorial gives a detailed explanation of the 
`~halotools.empirical_models.HeavisideAssembias` class. 
In the :ref:`example_heaviside_usage` section we'll provide 
a couple of examples of how you can use the orthogonal 
mix-in pattern to create new, assembly-biased models from existing models. 
You can learn how to hand-tailor the features of the 
`~halotools.empirical_models.HeavisideAssembias` class in the section on 
:ref:`heaviside_feature_customization`. To understand the underlying 
python implementation, see :ref:`heaviside_assembias_source_code_notes`. 


.. _example_heaviside_usage: 

Example Usage
==============

Assembly-biased occupation statistics 
---------------------------------------

Suppose you have a component model whose behavior you'd like to 
decorate with assembly bias. Let's use the 
`~halotools.empirical_models.Zheng07Sats` as a specific example. 
This class governs the halo occupation statistics of satellite galaxies. 
The primary function governing the behavior of this model is 
`~halotools.empirical_models.Zheng07Sats.mean_occupation`, which controls the 
mean number of satellites as a function of halo mass. 
In order to build an assembly-biased version of the 
`~halotools.empirical_models.Zheng07Sats` model, you define a new class as follows:

.. code:: python

    class AssembiasZheng07Sats(Zheng07Sats, HeavisideAssembias):

        def __init__(self, **kwargs):

            Zheng07Sats.__init__(self, **kwargs)

            HeavisideAssembias.__init__(self, 
                method_name_to_decorate = 'mean_occupation', 
                lower_assembias_bound = 0, 
                upper_assembias_bound = np.inf, 
                **kwargs)

That's it. You now have a new model, `AssembiasZheng07Sats`, that is identical 
in every respect to the `~halotools.empirical_models.Zheng07Sats` model, except now 
your new model includes the effects of assembly bias. Thus to define your new class, 
you simply sub-class from the baseline model, 
and then use the orthogonal mix-in design pattern above to sub-class from the 
`~halotools.empirical_models.HeavisideAssembias` class. Inside the constructor 
of your new class, you need to call the constructor of the 
`~halotools.empirical_models.HeavisideAssembias` class with three arguments:

    1. name of the baseline method being decorated with assembly bias, and 
    2. lower and upper bounds returned by the baseline method

There are a large number of additional keyword arguments to the 
`~halotools.empirical_models.HeavisideAssembias` class that allow you to customize 
the behavior of instances of your new model; you can read about these options in 
the documentation below. For now, just note that the above pattern can be matched 
for *any* method in *any* Halotools model (e.g., occupation statistics, quenched fractions, 
radial distributions of satellites, etc.). 
You can also use `~halotools.empirical_models.HeavisideAssembias` to decorate baseline models 
that you write yourself, not just component models that are already in Haltools. 
The `~halotools.empirical_models.HeavisideAssembias` class is a very general tool 
for studying how assembly bias influences large-scale structure. 

Assembly-biased quiescent fractions 
-------------------------------------

Now let's look at one more example of how 
the `~halotools.empirical_models.HeavisideAssembias` class can be used to decorate 
some underlying behavior with assembly bias. In this next example, we'll work with 
the `~halotools.empirical_models.Tinker13Cens` as our baseline class. 
This is an HOD-style model for central galaxy abundance as a function of halo mass, 
where the occupation statistics are different for quiescent/star-forming centrals. 
The star formation rate designation is controlled by the 
`~halotools.empirical_models.Tinker13Cens.mean_quiescent_fraction` method, which 
is naturally bounded by zero and unity. Matching the pattern above allows us to 
construct a new model in which the SFR-designation has assembly bias:


.. code:: python

    class AssembiasTinker13Cens(Tinker13Cens, HeavisideAssembias):

        def __init__(self, **kwargs):

            Tinker13Cens.__init__(self, **kwargs)
        
            HeavisideAssembias.__init__(self, 
                method_name_to_decorate = 'mean_quiescent_fraction', 
                lower_assembias_bound = 0., 
                upper_assembias_bound = 1., 
                **kwargs)


.. _heaviside_feature_customization: 

Customizing the behavior of an assembly-biased model 
=======================================================



