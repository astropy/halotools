
.. _custom_hod_model_building_tutorial:

***********************************************
Building a Customized HOD-style Model and Mock
***********************************************

In this tutorial, we'll cover how to design your own HOD-style model of
the galaxy-halo connection. We'll start out with a relatively simple
model to illustrate the basic nuts and bolts. Then we'll build two
successively more complicated example models. After completing this
tutorial, you will know how to take full advantage of the flexibility
offered by Halotools to construct quite complex HOD models with only a
few lines of code.

Once you have learned how to build a model, you can learn how to study
its observational predictions in **THESE DOCS.**

To read about other styles of models you can build with Halotools, see
**THESE DOCS**.

.. code:: python

    from halotools import empirical_models as models

Build a satellite population
----------------------------

.. code:: python

    sat_nickname = 'sats'
    
    occupation_model = models.hod_components.Kravtsov04Sats(threshold = -19)
    
    halo_prof_model = models.halo_prof_components.NFWProfile()
    
    satellite_component_dict = {'occupation' : occupation_model, 
                                'profile' : halo_prof_model}

Build a central population
--------------------------

.. code:: python

    cen_nickname = 'cens'
    
    occupation_model = models.hod_components.Kravtsov04Cens(threshold = -19)
    
    halo_prof_model = models.halo_prof_components.TrivialProfile()
    
    central_component_dict = {'occupation' : occupation_model, 
                                'profile' : halo_prof_model}

Bundle the populations together into a composite model blueprint
----------------------------------------------------------------

This will serve as the blueprint used by the HodModelFactory to build a
composite model object

.. code:: python

    composite_model_blueprint = {cen_nickname : central_component_dict, 
                            sat_nickname : satellite_component_dict
                            }

Pass the blueprint to the Model Factory, which knows what to do
---------------------------------------------------------------

.. code:: python

    my_model = models.HodModelFactory(composite_model_blueprint)

Now that you have built a model, it's easy to use it to rapidly generate
a mock galaxy population. Whether you've built a very simple, or very
complex mock, the above and below syntax is always the same:

.. code:: python

    my_model.populate_mock()

Let's take a quick look at what we've got:

.. code:: python

    print(my_model.mock.galaxy_table[0:5])


.. parsed-literal::

    halo_haloid          halo_pos [3]          ... gal_NFWmodel_conc gal_type
    ----------- ------------------------------ ... ----------------- --------
     3060299659 35.7249908447 .. 17.7129898071 ...     6.45777867233     cens
     3060313505  45.2089195251 .. 39.911239624 ...     6.47874642155     cens
     3058441127 21.8120098114 .. 9.54759025574 ...     6.68856074462     cens
     3058442008 26.1803398132 .. 6.51834011078 ...     6.79585452177     cens
     3058452897 1.74397003651 .. 17.8251895905 ...     6.88196980011     cens


