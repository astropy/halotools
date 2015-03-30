
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
    gal_prof_model = models.gal_prof_factory.GalProfFactory(sat_nickname, halo_prof_model)
    
    satellite_component_dict = {'occupation' : occupation_model, 
                                'profile' : gal_prof_model}

Build a central population
--------------------------

.. code:: python

    cen_nickname = 'cens'
    
    occupation_model = models.hod_components.Kravtsov04Cens(threshold = -19)
    
    halo_prof_model = models.halo_prof_components.TrivialProfile()
    gal_prof_model = models.gal_prof_factory.GalProfFactory(cen_nickname, halo_prof_model)
    
    central_component_dict = {'occupation' : occupation_model, 
                                'profile' : gal_prof_model}

Bundle the populations together into a composite model blueprint
----------------------------------------------------------------

This will serve as the blueprint used by the HodModelFactory to build a
composite model object

.. code:: python

    composite_model_blueprint = {cen_nickname : central_component_dict, 
                            sat_nickname : satellite_component_dict
                            }

Pass the blueprint to the HodModelFactory, which knows what to do with it
-------------------------------------------------------------------------

.. code:: python

    my_model_object = models.HodModelFactory(composite_model_blueprint)

