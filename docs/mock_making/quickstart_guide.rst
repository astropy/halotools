.. _mock_making_quickstart:

********************************
Quickstart Guide to Making Mocks
********************************

The simplest way to build a mock galaxy population 
is to just use the default pre-loaded model, 
and call its populate method. 

>>> from halotools.empirical_models import preloaded_models
>>> hod_model = preloaded_models.Kravtsov04()

Halotools installs to use abundance matching 
as the default model, and the redshift-zero 
snapshot of Bolshoi as the default simulation. 
You can configure your choice for the default 
mock in defaults.py. Just set the values of 
defaults.default_simulation to your preferred 
snapshot; you can read which simulations come 
packaged with halotools here :ref:`simulation_list`. 
The options for defaults.default_model 
are those listed at :ref:`list_of_default_models`






