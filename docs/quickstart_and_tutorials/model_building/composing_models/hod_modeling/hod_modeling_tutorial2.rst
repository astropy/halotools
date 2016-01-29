.. _hod_modeling_tutorial2:

****************************************************************
Example 2: An HOD-style model with additional features
****************************************************************

.. currentmodule:: halotools.empirical_models

In this section of the :ref:`hod_modeling_tutorial0`, 
we'll build a composite model that is not part of the Halotools code base 
by composing together a collection of Halotools-provided features. 
Before reading on, be sure you have read and understood 
:ref:`hod_modeling_tutorial1`. 

Overview of the new model
=============================

The model we'll build will be based on the ``leauthaud11`` HOD, 
but we'll add an additional component model that governs whether or not 
our centrals and satellites are quiescent or star-forming. 
The new component model we'll use is `HaloMassInterpolQuenching`. 
In this model, galaxies are assigned a boolean designation as to whether or 
not they are quiescent. Briefly, the way the model works is that you specify 
what the quiescent fraction is at a set of input control points in halo mass, 
and the model interpolates between these control points to calculate the 
quiescent fraction at any mass. See the `HaloMassInterpolQuenching` docstring for details. 

Source code for the new model 
===============================

.. code:: python

	from halotools.empirical_models import HodModelFactory

	from halotools.empirical_models import TrivialPhaseSpace, Leauthaud11Cens
	another_cens_occ_model =  Leauthaud11Cens()
	another_cens_prof_model = TrivialPhaseSpace()

	from halotools.empirical_models import NFWPhaseSpace, Leauthaud11Sats
	another_sats_occ_model =  Leauthaud11Sats()
	another_sats_prof_model = NFWPhaseSpace()

	from halotools.empirical_models import HaloMassInterpolQuenching
	sat_quenching = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9], gal_type = 'satellites')
	cen_quenching = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.95], gal_type = 'centrals')

	model_instance = HodModelFactory(
		centrals_occupation = another_cens_occ_model, 
		centrals_profile = another_cens_prof_model, 
		satellites_occupation = another_sats_occ_model, 
		satellites_profile = another_sats_prof_model, 
		centrals_quenching = cen_quenching, 
		satellites_quenching = sat_quenching
		)




