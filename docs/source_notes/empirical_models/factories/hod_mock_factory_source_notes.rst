:orphan:

.. currentmodule:: halotools.empirical_models

.. _hod_mock_factory_source_notes:

********************************************************************
Source code notes on `HodMockFactory` 
********************************************************************

This section of the documentation provides detailed notes 
for how the `HodMockFactory` populates halo catalogs with synthetic galaxy populations. 
The `HodMockFactory` uses composite models built with the `HodModelFactory`, which 
is documented in the :ref:`hod_model_factory_source_notes`. 

.. _galprops_assigned_before_mc_occupation: 

Galaxy properties assigned prior to calling the occupation components 
========================================================================


.. _determining_the_gal_type_slice:

Determining the appropriate gal_type slice
========================================================================

This section of the tutorial is referenced by :ref:`hod_model_factory_source_notes` 
and explains the following mechanism. 

setattr(getattr(self, new_method_name), 'gal_type', gal_type) # line 4
setattr(getattr(self, new_method_name), 'feature_name', feature_name) # line 5
