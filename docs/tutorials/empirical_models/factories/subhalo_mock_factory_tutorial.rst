:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _subhalo_mock_factory_tutorial:

****************************************************
Tutorial on the SubhaloMockFactory Class
****************************************************

This section of the documentation provides detailed notes 
on the source code implementation of the `SubhaloMockFactory` class. 
This class is responsible for populating subhalo catalogs with synthetic galaxy populations 
using composite models built with the `SubhaloModelFactory`. 

Outline 
========

We will start in :ref:`basic_syntax_subhalo_mocks` with a high-level overview of the functionality 
of the `SubhaloMockFactory` class. In :ref:`populate_subhalo_mock_convenience_method` we will 
describe the most common way in which the `SubhaloMockFactory` class is used: by calling the 
``populate_mock`` method bound to all Halotools composite models. We provide detailed 
notes on the source code of the mock factory in :ref:`subhalo_mock_algorithm`. 


.. _basic_syntax_subhalo_mocks:

Basic syntax for making subhalo-based mocks
===============================================


.. _populate_subhalo_mock_convenience_method:

The ``populate_mock`` convenience method
=====================================================


.. _subhalo_mock_algorithm:

Algorithm for populating subhalo-based mocks 
================================================



