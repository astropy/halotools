:orphan:

.. _halotools_science_overview:

***************************
Halotools Science Overview
***************************

In this section of the documentation we give a qualitative description of the motivation and functionality of Halotools in broad strokes. You can get a more detailed picture of the package by navigating to :ref:`getting_started`. 

Halotools is a fully open-source project, and is the product of collaboration by many scientists from numerous universities. If you are interested in contributing to Halotools, and/or in learning more about how the package works under the hood, see the :ref:`tutorials_by_category`. 

Core Science Aim
=====================

The core science aim of Halotools is to provide a generalized platform for building and testing models of the galaxy-halo connection. Halotools achieves this via a standardized interface for generating mock galaxy populations. The interface has been built with the following considerations:

	**Simplicity:** Building a model, generating a synthetic galaxy population, and making mock observations can be accomplished in just a few lines of easy-to-read python code. 

	**Modularity:** Model components can easily be swapped in and out, so complex models can be constructed by composing a collection of simple features written in independent modules.  

	**Performance:** Behind the interface, the back-end is heavily optimized with MCMC-type applications in mind; generating model predictions such as galaxy clustering and lensing typically only takes a few seconds on a modern laptop. 

For convenience, Halotools comes pre-loaded with traditional models such as the HOD, the CLF, and the many variants of abundance matching. The package also includes numerous new classes of previously-unexplored models, and a range of flexible templates for building and testing models based on your own ideas about galaxies. In the sections below we elaborate on the three branches of astrophysics that Halotools is designed to study.

Cosmology 
=====================

The Halotools approach is to directly populate simulated dark matter halos with mock galaxies, and then make measurements on each Monte Carlo-realized universe as you would on an observed galaxy catalog. Direct mock population offers a powerful way to expand the set of cosmological observables deep into the nonlinear regime, while still maintaining the rigor of the precision-cosmology program. This approach offers three distinct advantages:

	**New observables:** Freed from the restrictions of calibrated fitting functions, direct mock population permits the study of any statistic that can be computed from a mock, such as marked correlations, void probabilty functions, and group-based statistics. 

	**Model sophistication:** With Halotools, your models are not limited by restrictive-but-common assumptions such as that a halo's total mass is its only physically relevant property. You can connect galaxies to halos in whatever manner you wish. 

	**Systematic rigor:** Because Halotools uses simulations *directly,* by comparing parameter inferences derived from fitting functions you can rigorously test the assumptions of traditional methods and quantify systematics in terms of your science target of interest. 


Galaxy Evolution 
=====================

Historically, it has been challenging to form clear connections between the predictions made by 1. hydrodynamical simulations, 2. traditional semi-analytical models, and 3. cosmological models such as the HOD. One of the chief goals of Halotools is to provide a bridge between these and other complementary approaches to modeling galaxy evolution.

Direct mock population is the linchpin of this program. With Halotools, synthetic realizations of cosmological models can be built directly into any simulated box, permitting both statistical and halo-by-halo comparisons. By successively introducing model features in such comparisons, it becomes possible to ask very targeted questions about how a feature of one galaxy evolution model manifests in the language of another. We will expand on various aspects of this program throughout the Halotools documentation. 


Structure Formation
==========================================

Halotools provides fast, easy-to-use python code to analyze cosmological simulations. There is end-to-end support for downloading publicly available “raw” catalogs of simulated data, reducing the (quite large) catalogs to memory-mapped binary files, managing the cache of simulations, and studying the data with a variety of common analysis techniques. 

Here are a few examples of questions about cosmological structure formation that you can explore using Halotools:

	* What are the infall and orbital histories of satellites of a Milky Way-type halo? 

	* How does a halo's large-scale environment impact its dark matter accretion rate?

	* What information about a halo’s assembly history is contained in its present-day internal structure?


