.. _halotools_overview:

************************
Halotools Overview
************************

Here we’ll give a qualitative description of the motivation and functionality of Halotools in broad strokes. You can get a more detailed picture of the package from the rest of the :ref:`user-docs`. 

Halotools is a fully open-source project, and is the product of many scientists collaborating across numerous universities. If you are interested in contributing to Halotools, see the :ref:`developer-docs`. 


Studying Cosmological Models of Structure Formation 
======================================================================

The core science aim of Halotools is to provide a generalized platform to build and test models of the formation and evolution of cosmological structure. The Halotools approach is to directly populate simulated dark matter halos with mock galaxies, and then make measurements on each Monte Carlo-realized universe as you would an observed galaxy catalog. In addition to galaxy evolution studies, the same techniques are also relevant for cosmology applications, as Halotools creates the capability to systematically test a wide variety of assumptions made in conventional cosmological likelihood analyses. 

Direct mock-population offers a powerful and flexible way to study the galaxy-halo connection (see :ref:`mock_making_quickstart` for an overview). When building models in this way, you have full access to all the information available about each halo in the simulation, and you are not limited by restrictive-but-common assumptions such as that a halo's total mass is its only physically relevant property. Moreover, after populating a simulation with galaxies, you can then make any measurement on the mock that could be made on a real galaxy catalog, including two-point clustering, galaxy-galaxy lensing, group identification, isolation criteria, etc. Halotools comes with a set of easy-to-use methods to compute each of these mock measurements and more; the efficiency of these computations is highly optimized for the MCMC-type applications the package is designed for.

Halotools comes pre-loaded with many well-studied classes of cosmological structure formation models, including the Halo Occupation Distribution (HOD), the Conditional Luminosity Function (CLF), abundance matching and its many variants, and so forth. Using a range of these techniques, the package also comes with a handful of pre-computed, realistic mock galaxy catalogs. You can learn how to take advantage of these ready-to-use models by reading the :ref:`model_building_quickstart`. Additionally, there is great diversity in the kinds of models that you can build with Halotools using just a few lines of code. By running our likelihood analysis tools against astronomical measurements, you can constrain the parameters of a traditional empirical model, or of a model you build yourself with the generalized platform we provide. 


Analyzing Cosmological Simulations
===================================

Halotools provides fast, easy-to-use Python code to analyze cosmological simulations. There is end-to-end support for downloading publicly available “raw” catalogs of simulated data, reducing the (quite large) catalogs to memory-mapped binary files, managing the cache of simulations, and studying the data with a variety of common analysis techniques. You can learn how to get started with Halotools' simulation analysis tools by reading the :ref:`sim_analysis`. 

Here are a few examples of questions you can use Halotools to explore:

	* What does the velocity structure look like around a Milky Way-type halo? 

	* How does a halo's large-scale environment impact its dark matter accretion rate?

	* What information can we extract about a halo’s assembly history from its present-day internal structure?


