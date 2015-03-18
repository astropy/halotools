************************
Halotools Overview
************************

Here we’ll give a qualitative description of the motivation and functionality of Halotools in broad strokes. You can get a more detailed picture of the package from the rest of the documentation. 

Studying Models of Galaxy Evolution
===================================

The core science aim of Halotools is to provide a generalized platform to build and test models of the formation and evolution of cosmological structure. The way you can accomplish this with the code is to directly populate simulated dark matter halos with mock galaxies. You can then make measurements on each Monte Carlo-realized universe as you would an observed galaxy catalog. In addition to galaxy evolution studies, the same techniques are also relevant for cosmology applications, as Halotools creates the capability to systematically test a wide variety of assumptions made in conventional cosmological likelihood analyses. 

Direct mock population offers a powerful and flexible way to study the galaxy-halo connection. When building models in this way, you have full access to all the information available about each halo in the simulation, and you are not limited by restrictive-but-common assumptions about a halo, such as that a halo's total mass is its only physically relevant property. Moreover, after populating a simulation with galaxies you can then make any measurement on the mock that could be made on a real galaxy catalog, including two-point clustering, galaxy-galaxy lensing, group identification, isolation criteria, etc. Halotools comes with a set of easy-to-use methods to compute each of these mock measurements and more; the methods are highly optimized for the MCMC-type applications it is designed for.

Halotools comes pre-loaded with many well-studied classes of empirical models in the literature, including the Halo Occupation Distribution (HOD), the Conditional Luminosity Function (CLF), abundance matching and its many variants, and so forth. Using a range of these techniques, the package comes with a handful of pre-computed, realistic mock galaxy catalogs. Additionally, there is great flexibility in the kinds of models that you can build with Halotools using just a few lines of code. You can also use our likelihood analysis sub-package to utilize a new or existing dataset to constrain the parameters of a traditional empirical model, or one you build yourself with the generalized platform we provide. 


Analyzing Cosmological Simulations
===================================

Halotools provides fast, easy-to-use Python code to analyze cosmological simulations of structure formation. There is end-to-end support for downloading widely used “raw” catalogs of simulated data, reducing the (quite large) catalogs to memory-mapped binary files, managing the cache of simulations, and studying the data with a variety of common analysis techniques. Here are a few examples questions you can use Halotools to explore:

	* What does the velocity structure look like around a Milky Way-type halo? 

	* How does large-scale environment impact dark matter accretion rate?

	* What information can we extract about a halo’s assembly history from its present-day internal structure?


