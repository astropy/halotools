---
title: 'halotools: A New Release Adding Intrinsic Alignments to Halo Based Methods'
tags:
    - Python
    - Cosmology
    - Weak Gravitational Lensing
    - Intrinsic Alignments
authors:
    - name: Nicholas Van Alfen
      orcid: 0000-0003-0049-2861
      equal-contrib: true
      affiliation: 1
    - name: Duncan Campbell
      orcid: 0000-0002-0650-9903
      equal-contrib: true
      affiliation: 2
    - name: Andrew Hearin
      orcid: 0000-0003-2219-6852
      equal-contrib: true
      affiliation: 3
affiliations:
    - name: Department of Physics, Northeastern University, Boston, MA 02115, USA
      index: 1
    - name: McWilliams Center for Cosmology, Department of Physics, Carnegie Mellon University, Pittsburgh, PA 15213, USA
      index: 2
    - name: Argonne National Laboratory, Lemont, IL 60439, USA
      index: 3
date: 14 November 2023
bibliography: paper.bib
---

# Summary
The halo model describes the matter distribution of dark matter as ellipsoidal clouds of dark matter particles that we call halos. Within these halos, galaxies form and their shape and orientation may be related to those of their host halo as well as with large-scale structure of the universe. When observing galaxies in the universe, we measure a shape and orientation, some of which comes from gravitational lensing, and some of which comes from the intrinsic shape and orientation of the galaxy itself. This intrinsic shape and orientation, known as intrinsic alignment (IA), can contribute to statistical measurements made on dark matter surveys, such as weak lensing, and is an important systematic to account for, as well as being important to the accurate modeling of galaxy formation. `Halotools` already provides tools for modeling the relationship between galaxies and the halos in which they reside (the galaxy-halo connection), and is widely used in the field, but this new release adds the ability to model IA, expanding previous capabilities of the package. This functionality allows for the possibility of using halotools to test and validate existing simulations using IA and provides tools for ongoing research in the field.

# Statement of Need
While `halotools` already provides many features for work involving the galaxy-halo connection, past versions have not considered IA. Among other things, IA becomes an important systematic in weak lensing measurements, which serve as an important probe in dark matter surveys. These weak lensing shear measurements, that is measurements on the statistical correlations of the ellipticity of various light sources (e.g. galaxies), help researchers probe the distribution of matter in a given area as high concentrations of dark matter create a stronger lensing signal. However, the intrinsic shape and orientation of galaxies will of course contribute to this measurement as well.

Galaxies can influence the intrinsic shapes and orientations of other nearby galaxies through gravitational and tidal interactions. The gravitational pull and tidal fields can stretch and turn a given galaxy. This intrinsic alignment will contribute to shear measurements, though this contribution has nothing to do with the gravitational distortion of light and everything to do with the aforementioned intrinsic properties. As such, IA serves as a contaminant in weak lensing surveys and accurately modeling it is important in order to do detailed science in this area. With upcoming surveys like the Legacy Survey of Space and Time (LSST) [@ivezic_2019], analyses of the data will need to consider contributions from IA in order to properly measure the statistics important for understanding dark matter and to provide a way for other models and simulations to validate their results.

# Significance
`Halotools` was initially published in 2017 as a way to generate mock universes using existing catalogs of dark matter halos [@Hearin_2017]. It provides a way for users to create a type of model called a halo occupation distribution (HOD) model, which enables a modular approach to mock universe creation.

The user can provide a series of component models to this HOD model describing features that will govern how `halotools` will populate these dark matter halo catalogs with galaxies, generating a catalog that can be used for modeling. This release provides a simple way to include component models to describe galaxy alignment, including IA in the model in a similar fashion as how other features more typical of HOD models are defined. This modular approach allows a highly flexible method of including and altering how the overal HOD model is built.

This flexibility allows the potential for `halotools` HOD models to be used in place of more expensive hydrodynamic simulations for some use cases, as explained in [@vanalfen_2023]. While these more complex simulations contain important interactions, they take much more time and many more computational resources than `halotools` and HOD models in general. As such, if HOD galaxy catalogs with realistically complex IA can be produced more cheaply and can even contain realistically complex correlations comparable to those seen in hydrodynamic simulations, it will be of great benefit to those wishing to study the effects of IA.

In [@vanalfen_2023], we can see the flexibility of the `halotools` package described here in its ability to be tuned to create galaxy catalogs with IA comparable to more complex simulations. Specifically, Figure \ref{IllustrisComparison} (taken from Figure 11 in [@vanalfen_2023]) shows various IA correlation functions from both IllustrisTNG300-1 [@nelson_2017; @pillepich_2017; @naiman_2017; @springel_2017; @marinacci_2017] and a galaxy catalog generated using `halotools` with its available Bolshoi-Planck (Bolplanck) halo catalog [@Klypin_2011]. As seen in the figure, the `halotools` method is flexible enough that the parameters can be tuned to create realistically complex IA signals comparable to the more expensive hydrodynamic simulations.

![Figure 11 from [@vanalfen_2023] showing correlation functions from IllustrisTNG300 (points with error bars) and correlation functions measured on an HOD made with `halotools` (solid lines with shaded error regions). The parameters for the HOD model were adjusted such that the resulting correlations would match those of IllustrisTNG300, showcasing the flexibility of the model.\label{IllustrisComparison}](figures/Illustris.pdf)

`Halotools` is already widely used by physicists working on galaxy halo connection research and this release was designed to aid upcoming dark matter surveys, specifically LSST mentioned earlier. In preparation for upcoming surveys, a suite of modeling tools and analysis pipelines are being developed, including this new IA release of `halotools`. The specific advantage of the type of model `halotools` generates is that they are faster and lighter-weight than more expensive simulations, allowing users to quickly generate and populate catalogs of galaxies following a set of parameters given without losing desired complexity in the output galaxy catalogs.

# Structure
Currently, to build an mock galaxy catalog using `halotools` with IA, the user needs to provide one of each of the following (with optional components in parentheses):

\begin{itemize}
    \item Occupation Model: Determines the number density of galaxies within a given halo.
    \item Phase Space Model: Determines the location of a galaxy withing its halo, as well as velocities and other properties relative to its halo.
    \item Alignment Model: The focus of this release. Determines the orientation of the galaxy by aligning with respect to some reference vector (halo major axis, radial vector to center of halo, etc.) according the the alignment strength.
    \item (Alignment Strength Model: Optional component. Allows each galaxy its own alignment strength based on individual properties (e.g. distance from center of its host halo) rather than assigning all galaxies a single alignment strength.)
\end{itemize}

# Future Work
In the current iteration of IA tools available through `halotools`, we only consider orientation, rather than full shape information. There are currently plans to extend the functionality of the package to this more complete picture, but the scope is such that we have left this for future work. There are also plans to expand the available alignment models and allow for more complex determinations of alignment strength such as assigning each galaxy an alignment strength based on redshift, color, luminosity, mass, etc.

# References
