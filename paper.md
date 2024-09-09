---
title: 'Halotools: A New Release Adding Intrinsic Alignments to Halo Based Methods'
tags:
    - Python
    - Cosmology
    - Weak Gravitational Lensing
    - Intrinsic Alignments
authors:
    - name: Nicholas Van Alfen
      orcid: 0000-0003-0049-2861
      corresponding: true
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
    - name: Jonathan Blazek
      orcid: 0000-0002-4687-4657
      equil-contrib: true
      affiliation: 1
affiliations:
    - name: Department of Physics, Northeastern University, Boston, MA 02115, USA
      index: 1
    - name: McWilliams Center for Cosmology, Department of Physics, Carnegie Mellon University, Pittsburgh, PA 15213, USA
      index: 2
    - name: Argonne National Laboratory, Lemont, IL 60439, USA
      index: 3
date: 21 August 2024
bibliography: paper.bib
---

# Summary
`Halotools`, initially published in 2017, is a Python package for cosmology and astrophysics designed to generate mock universes using existing catalogs of dark matter halos [@Hearin_2017]. The halo model describes the matter distribution of dark matter as gravitationally self-bound clouds of dark matter particles that we call halos. `Halotools` was designed to take an underlying catalog of dark matter halos and populate them with galaxies using subhalo abundance, or halo occupation distribution (HOD) models, creating catalogs of simulated galaxies for use in research. This release (v0.9.0) adds functionality to align galaxies, injecting what are known as intrinsic alignments (IA) into these catalogs. As such, these simulated galaxy catalogs can now be created with realistically complex correlations between galaxies, mimicking some effects seen in more expensive hydrodynamic simulations.

`Halotools` is a fast and flexible package that can create a wide range of galaxy catalogs for use in validating existing IA mitigation models. One of the main advantages to using `halotools` over other methods is that `halotools` is lightweight and modular, able to run on a personal laptop with no need for heavy computing resources.

# Statement of Need
Following the halo model, galaxies form within dark matter halos, and the intrinsic shapes and orientations of these galaxies may be related to those of the host halo and with the large-scale structure of the universe (e.g. the local gravitational tidal field), an effect known as intrinsic alignments (IA) [see, e.g., @Hirata_2004; @Blazek_2019].

The observed shapes and orientations also have a contribution from weak gravitational lensing, the measurement of which is a pillar of modern observational cosmology [e.g. @kids_2020; @des_keypaper_2022; @hsc_2023].

IA can thus become an important systematic effect to weak lensing measurements, and it must be properly understood and mitigated to ensure accurate cosmological results [e.g. @Krause_2016; @samuroff_2017; @secco_2021].

Measurements of weak lensing shear
help researchers probe the distribution of matter and dark energy. The large-scale structure of the universe can influence the intrinsic shapes and orientations of galaxies through gravitational interactions. As such, accurately modeling this effect is important for precision cosmology with weak lensing. With upcoming surveys like the Rubin Observatory Legacy Survey of Space and Time (LSST) [@Ivezic_2019], analyses of the data will need to consider contributions from IA.

A fast and flexible simulation method that includes IA is required to to provide realistic mock galaxy catalogs and to test other IA models.

Understanding and measuring IA also provides a window into the accurate modeling of galaxy formation and a probe of cosmic structure and potentially new physics [e.g. @chisari_2013]. `Halotools` already provides tools for modeling the relationship between galaxies and the halos in which they reside (the galaxy--halo connection), and it is widely used in the field. The expanded functionality added in this release allows for the possibility of using halotools to produce mock galaxy catalogs with realistically complex galaxy orientations. These catalogs can then be used to test and validate IA models, to study IA in observational data and in hydrodynamic simulations [e.g. @nelson_2017; @pillepich_2017; @naiman_2017; @springel_2017; @marinacci_2017], and to provide a fully nonlinear, simulation-based model for observed galaxy clustering and lensing statistics.

# Significance
`Halotools` provides a way for users to create halo occupation models such as abundance matching and the halo occupation distribution (HOD), and enables a modular approach to mock universe creation.

The user can provide a series of component models to the HOD model describing features that will govern how `halotools` populates these dark matter halos with galaxies, generating a catalog that can be used for modeling. This release provides a simple way to include component models to describe galaxy alignment, including IA similarly to how other features more typical of HOD models are defined.

The new release of `halotools` creates the capability to construct realistically complex IA correlations--comparable to those of a hydrodynamic simulation--at a tiny fraction of the computational cost of a hydrodynamic simulation, as explained in @vanalfen_2024. This flexibility expands `halotools` to be of considerable benefit to simulation-based studies of IA. In @vanalfen_2024, the authors demonstrated the flexibility of the `halotools` package to create galaxy catalogs with IA comparable to various aspects of high-resolution cosmological simulations. Specifically, Figure \ref{IllustrisComparison} [taken from Figure 12 in @vanalfen_2024] shows various IA correlation functions from both IllustrisTNG300-1 [@nelson_2017; @pillepich_2017; @naiman_2017; @springel_2017; @marinacci_2017] and a galaxy catalog generated using `halotools` with its available Bolshoi-Planck (Bolplanck) halo catalog [@Klypin_2011].

![Figure 12 from @vanalfen_2024 showing correlation functions from IllustrisTNG300 (points with error bars) and correlation functions measured on an HOD made with `halotools` (solid lines with shaded error regions). The parameters for the HOD model were adjusted such that the resulting correlations would match those of IllustrisTNG300, showcasing the flexibility of the model.\label{IllustrisComparison}](figures/Illustris.pdf)

This release is part of a suite of modeling tools and analysis pipelines being developed to aid upcoming cosmological surveys, including LSST, Euclid, and Roman. The specific advantage of the type of model `halotools` generates is that they are faster and lighter-weight than more expensive simulations, allowing users to quickly generate and populate catalogs of galaxies following a set of parameters. The efficiency of `halotools` also allows for direct simulation-based modeling.

# Structure
Currently, to build a mock galaxy catalog using `halotools` with IA, the user needs to provide one of each of the following (with optional components in parentheses):

\begin{itemize}
    \item Occupation Model: Determines the number density of galaxies within a given halo.
    \item Phase Space Model: Determines the location and velocity of a galaxy within its halo.
    \item Alignment Model: The focus of this release. Determines the orientation of the galaxy by aligning with respect to some reference vector (halo major axis, radial vector to center of halo, etc.) according to the alignment strength, a parameter that can either be set globally or vary between objects.
    \item (Alignment Strength Model: Optional component added in this release. Allows each galaxy its own alignment strength based on individual properties (e.g. distance from center of its host halo) rather than assigning all galaxies a single alignment strength.)
\end{itemize}

# Future Work
In the current iteration of IA tools available through `halotools`, we only consider orientation, rather than full shape information. Plans for future work include extending the functionality of the package to incorporate distributions of three-dimensional shapes. We also plan to expand the available alignment models and to allow for more complex determinations of alignment strength, such as assigning each galaxy an alignment strength based on redshift, color, luminosity, mass, etc.

# Acknowledgements
This work was supported in part by NASA under the OpenUniverse effort (JPL Contract Task 70-711320) and the Roman Research and Support Participation program (grant 80NSSC24K0088), by NSF Award AST-2206563, and by the DOE Office of Science under award DE-SC0024787 and the SCGSR program, administered by ORISE which is managed by ORAU under contract number DE-SC0014664. Work done at Argonne was supported under the DOE contract DE-AC02-06CH11357.

# References
