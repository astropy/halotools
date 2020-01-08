:orphan:

.. _whats_new_v0p7:

*******************************************
What's New in (unreleased) Halotools v0.7?
*******************************************

Halotools ``v0.7`` is currently under development. The latest release is ``v0.6``, which can now be installed with conda or pip. New features currently be developed for future release ``v0.7`` are summarized below. See :ref:`changelog` for details on smaller issues and bug-fixes. See :ref:`whats_new_v0x_history` for full release history information.


New Utility Functions
=====================

Probabilistic binning
------------------------------------------------
The `~halotools.utils.fuzzy_digitize` function in `halotools.utils` allows you to discretize an
array in a probabilistic fashion, which can be useful for applications of conditional abundance matching.

Estimation of Conditional Probability Distributions
-----------------------------------------------------
The `~halotools.utils.sliding_conditional_percentile` function in `halotools.utils` calculates Prob(< y | x) for any arbitrary distribution of two-dimensional data. This function can be used to estimate, for example, quantiles of galaxy size as a function of stellar mass, and also should be useful in applications of conditional abundance matching.


New Mock Observables
====================

Inertia Tensor calculation
-------------------------------
The pairwise calculation `~halotools.mock_observables.inertia_tensor_per_object` computes the inertia tensor of a mass distribution surrounding each point in a sample of galaxies or halos.

API Changes
===========

* The old implementation of the `~halotools.empirical_models.conditional_abunmatch` function has been renamed to be `~halotools.empirical_models.conditional_abunmatch_bin_based`.

* There is an entirely distinct, bin-free implementation of Conditional Abundance Matching that now bears the name `~halotools.empirical_models.conditional_abunmatch`.
