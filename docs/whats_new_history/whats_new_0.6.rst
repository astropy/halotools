:orphan:

.. _whats_new_v0p6:

*****************************
What's New in Halotools v0.6?
*****************************

Halotools ``v0.6`` is now available for installation with conda and pip. New features are summarized below. See :ref:`changelog` for details on smaller issues and bug-fixes. See :ref:`whats_new_v0x_history` for full release history information.

Quick Installation Verification
===============================

It is now much faster to verify successful installation of the latest version of halotools. After installing the code, you can now run a targeted subset of unit tests, rather than the entire test suite:

.. code:: python

    import halotools
    halotools.test_installation()


New Mock Observables
====================

Radial distances and velocities
-------------------------------
There are new functions in the `~halotools.mock_observables` sub-package that calculate element-wise radial distances and velocities, `~halotools.mock_observables.radial_distance`  and `~halotools.mock_observables.radial_distance_and_velocity`.

Jackknife error estimation
---------------------------
New functions `~halotools.mock_observables.wp_jackknife` and `~halotools.mock_observables.rp_pi_tpcf_jackknife` give jackknife error estimation for the `~halotools.mock_observables.wp` and `~halotools.mock_observables.rp_pi_tpcf` functions.

Weighted pair counts as a function of (s, mu)
---------------------------------------------
New `halotools.mock_observables.pair_counters.weighted_npairs_s_mu` function gives weighting option for counting pairs in (s, mu) space.

Utility functions
==================

Matching two distributions for statistical comparison
------------------------------------------------------
The `halotools.utils.distribution_matching_indices` function Monte Carlo resamples one distribution until it matches another.


API Change
==========

* The `~halotools.mock_observables.pair_counters.npairs_s_mu` function now has a change to the meaning of the ``mu_bins`` input argument. This argument should now be interpreted as the conventional mu=cos(theta_LOS) instead of mu=sin(theta_LOS).
