:orphan:

.. _whats_new_v0p6:

*****************************
What's New in Halotools v0.6?
*****************************

Halotools ``v0.6`` is currently under development. All significant additions/differences between
the current version of the ``master`` branch and the official ``v0.5`` release on pip are summarized below.


New Mock Observables
====================

Radial distances and velocities
-------------------------------
There are new functions in the `~halotools.mock_observables` sub-package that calculate element-wise radial distances and velocities, `~halotools.mock_observables.radial_distance`  and `~halotools.mock_observables.radial_distance_and_velocity`.

API Change
==========

* The `~halotools.mock_observables.pair_counters.npairs_s_mu` function now has a change to the meaning of the ``mu_bins`` input argument. This argument should now be interpreted as the conventional mu=cos(theta_LOS) instead of mu=sin(theta_LOS).
