:orphan:

.. _whats_new_v0p7:

*******************************************
What's New in (unreleased) Halotools v0.7?
*******************************************

Halotools ``v0.7`` is currently under development. The latest release is ``v0.6``, which can now be installed with conda or pip. New features currently be developed for future release ``v0.7`` are summarized below. See :ref:`changelog` for details on smaller issues and bug-fixes. See :ref:`whats_new_v0x_history` for full release history information.


New Utility Functions
=====================

Rotations, dot products, and other 3d operations
------------------------------------------------
There are many new functions in `halotools.utils` subpackage related to spatial rotations in three dimensions:

    * `~halotools.utils.elementwise_dot`
    * `~halotools.utils.angles_between_list_of_vectors`
    * `~halotools.utils.rotation_matrices_from_angles`
    * `~halotools.utils.rotation_matrices_from_vectors`
    * `~halotools.utils.rotate_vector_collection`


New Mock Observables
====================

Inertia Tensor calculation
-------------------------------
The pairwise calculation `halotools.mock_observables.inertia_tensor_per_object_3d` computes the inertia tensor of a mass distribution surrounding each point in a sample of galaxies or halos.
