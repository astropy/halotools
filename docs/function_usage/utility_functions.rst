:orphan:

.. _utility_functions:


.. currentmodule:: halotools.utils


Cross-matching catalogs with a common object ID
========================================================

.. autosummary::

    crossmatch
    unsorting_indices


Calculating quantities for objects grouped into a common halo
===============================================================

.. autosummary::

	group_member_generator
    compute_richness


Generating Monte Carlo realizations
===============================================================

.. autosummary::

    monte_carlo_from_cdf_lookup
    build_cdf_lookup


Matching one distribution to another
===============================================================

.. autosummary::

    distribution_matching_indices
    resample_x_to_match_y
    bijective_distribution_matching

Rotations, dot products, and other operations in 3d space
===============================================================

.. autosummary::

    elementwise_dot
    elementwise_norm
    angles_between_list_of_vectors
    vectors_between_list_of_vectors
    rotation_matrices_from_angles
    rotation_matrices_from_vectors
    rotate_vector_collection

Probabilistic binning
===============================================================

.. autosummary::

    fuzzy_digitize

Estimating two-dimensional PDFs
===============================================================

.. autosummary::

    sliding_conditional_percentile


Satellite orientations and intra-halo positions
===============================================================

.. autosummary::

    rotate_satellite_vectors
    calculate_satellite_radial_vector
    reposition_satellites_from_radial_vectors
