""" Module containing helper functions used to process the arguments
passed to functions in the `~halotools.mock_observables.radial_profiles` sub-package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = ('bounds_check_sample2_quantity', )


def bounds_check_sample2_quantity(sample2, sample2_quantity):
    """ Function enforces that input ``sample2_quantity`` has the appropriate shape.
    """
    npts2 = sample2.shape[0]
    sample2_quantity = np.atleast_1d(sample2_quantity).astype('f8')
    npts_quantity2 = len(sample2_quantity)
    try:
        assert npts_quantity2 == npts2
    except AssertionError:
        msg = ("Input ``sample2_quantity`` has %i elements, "
            "but input ``sample2`` has %i elements.\n" % (npts_quantity2, npts2))
        raise ValueError(msg)
    return sample2_quantity


def get_normalized_rbins(rbins_absolute, rbins_normalized, normalize_rbins_by, sample1):
    """ Function processes the various options for the input rbins arguments and returns
    Numpy arrays ``rbins_normalized`` and ``normalize_rbins_by``.

    If ``rbins_absolute`` is provided, it is enforced that neither
    ``rbins_normalized`` nor ``normalize_rbins_by`` is provided.

    If ``rbins_normalized`` is provided, it is enforced that ``normalize_rbins_by`` must
    be provided, and that ``rbins_absolute`` is not provided.

    The returned value of ``normalize_rbins_by`` is enforced to have the same length
    as the input ``sample1``, and if the input ``normalize_rbins_by`` argument is None,
    an array of ones is returned.
    """
    if (rbins_absolute is None) & (rbins_normalized is None):
        msg = ("You must either provide a ``rbins_absolute`` argument \n"
            "or a ``rbins_normalized`` argument.\n")
        raise ValueError(msg)
    elif rbins_normalized is None:
        if normalize_rbins_by is not None:
            msg = ("If you provide the ``rbins_absolute`` argument, \n"
                "you should not provide the ``normalize_rbins_by`` argument.\n")
            raise ValueError(msg)
        else:
            rbins_normalized = np.atleast_1d(rbins_absolute)
            normalize_rbins_by = np.ones(len(sample1))
    elif rbins_absolute is None:
        if normalize_rbins_by is None:
            msg = ("If you provide the ``rbins_normalized`` argument, \n"
                "you must also provide the ``normalize_rbins_by`` argument.\n")
            raise ValueError(msg)
        else:
            rbins_normalized = np.atleast_1d(rbins_normalized)
            normalize_rbins_by = np.atleast_1d(normalize_rbins_by)
            try:
                assert len(normalize_rbins_by) == len(sample1)
            except AssertionError:
                msg = ("Your input ``normalize_rbins_by`` must have the same number of elements \n"
                    "as the number of points in ``sample1``.\n")
                raise ValueError(msg)
    else:
        msg = ("Do not provide both ``rbins_normalized`` and ``rbins_absolute`` arguments.")
        raise ValueError(msg)

    try:
        assert np.all(normalize_rbins_by > 0)
    except AssertionError:
        msg = ("Input ``normalize_rbins_by`` must be strictly positive.")
        raise ValueError(msg)

    try:
        assert np.all(rbins_normalized >= 0)
    except AssertionError:
        msg = ("Input ``rbins_normalized`` must be strictly positive.")
        raise ValueError(msg)

    return rbins_normalized, normalize_rbins_by


def enforce_maximum_search_length_3d(rbins_normalized, normalize_rbins_by, period):
    """ Require that the input rbins does not result in an attempt to count pairs
    over distances exceeding period/3 in any dimension.
    """
    max_r_max = np.amax(rbins_normalized)*np.amax(normalize_rbins_by)
    try:
        assert np.all(max_r_max < period/3.)
    except AssertionError:
        msg = ("Your choice for the input rbins implies that you are attempting to \n"
            "count pairs of points over a search radius of %.2f.\n"
            "This exceeds the maximum permitted search length of period/3.\n"
            "If you really need to count pairs over distances this large, \n"
            "you should be using a larger simulation.")
        raise ValueError(msg % max_r_max)
