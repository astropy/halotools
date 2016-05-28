# -*- coding: utf-8 -*-
"""
Module containing the HOD-style composite model
published in Zheng et al. (2007), arXiv:0703457.
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

from ... import model_defaults
from ...occupation_models import zheng07_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

from ....sim_manager import sim_defaults


__all__ = ['zheng07_model_dictionary']


def zheng07_model_dictionary(
        threshold=model_defaults.default_luminosity_threshold,
        redshift=sim_defaults.default_redshift, **kwargs):
    """ Dictionary for an HOD-style based on Zheng et al. (2007), arXiv:0703457.

    See :ref:`zheng07_composite_model` for a tutorial on this model.

    There are two populations, centrals and satellites.
    Central occupation statistics are given by a nearest integer distribution
    with first moment given by an ``erf`` function; the class governing this
    behavior is `~halotools.empirical_models.occupation_components.Zheng07Cens`.
    Central galaxies are assumed to reside at the exact center of the host halo;
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

    Satellite occupation statistics are given by a Poisson distribution
    with first moment given by a power law that has been truncated at the low-mass end;
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Zheng07Sats`;
    satellites in this model follow an (unbiased) NFW profile, as governed by the
    `~halotools.empirical_models.NFWPhaseSpace` class.

    This composite model is built by the `~halotools.empirical_models.HodModelFactory`.

    Parameters
    ----------
    threshold : float, optional
        Luminosity threshold of the galaxy sample being modeled.
        Default is set in the `~halotools.empirical_models.model_defaults` module.

    redshift : float, optional
        Redshift of the galaxy population being modeled.
        If you will be using the model instance to populate mock catalogs,
        you must choose a redshift that is consistent with the halo catalog.
        Default is set in the `~halotools.empirical_models.model_defaults` module.

    Returns
    -------
    model_dictionary : dict
        Dictionary of keywords to be passed to
        `~halotools.empirical_models.HodModelFactory`

    Examples
    --------
    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('zheng07', threshold = -21)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """

    ####################################
    # Build the `occupation` feature
    centrals_occupation = zheng07_components.Zheng07Cens(
        threshold=threshold, redshift=redshift)

    # Build the `profile` feature
    centrals_profile = TrivialPhaseSpace(redshift=redshift, **kwargs)

    ####################################
    # Build the occupation model
    satellites_occupation = zheng07_components.Zheng07Sats(
        threshold=threshold, redshift=redshift, **kwargs)
    satellites_occupation._suppress_repeated_param_warning = True

    # Build the profile model
    satellites_profile = NFWPhaseSpace(redshift=redshift, **kwargs)

    return ({'centrals_occupation': centrals_occupation,
        'centrals_profile': centrals_profile,
        'satellites_occupation': satellites_occupation,
        'satellites_profile': satellites_profile})
