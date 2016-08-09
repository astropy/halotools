"""
Module containing the HOD-style composite model based on Leauthaud et al. (2011).
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ... import model_defaults
from ...occupation_models import leauthaud11_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

__all__ = ['leauthaud11_model_dictionary']


def leauthaud11_model_dictionary(threshold=model_defaults.default_stellar_mass_threshold,
        **kwargs):
    """
    Dictionary to build an HOD-style based on Leauthaud et al. (2011), arXiv:1103.2077.
    The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation.

    See :ref:`leauthaud11_composite_model` for a tutorial on this model.

    There are two populations, centrals and satellites.
    Central occupation statistics are given by a nearest integer distribution
    with first moment given by an ``erf`` function; the class governing this
    behavior is `~halotools.empirical_models.occupation_components.Leauthaud11Cens`.
    Central galaxies are assumed to reside at the exact center of the host halo;
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

    Satellite occupation statistics are given by a Poisson distribution
    with first moment given by a power law that has been truncated at the low-mass end;
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Leauthaud11Sats`;
    satellites in this model follow an (unbiased) NFW profile, as governed by the
    `~halotools.empirical_models.NFWPhaseSpace` class.

    This composite model is built by the `~halotools.empirical_models.HodModelFactory`.

    Parameters
    ----------
    threshold : float, optional
        Stellar mass threshold of the mock galaxy sample.
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    concentration_binning : tuple, optional
        Three-element tuple. The first entry will be the minimum
        value of the concentration in the lookup table for the satellite NFW profile,
        the second entry the maximum, the third entry
        the linear spacing of the grid.
        Default is set in `~halotools.empirical_models.model_defaults`.
        If high-precision is not required, the lookup tables will build much faster if
        ``concentration_binning`` is set to (1, 25, 0.5).

    Returns
    -------
    model_dictionary : dict
        Dictionary passed to `~halotools.empirical_models.HodModelFactory`

    Examples
    --------
    >>> from halotools.empirical_models import HodModelFactory
    >>> model_dictionary = leauthaud11_model_dictionary()
    >>> model_instance = HodModelFactory(**model_dictionary)

    The default settings are set in the `~halotools.empirical_models.model_defaults` module.
    To load a model based on a different threshold and redshift:

    >>> model_dictionary = leauthaud11_model_dictionary(threshold = 11, redshift = 1)
    >>> model_instance = HodModelFactory(**model_dictionary)

    For this model, you can also use the following syntax candy,
    which accomplishes the same task as the above:

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('leauthaud11', threshold = 11, redshift = 1)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """
    # Build model for centrals
    # Build the occupation model
    centrals_occupation = leauthaud11_components.Leauthaud11Cens(threshold=threshold, **kwargs)
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model

    centrals_profile = TrivialPhaseSpace(**kwargs)

    # Build model for satellites
    # Build the occupation model
    satellites_occupation = leauthaud11_components.Leauthaud11Sats(threshold=threshold, **kwargs)
    # Build the profile model
    satellites_profile = NFWPhaseSpace(**kwargs)

    return ({'centrals_occupation': centrals_occupation,
        'centrals_profile': centrals_profile,
        'satellites_occupation': satellites_occupation,
        'satellites_profile': satellites_profile})
