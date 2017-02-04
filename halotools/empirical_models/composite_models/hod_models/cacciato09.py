"""
Module containing the HOD-style composite model based on Cacciato et al. (2009).
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ...occupation_models import cacciato09_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

__all__ = ['cacciato09_model_dictionary']


def cacciato09_model_dictionary(threshold=10, **kwargs):
    """
    Dictionary to build an CLF-style model based on Cacciato et al. (2009),
    arXiv:0807.4932.
    The behavior of this model is governed by a conditional luminosity function
    (CLF).
    See :ref:`cacciato09_composite_model` for a tutorial on this model.


    There are two populations, centrals and satellites.
    Central occupation statistics are given by a nearest integer distribution;
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Cacciato09Cens`.
    Central galaxies are assumed to reside at the exact center of the host halo;
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

    Satellite occupation statistics are given by a Poisson distribution
    with first moment given by a power law with an exponential cut-off at the
    high-luminosity end;
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Cacciato09Sats`;
    satellites in this model follow an (unbiased) NFW profile, as governed by the
    `~halotools.empirical_models.NFWPhaseSpace` class.

    This composite model is built by the `~halotools.empirical_models.HodModelFactory`.

    Parameters
    ----------
    threshold : float, optional
        Logarithm of the primary galaxy property threshold. If the primary
        galaxy property is luminosity, it is given in h=1 solar luminosity
        units.

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
    >>> model_dictionary = cacciato09_model_dictionary()
    >>> model_instance = HodModelFactory(**model_dictionary)

    The default settings are set in the `~halotools.empirical_models.model_defaults` module.
    To load a model based on a different threshold and redshift:

    >>> model_dictionary = cacciato09_model_dictionary(threshold = 11, redshift = 1)
    >>> model_instance = HodModelFactory(**model_dictionary)

    For this model, you can also use the following syntax candy,
    which accomplishes the same task as the above:

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('cacciato09', threshold = 11, redshift = 1)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """
    # Build model for centrals
    # Build the occupation model
    centrals_occupation = cacciato09_components.Cacciato09Cens(
        threshold=threshold, **kwargs)
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model
    centrals_profile = TrivialPhaseSpace(**kwargs)

    # Build model for satellites
    # Build the occupation model
    satellites_occupation = cacciato09_components.Cacciato09Sats(
        threshold=threshold, **kwargs)
    # Build the profile model
    satellites_profile = NFWPhaseSpace(**kwargs)

    return ({'centrals_occupation': centrals_occupation,
             'centrals_profile': centrals_profile,
             'satellites_occupation': satellites_occupation,
             'satellites_profile': satellites_profile})
