"""
Module containing some commonly used composite HOD models.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ... import factories
from ...occupation_models import leauthaud11_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

__all__ = ['hearin15_model_dictionary']


def hearin15_model_dictionary(central_assembias_strength=1,
        central_assembias_strength_abscissa=[1e12],
        satellite_assembias_strength=0.2,
        satellite_assembias_strength_abscissa=[1e12],
        **kwargs):
    """
    Dictionary to build an HOD-style model in which
    central and satellite occupations statistics are assembly-biased.

    See :ref:`hearin15_composite_model` for a tutorial on this model.

    Parameters
    ----------
    threshold : float, optional
        Stellar mass threshold of the mock galaxy sample.
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    redshift : float, optional
        Default is set in the `~halotools.sim_manager.sim_defaults` module.

    sec_haloprop_key : string, optional
        String giving the column name of the secondary halo property modulating
        the occupation statistics of the galaxies.
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    central_assembias_strength : float or list, optional
        Fraction or list of fractions between -1 and 1 defining
        the assembly bias correlation strength. Default is a constant strength of 0.5.

    central_assembias_strength_abscissa : list, optional
        Values of the primary halo property at which the assembly bias strength is specified.
        Default is a constant strength of 0.5.

    satellite_assembias_strength : float or list, optional
        Fraction or list of fractions between -1 and 1 defining
        the assembly bias correlation strength. Default is a constant strength of 0.5.

    satellite_assembias_strength_abscissa : list, optional
        Values of the primary halo property at which the assembly bias strength is specified.
        Default is a constant strength of 0.5.

    split : float, optional
        Fraction between 0 and 1 defining how
        we split halos into two groupings based on
        their conditional secondary percentiles.
        Default is 0.5 for a constant 50/50 split.

    concentration_binning : tuple, optional
        Three-element tuple. The first entry will be the minimum
        value of the concentration in the lookup table for the satellite NFW profile,
        the second entry the maximum, the third entry
        the linear spacing of the grid.
        Default is set in `~halotools.empirical_models.model_defaults`.
        If high-precision is not required, the lookup tables will build much faster if
        ``concentration_binning`` is set to (1, 25, 0.5).

    Examples
    --------

    >>> model_dictionary = hearin15_model_dictionary()
    >>> model_instance = factories.HodModelFactory(**model_dictionary)

    The default settings are set in the `~halotools.empirical_models.model_defaults` module.
    To load a model based on a different threshold and redshift:

    >>> model_dictionary = hearin15_model_dictionary(threshold = 11, redshift = 1)
    >>> model_instance = factories.HodModelFactory(**model_dictionary)

    For this model, you can also use the following syntax candy,
    which accomplishes the same task as the above:

    >>> model_instance = factories.PrebuiltHodModelFactory('hearin15', threshold = 11, redshift = 1)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """
    ##############################
    # Build the occupation model
    if central_assembias_strength == 0:
        centrals_occupation = leauthaud11_components.Leauthaud11Cens(**kwargs)
    else:
        centrals_occupation = leauthaud11_components.AssembiasLeauthaud11Cens(
            assembias_strength=central_assembias_strength,
            assembias_strength_abscissa=central_assembias_strength_abscissa,
            **kwargs)

    # Build the profile model
    centrals_profile = TrivialPhaseSpace(**kwargs)

    ##############################
    # Build the occupation model

    if satellite_assembias_strength == 0:
        satellites_occupation = leauthaud11_components.Leauthaud11Sats(**kwargs)
    else:
        satellites_occupation = leauthaud11_components.AssembiasLeauthaud11Sats(
            assembias_strength=satellite_assembias_strength,
            assembias_strength_abscissa=satellite_assembias_strength_abscissa,
            cenocc_model=centrals_occupation, **kwargs)

    # Build the profile model
    satellites_profile = NFWPhaseSpace(**kwargs)
    satellites_profile._suppress_repeated_param_warning = True

    return ({'centrals_occupation': centrals_occupation,
        'centrals_profile': centrals_profile,
        'satellites_occupation': satellites_occupation,
        'satellites_profile': satellites_profile})
