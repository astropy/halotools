r"""
Module containing the `~halotools.empirical_models.HaloMassInterpolQuenching` class
responsible for providing a mapping between halo mass and galaxy quenching designation.
"""
from __future__ import (
    division, print_function, absolute_import)

import numpy as np
from ..component_model_templates import BinaryGalpropInterpolModel

__all__ = ['HaloMassInterpolQuenching']


class HaloMassInterpolQuenching(BinaryGalpropInterpolModel):
    r""" Model for the quiescent fraction as a function of halo mass
    defined by interpolating between a set of input control points.

    Notes
    -------
    The interpolation is automatically done in log-space.

    See also
    ----------
    BinaryGalpropInterpolModel : Parent class from which all behavior derives.
    """

    def __init__(self, halo_mass_key,
            halo_mass_abscissa, quiescent_fraction_control_values, **kwargs):
        r"""
        Parameters
        -----------
        halo_mass_key : string
            Name of the column of the halo table storing the
            mass-like variable the model is based on,
            e.g., 'halo_mvir' or 'halo_m200b'.

        halo_mass_abscissa : array_like
            Values of halo mass at which the quiescent fraction is specified.

        quiescent_fraction_control_values : array_like
            Values of the quiescent fraction evaluated at the ``halo_mass_abscissa``.

        gal_type : string, optional
            Name of the galaxy population.
            This is only relevant if you are building an HOD-style composite model.

        Examples
        ----------

        Suppose you wish to build a model for quenching in which
        1/4 of galaxies in Milky Way halos are quiescent and 9/10 of galaxies
        in cluster halos are quiescent:

        >>> model_instance = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.9])

        Your model has a method called ``mean_quiescent_fraction`` that accepts
        a ``prim_haloprop`` keyword argument:

        >>> mass_array = np.logspace(10, 15, 1000)
        >>> quiescent_fraction = model_instance.mean_quiescent_fraction(prim_haloprop = mass_array)

        You can also generate Monte Carlo realizations of quiescent designation:

        >>> quiescent_designation = model_instance.mc_quiescent(prim_haloprop = mass_array)

        Now ``quiescent_designation`` is a boolean-valued array of the same length as the
        input ``mass_array``. True values correspond to quiescent galaxies, and conversely.

        At any time, you can change the values of the quiescent fraction in your model
        by changing the appropriate key in ``param_dict``:

        >>> model_instance.param_dict['quiescent_ordinates_param1'] = 0.35

        The above line of code changed the quiescent fraction to 0.35 at the first control value
        of :math:`M_{\rm vir} = 10^{12}M_{\odot}`. You will have one parameter for every
        control value you used to instantiate the model. While you can always change the
        quiescent fraction of your model instance at any given control value, you cannot
        change the halo masses at which the control values are evaluated. To do that,
        you must instantiate a new model.

        If you passed in a ``gal_type`` keyword, the keys of your ``param_dict`` will
        reflect this choice:

        >>> model_instance = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.9], gal_type = 'centrals')
        >>> model_instance.param_dict['centrals_quiescent_ordinates_param1'] = 0.35

        The purpose for this distinction is to provide disambiguation for composite models
        that use the `HaloMassInterpolQuenching` class for more than one galaxy population.
        """
        quiescent_fraction_control_values = np.atleast_1d(quiescent_fraction_control_values)

        halo_mass_abscissa = np.atleast_1d(halo_mass_abscissa)
        log10_halo_mass_abscissa = np.log10(halo_mass_abscissa)

        BinaryGalpropInterpolModel.__init__(self,
            galprop_name='quiescent', prim_haloprop_key=halo_mass_key,
            galprop_abscissa=log10_halo_mass_abscissa,
            galprop_ordinates=quiescent_fraction_control_values,
            logparam=True, **kwargs)
