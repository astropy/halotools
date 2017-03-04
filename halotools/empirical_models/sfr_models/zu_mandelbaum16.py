"""
Module containing the `~halotools.empirical_models.ZuMandelbaum16QuenchingCens`
and `~halotools.empirical_models.ZuMandelbaum16QuenchingSats` classes
responsible for providing a mapping between halo mass and galaxy quenching designation.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from ..component_model_templates import BinaryGalpropModel

__all__ = ('ZuMandelbaum16QuenchingCens', 'ZuMandelbaum16QuenchingSats')


class ZuMandelbaum16QuenchingCens(BinaryGalpropModel):
    """ Model for the quiescent fraction of centrals as a function of halo mass
    defined by an exponential function of halo mass.

    See :ref:`zu_mandelbaum16_composite_model` for a tutorial on this model.

    """

    def __init__(self, prim_haloprop_key='halo_m200m', **kwargs):
        """
        Parameters
        -----------
        prim_haloprop_key : string
            Name of the column of the halo table storing the
            mass-like variable the model is based on,
            e.g., 'halo_mvir' or 'halo_m200b'.

        Examples
        --------
        >>> model = ZuMandelbaum16QuenchingCens()
        """

        BinaryGalpropModel.__init__(self,
            galprop_name='quiescent', prim_haloprop_key=prim_haloprop_key)

        self.param_dict = self._retrieve_default_param_dict()
        self.gal_type = 'centrals'

    def mean_quiescent_fraction(self, **kwargs):
        r"""
        Quiescent fraction as a function of halo mass, modeled as an exponential:

        :math:`F_{\rm quiescent}(M_{\rm halo}) = 1 - {\rm exp}(-(M_{\rm halo}/M_{\rm char})^{\alpha})`

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        quiescent_fraction : array_like
            Array containing mean fraction of quiescent galaxies.

        Examples
        --------
        >>> model = ZuMandelbaum16QuenchingCens()
        >>> quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=1e12)
        """
        if 'table' in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs['table'][self.prim_haloprop_key])
        elif 'prim_haloprop' in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            raise KeyError("Must pass one of the following keyword arguments "
                "to mean_stellar_mass:\n``table`` or ``prim_haloprop``")

        mass_ratio = halo_mass/self.param_dict['quenching_mass_centrals']
        exparg = mass_ratio**self.param_dict['quenching_exp_power_centrals']
        result = 1 - np.exp(-exparg)
        result = np.where(result < 0, 0, result)
        result = np.where(result > 1, 1, result)
        return result

    def _retrieve_default_param_dict(self):
        d = {}
        d['quenching_mass_centrals'] = 10**12.2
        d['quenching_exp_power_centrals'] = 0.38
        return d


class ZuMandelbaum16QuenchingSats(BinaryGalpropModel):
    """ Model for the quiescent fraction of satellites as a function of halo mass
    defined by an exponential function of halo mass.

    See :ref:`zu_mandelbaum16_composite_model` for a tutorial on this model.

    """

    def __init__(self, prim_haloprop_key='halo_m200m', **kwargs):
        """
        Parameters
        -----------
        prim_haloprop_key : string
            Name of the column of the halo table storing the
            mass-like variable the model is based on,
            e.g., 'halo_mvir' or 'halo_m200b'.

        Examples
        --------
        >>> model = ZuMandelbaum16QuenchingSats()
        """

        BinaryGalpropModel.__init__(self,
            galprop_name='quiescent', prim_haloprop_key=prim_haloprop_key)

        self.param_dict = self._retrieve_default_param_dict()
        self.gal_type = 'satellites'

    def mean_quiescent_fraction(self, **kwargs):
        r"""
        Quiescent fraction as a function of halo mass, modeled as an exponential:

        :math:`F_{\rm quiescent}(M_{\rm halo}) = 1 - {\rm exp}(-(M_{\rm halo}/M_{\rm char})^{\alpha})`

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        quiescent_fraction : array_like
            Array containing mean fraction of quiescent galaxies.

        Examples
        --------
        >>> model = ZuMandelbaum16QuenchingSats()
        >>> quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop=1e12)
        """
        if 'table' in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs['table'][self.prim_haloprop_key])
        elif 'prim_haloprop' in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            raise KeyError("Must pass one of the following keyword arguments "
                "to mean_stellar_mass:\n``table`` or ``prim_haloprop``")

        mass_ratio = halo_mass/self.param_dict['quenching_mass_satellites']
        exparg = mass_ratio**self.param_dict['quenching_exp_power_satellites']
        result = 1 - np.exp(-exparg)
        result = np.where(result < 0, 0, result)
        result = np.where(result > 1, 1, result)
        return result

    def _retrieve_default_param_dict(self):
        d = {}
        d['quenching_mass_satellites'] = 10**12.17
        d['quenching_exp_power_satellites'] = 0.15
        return d
