"""
Module defining the `~halotools.empirical_models.SFRBiasedNFWPhaseSpace` class
governing the phase space distribution of massless tracers of an NFW potential,
where the concentration of the tracers is permitted to differ from the
host halo concentration.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from .biased_nfw_phase_space import BiasedNFWPhaseSpace


__author__ = ('Andrew Hearin', )
__all__ = ('SFRBiasedNFWPhaseSpace', )


missing_quiescent_key_msg = ("The `SFRBiasedNFWPhaseSpace` class "
    "can only be used to make mocks \nin concert "
    "with some other component model that is responsible for \nmodeling an"
    "``quiescent`` property of the ``galaxy_table``.\n")


class SFRBiasedNFWPhaseSpace(BiasedNFWPhaseSpace):
    r""" Model for the phase space distribution of galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    where the concentration of the tracers is permitted to differ from the
    host halo concentration, independently for red and blue galaxies.

    For a review of the mathematics underlying the NFW profile,
    including descriptions of how the relevant equations are
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`.

    Notes
    ------
    The `SFRBiasedNFWPhaseSpace` class can only be used to make mocks in concert
    with some other component model that is responsible for modeling an
    ``quiescent`` property of the ``galaxy_table``.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        conc_gal_bias_logM_abscissa : array_like, optional
            Numpy array of shape (num_gal_bias_bins, ) storing the values of
            log10(Mhalo). For each entry of ``conc_gal_bias_logM_abscissa``,
            there will be a corresponding parameter in the ``param_dict`` allowing
            you to vary the strength of the galaxy concentration bias,
            using log-linear interpolation for the intermediate values
            between these control points.

        conc_mass_model : string or callable, optional
            Specifies the function used to model the relation between
            NFW concentration and halo mass.
            Can either be a custom-built callable function,
            or one of the following strings:
            ``dutton_maccio14``, ``direct_from_halo_catalog``.

        cosmology : object, optional
            Instance of an astropy `~astropy.cosmology`.
            Default cosmology is set in
            `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str, optional
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.

        concentration_key : string, optional
            Column name of the halo catalog storing NFW concentration.

            This argument is only relevant when ``conc_mass_model``
            is set to ``direct_from_halo_catalog``. In such a case,
            the default value is ``halo_nfw_conc``,
            which is consistent with all halo catalogs provided by Halotools
            but may differ from the convention adopted in custom halo catalogs.

        concentration_bins : ndarray, optional
            Array storing how halo concentrations will be digitized when building
            a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            concentration parameter can be recovered in a likelihood analysis.

        conc_gal_bias_bins : ndarray, optional
            Array storing how biases in galaxy concentrations will be digitized
            when building a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            galaxy bias parameter can be recovered in a likelihood analysis.

        profile_integration_tol : float, optional
            Default is 1e-5

        Examples
        ---------
        >>> biased_nfw = SFRBiasedNFWPhaseSpace()

        The behavior of the `~halotools.empirical_models.SFRBiasedNFWPhaseSpace`
        model is controlled by the values
        stored in its ``param_dict``. In the above default model, we have two parameters
        that control the concentration of the satellites,
        ``quiescent_conc_gal_bias`` and ``star_forming_conc_gal_bias``.
        The value of :math:`F_{\rm q}` = ``quiescent_conc_gal_bias`` sets the value for
        the concentration of quiescent satellites according to
        :math:`F_{\rm q} * c_{\rm halo},` where :math:`c_{\rm halo}` is the
        value of the halo concentration specified by the concentration-mass model.


        By default, :math:`F_{\rm q} = F_{\rm sf} = 1`.
        Quiescent satellites with *larger* values of
        :math:`F_{\rm q}` will be *more* radially concentrated and have *smaller*
        radial velocity dispersions, and likewise for star-forming galaxies.

        The `SFRBiasedNFWPhaseSpace` gives you the option to allow :math:`F_{\rm q}`
        and :math:`F_{\rm sf}` to vary with halo mass. You can accomplish this via the
        ``conc_gal_bias_logM_abscissa`` keyword:

        >>> biased_nfw_mass_dep = SFRBiasedNFWPhaseSpace(conc_gal_bias_logM_abscissa=[12., 15.])

        In the above model, the values of :math:`F_{\rm q} = F_{\rm q}(M_{\rm halo})`
        and :math:`F_{\rm sf} = F_{\rm sf}(M_{\rm halo})`
        can be independently specified at either of the two control points,
        :math:`10^{12}M_{\odot}` or :math:`10^{15}M_{\odot}`.
        For every element in the input ``conc_gal_bias_logM_abscissa``, there is a
        correposponding value in the ``param_dict`` controlling the value of
        :math:`F_{\rm q}` and :math:`F_{\rm sf}` at that mass.

        For example, in the ``biased_nfw_mass_dep`` defined above,
        to set the value of :math:`F_{\rm q}(M_{\rm halo} = 10^{15})`:

        >>> biased_nfw_mass_dep.param_dict['quiescent_conc_gal_bias_param1'] = 2

        To triple the value of :math:`F_{\rm sf}(M_{\rm halo} = 10^{12})`:

        >>> biased_nfw_mass_dep.param_dict['star_forming_conc_gal_bias_param0'] *= 3

        Values of :math:`F_{\rm q}` and :math:`F_{\rm sf}` at masses
        between the control points will be determined by log-linear interpolation.
        When extrapolating :math:`F_{\rm q}` and :math:`F_{\rm sf}` beyond
        the specified range, the values will be kept constant at the end point values.
        """
        BiasedNFWPhaseSpace.__init__(self, **kwargs)

    def _initialize_conc_bias_param_dict(self, **kwargs):
        r""" Set up the appropriate number of keys in the parameter dictionary
        and give the keys standardized names.
        """

        if 'conc_gal_bias_logM_abscissa' in list(kwargs.keys()):
            _conc_bias_logM_abscissa = np.atleast_1d(
                kwargs.get('conc_gal_bias_logM_abscissa')).astype('f4')

            d_q = ({'quiescent_conc_gal_bias_param'+str(i): 1.
                for i in range(len(_conc_bias_logM_abscissa))})
            d_sf = ({'star_forming_conc_gal_bias_param'+str(i): 1.
                for i in range(len(_conc_bias_logM_abscissa))})

            d_abscissa = ({'conc_gal_bias_logM_abscissa_param'+str(i): float(logM)
                for i, logM in enumerate(_conc_bias_logM_abscissa)})
            self._num_conc_bias_params = len(_conc_bias_logM_abscissa)

            d = {}
            d.update(d_q)
            d.update(d_sf)
            d.update(d_abscissa)
            return d

        else:
            return {'quiescent_conc_gal_bias': 1., 'star_forming_conc_gal_bias': 1.}

    def calculate_conc_gal_bias(self, seed=None, **kwargs):
        r""" Calculate the ratio of the galaxy concentration to the halo concentration,
        :math:`c_{\rm gal}/c_{\rm halo}`.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing the mass-like variable, e.g., ``halo_mvir``.

            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        quiescent : array, optional
            Boolean array storing whether the galaxy is quiescent.
            Must be passed together with prim_haloprop argument.

        table : object, optional
            `~astropy.table.Table` storing the halo catalog.

            If your NFW model is based on the virial definition,
            then ``halo_mvir`` must appear in the input table,
            and likewise for other halo mass definitions.

            If ``table`` is not passed,
            then ``prim_haloprop`` and ``quiescent`` keyword arguments must be passed.

        Returns
        -------
        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`F_{\rm gal} = c_{\rm gal}/c_{\rm halo}`.

        Examples
        --------
        >>> model = SFRBiasedNFWPhaseSpace()
        >>> mass = np.logspace(10, 15, 100)
        >>> quiescent = np.zeros(100).astype(bool)
        >>> quiescent[::2] = True
        >>> conc_gal_bias = model.calculate_conc_gal_bias(prim_haloprop=mass, quiescent=quiescent)
        """
        if 'table' in list(kwargs.keys()):
            table = kwargs['table']
            mass = table[self.prim_haloprop_key]
            try:
                quiescent = table['quiescent']
            except KeyError:
                raise KeyError(missing_quiescent_key_msg)
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop']).astype('f4')
            quiescent = np.atleast_1d(kwargs['quiescent']).astype(bool)
            if len(quiescent) == 1:
                quiescent = (np.zeros_like(mass) + quiescent[0]).astype(bool)
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``assign_conc_gal_bias`` function of the ``BiasedNFWPhaseSpace`` class.\n")
            raise KeyError(msg)

        if 'conc_gal_bias_logM_abscissa_param0' in self.param_dict.keys():
            abscissa_keys = list('conc_gal_bias_logM_abscissa_param'+str(i)
                for i in range(self._num_conc_bias_params))
            abscissa = [self.param_dict[key] for key in abscissa_keys]

            q_ordinates_keys = list('quiescent_conc_gal_bias_param'+str(i)
                for i in range(self._num_conc_bias_params))
            q_ordinates = [self.param_dict[key] for key in q_ordinates_keys]

            sf_ordinates_keys = list('star_forming_conc_gal_bias_param'+str(i)
                for i in range(self._num_conc_bias_params))
            sf_ordinates = [self.param_dict[key] for key in sf_ordinates_keys]

            result = np.zeros_like(mass)
            result[quiescent] = np.interp(np.log10(mass[quiescent]),
                abscissa, q_ordinates)
            result[~quiescent] = np.interp(np.log10(mass[~quiescent]),
                abscissa, sf_ordinates)
        else:
            result = np.zeros_like(mass)
            result[quiescent] = self.param_dict['quiescent_conc_gal_bias']
            result[~quiescent] = self.param_dict['star_forming_conc_gal_bias']

        if 'table' in list(kwargs.keys()):
            table['conc_gal_bias'][:] = result
            halo_conc = table['conc_NFWmodel']
            gal_conc = self._clipped_galaxy_concentration(halo_conc, result)
            table['conc_galaxy'][:] = gal_conc
        else:
            return result
