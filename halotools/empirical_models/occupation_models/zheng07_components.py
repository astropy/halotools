r"""
This module contains occupation components used by the Zheng07 composite model.
"""

import numpy as np
from scipy.special import erf
import warnings

from .occupation_model_template import OccupationComponent

from .. import model_defaults
from ..assembias_models import HeavisideAssembias, PreservingNgalHeavisideAssembias

from ...custom_exceptions import HalotoolsError

__all__ = ('Zheng07Cens', 'Zheng07Sats',
           'AssembiasZheng07Cens', 'AssembiasZheng07Sats',
           'PreservingNgalAssembiasZheng07Cens', 'PreservingNgalAssembiasZheng07Sats')


class Zheng07Cens(OccupationComponent):
    r""" ``Erf`` function model for the occupation statistics of central galaxies,
    introduced in Zheng et al. 2005, arXiv:0408564. This implementation uses
    Zheng et al. 2007, arXiv:0703457, to assign fiducial parameter values.

    .. note::

        The `Zheng07Cens` model is part of the ``zheng07``
        prebuilt composite HOD-style model. For a tutorial on the ``zheng07``
        composite model, see :ref:`zheng07_composite_model`.

    """

    def __init__(self,
            threshold=model_defaults.default_luminosity_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        Examples
        --------
        >>> cen_model = Zheng07Cens()
        >>> cen_model = Zheng07Cens(threshold=-19.5)
        >>> cen_model = Zheng07Cens(prim_haloprop_key='halo_m200b')

        See also
        ----------
        :ref:`zheng07_composite_model`

        :ref:`zheng07_using_cenocc_model_tutorial`
        """
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Zheng07Cens, self).__init__(gal_type='centrals',
            threshold=threshold, upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters(self.threshold)

        self.publications = ['arXiv:0408564', 'arXiv:0703457']

    def mean_occupation(self, **kwargs):
        r""" Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 2 of arXiv:0703457.

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
        mean_ncen : array
            Mean number of central galaxies in the input table.

        Examples
        --------
        >>> cen_model = Zheng07Cens()

        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``cen_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_ncen = cen_model.mean_occupation(table=fake_sim.halo_table)

        Notes
        -----
        The `mean_occupation` method computes the following function:

        :math:`\langle N_{\mathrm{cen}} \rangle_{M} =
        \frac{1}{2}\left( 1 +
        \mathrm{erf}\left( \frac{\log_{10}M -
        \log_{10}M_{min}}{\sigma_{\log_{10}M}} \right) \right)`

        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            raise HalotoolsError(msg)

        logM = np.log10(mass)
        mean_ncen = 0.5*(1.0 + erf(
            (logM - self.param_dict['logMmin']) / self.param_dict['sigma_logM']))

        return mean_ncen

    def get_published_parameters(self, threshold, publication='Zheng07'):
        r"""
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters
        ----------

        threshold : float
            Luminosity threshold defining the SDSS sample
            to which Zheng et al. fit their HOD model.
            If the ``publication`` keyword argument is set to ``Zheng07``,
            then ``threshold`` must be agree with one of the published values:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        publication : string, optional
            String specifying the publication that will be used to set
            the values of ``param_dict``. Default is Zheng et al. (2007).

        Returns
        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            agree with the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> cen_model = Zheng07Cens()
        >>> cen_model.param_dict = cen_model.get_published_parameters(cen_model.threshold)

        """

        def get_zheng07_params(threshold):
            # Load tabulated data from Zheng et al. 2007, Table 1
            logMmin_array = [11.35, 11.46, 11.6, 11.75, 12.02, 12.3, 12.79, 13.38, 14.22]
            sigma_logM_array = [0.25, 0.24, 0.26, 0.28, 0.26, 0.21, 0.39, 0.51, 0.77]
            # define the luminosity thresholds corresponding to the above data
            threshold_array = np.arange(-22, -17.5, 0.5)
            threshold_array = threshold_array[::-1]

            threshold_index = np.where(threshold_array == threshold)[0]
            if len(threshold_index) == 0:
                msg = ("\nInput luminosity threshold "
                    "does not match any of the Table 1 values \nof "
                    "Zheng et al. 2007 (arXiv:0703457).\n"
                    "Choosing the best-fit parameters "
                    "associated the default_luminosity_threshold variable \n"
                    "set in the model_defaults module.\n"
                    "You can always manually change the values in ``param_dict``.\n")
                warnings.warn(msg)
                threshold = model_defaults.default_luminosity_threshold
                threshold_index = np.where(threshold_array == threshold)[0]

            param_dict = (
                {'logMmin': logMmin_array[threshold_index[0]],
                'sigma_logM': sigma_logM_array[threshold_index[0]]}
                )

            return param_dict

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07', 'zheng2007', 'Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Zheng07Cens, only supported best-fit models are currently Zheng et al. 2007")


class Zheng07Sats(OccupationComponent):
    r""" Power law model for the occupation statistics of satellite galaxies,
    introduced in Kravtsov et al. 2004, arXiv:0308519. This implementation uses
    Zheng et al. 2007, arXiv:0703457, to assign fiducial parameter values.

    :math:`\langle N_{sat} \rangle_{M} = \left( \frac{M - M_{0}}{M_{1}} \right)^{\alpha}`.

    .. note::

        The `Zheng07Sats` model is part of the ``zheng07``
        prebuilt composite HOD-style model. For a tutorial on the ``zheng07``
        composite model, see :ref:`zheng07_composite_model`.

    """

    def __init__(self,
            threshold=model_defaults.default_luminosity_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            modulate_with_cenocc=False, cenocc_model=None, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        modulate_with_cenocc : bool, optional
            If set to True, the `Zheng07Sats.mean_occupation` method will
            be multiplied by the the first moment of the centrals:

            :math:`\langle N_{\mathrm{sat}}\rangle_{M}\Rightarrow\langle N_{\mathrm{sat}}\rangle_{M}\times\langle N_{\mathrm{cen}}\rangle_{M}`

            The ``cenocc_model`` keyword argument works together with
            the ``modulate_with_cenocc`` keyword argument to determine how
            the :math:`\langle N_{\mathrm{cen}}\rangle_{M}` prefactor is calculated.

        cenocc_model : `OccupationComponent`, optional
            If the ``cenocc_model`` keyword argument is set to its default value
            of None, then the :math:`\langle N_{\mathrm{cen}}\rangle_{M}` prefactor will be
            calculated according to `Zheng07Cens.mean_occupation`.
            However, if an instance of the `OccupationComponent` class is instead
            passed in via the ``cenocc_model`` keyword,
            then the first satellite moment will be multiplied by
            the ``mean_occupation`` function of the ``cenocc_model``.
            The ``modulate_with_cenocc`` keyword must be set to True in order
            for the ``cenocc_model`` to be operative.
            See :ref:`zheng07_using_cenocc_model_tutorial` for further details.

        Examples
        --------
        >>> sat_model = Zheng07Sats()
        >>> sat_model = Zheng07Sats(threshold = -21)

        The ``param_dict`` attribute can be used to build an alternate
        model from an existing instance. This feature has a variety of uses. For example,
        suppose you wish to study how the choice of halo mass definition impacts HOD predictions:

        >>> sat_model1 = Zheng07Sats(threshold = -19.5, prim_haloprop_key='m200b')
        >>> sat_model1.param_dict['alpha_satellites'] = 1.05
        >>> sat_model2 = Zheng07Sats(threshold = -19.5, prim_haloprop_key='m500c')
        >>> sat_model2.param_dict = sat_model1.param_dict

        After executing the above four lines of code, ``sat_model1`` and ``sat_model2`` are
        identical in every respect, excepting only for the difference in the halo mass definition.

        A common convention in HOD modeling of satellite populations is for the first
        occupation moment of the satellites to be multiplied by the first occupation
        moment of the associated central population.
        The ``cenocc_model`` keyword arguments allows you
        to study the impact of this choice:

        >>> sat_model1 = Zheng07Sats(threshold=-18)
        >>> sat_model2 = Zheng07Sats(threshold = sat_model1.threshold, modulate_with_cenocc=True)

        Now ``sat_model1`` and ``sat_model2`` are identical in every respect,
        excepting only the following difference:

        :math:`\langle N_{\mathrm{sat}}\rangle^{\mathrm{model 2}} = \langle N_{\mathrm{cen}}\rangle\times\langle N_{\mathrm{sat}}\rangle^{\mathrm{model 1}}`

        See also
        ----------
        :ref:`zheng07_composite_model`

        :ref:`zheng07_using_cenocc_model_tutorial`

        """
        upper_occupation_bound = float("inf")

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Zheng07Sats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters(self.threshold)

        if cenocc_model is None:
            cenocc_model = Zheng07Cens(
                prim_haloprop_key=prim_haloprop_key, threshold=threshold)
        else:
            if modulate_with_cenocc is False:
                msg = ("You chose to input a ``cenocc_model``, but you set the \n"
                    "``modulate_with_cenocc`` keyword to False, so your "
                    "``cenocc_model`` will have no impact on the model's behavior.\n"
                    "Be sure this is what you intend before proceeding.\n"
                    "Refer to the Zheng et al. (2007) composite model tutorial for details.\n")
                warnings.warn(msg)

        self.modulate_with_cenocc = modulate_with_cenocc
        if self.modulate_with_cenocc:
            try:
                assert isinstance(cenocc_model, OccupationComponent)
            except AssertionError:
                msg = ("The input ``cenocc_model`` must be an instance of \n"
                    "``OccupationComponent`` or one of its sub-classes.\n")
                raise HalotoolsError(msg)

            self.central_occupation_model = cenocc_model

            self.param_dict.update(self.central_occupation_model.param_dict)

        self.publications = ['arXiv:0308519', 'arXiv:0703457']

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass.

        :math:`\langle N_{\mathrm{sat}} \rangle_{M} = \left( \frac{M - M_{0}}{M_{1}} \right)^{\alpha} \langle N_{\mathrm{cen}} \rangle_{M}`

        or

        :math:`\langle N_{\mathrm{sat}} \rangle_{M} = \left( \frac{M - M_{0}}{M_{1}} \right)^{\alpha}`,

        depending on whether a central model was passed to the constructor.

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> sat_model = Zheng07Sats()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_nsat = sat_model.mean_occupation(table=fake_sim.halo_table)

        """

        if self.modulate_with_cenocc:
            for key, value in self.param_dict.items():
                if key in self.central_occupation_model.param_dict:
                    self.central_occupation_model.param_dict[key] = value

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)

        M0 = 10.**self.param_dict['logM0']
        M1 = 10.**self.param_dict['logM1']

        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning
        mean_nsat = np.zeros_like(mass)

        idx_nonzero = np.where(mass - M0 > 0)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_nsat[idx_nonzero] = ((mass[idx_nonzero] - M0)/M1)**self.param_dict['alpha']

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(self.central_occupation_model, "baseline_mean_occupation",\
                                    self.central_occupation_model.mean_occupation)(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat

    def get_published_parameters(self, threshold, publication='Zheng07'):
        r"""
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters
        ----------
        threshold : float
            Luminosity threshold of the mock galaxy sample.
            Input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        publication : string, optional
            String specifying the publication that will be used to set
            the values of ``param_dict``. Default is Zheng et al. (2007).

        Returns
        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> sat_model = Zheng07Sats()
        >>> sat_model.param_dict = sat_model.get_published_parameters(sat_model.threshold)
        """

        def get_zheng07_params(threshold):
            # Load tabulated data from Zheng et al. 2007, Table 1
            logM0_array = [11.2, 10.59, 11.49, 11.69, 11.38, 11.84, 11.92, 13.94, 14.0]
            logM1_array = [12.4, 12.68, 12.83, 13.01, 13.31, 13.58, 13.94, 13.91, 14.69]
            alpha_array = [0.83, 0.97, 1.02, 1.06, 1.06, 1.12, 1.15, 1.04, 0.87]
            # define the luminosity thresholds corresponding to the above data
            threshold_array = np.arange(-22, -17.5, 0.5)
            threshold_array = threshold_array[::-1]

            threshold_index = np.where(threshold_array == threshold)[0]

            if len(threshold_index) == 0:
                msg = ("\nInput luminosity threshold "
                    "does not match any of the Table 1 values \nof "
                    "Zheng et al. 2007 (arXiv:0703457).\n"
                    "Choosing the best-fit parameters "
                    "associated the default_luminosity_threshold variable \n"
                    "set in the model_defaults module.\n"
                    "You can always manually change the values in ``param_dict``.\n")
                warnings.warn(msg)
                threshold = model_defaults.default_luminosity_threshold
                threshold_index = np.where(threshold_array == threshold)[0]
                warnings.warn(msg)

            param_dict = (
                {'logM0': logM0_array[threshold_index[0]],
                'logM1': logM1_array[threshold_index[0]],
                'alpha': alpha_array[threshold_index[0]]}
                )
            return param_dict

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07', 'zheng2007', 'Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Zheng07Sats, only supported best-fit models are currently Zheng et al. 2007")


class AssembiasZheng07Sats(Zheng07Sats, HeavisideAssembias):
    r""" Assembly-biased modulation of `Zheng07Sats`.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Zheng07Sats.__init__(self, **kwargs)
        HeavisideAssembias.__init__(self,
            method_name_to_decorate='mean_occupation',
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            **kwargs)


class AssembiasZheng07Cens(Zheng07Cens, HeavisideAssembias):
    r""" Assembly-biased modulation of `Zheng07Cens`.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Zheng07Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)



class PreservingNgalAssembiasZheng07Sats(Zheng07Sats, PreservingNgalHeavisideAssembias):
    r""" Assembly-biased modulation of `Zheng07Sats` that preserves N_gals.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Zheng07Sats.__init__(self, **kwargs)
        PreservingNgalHeavisideAssembias.__init__(self,
            method_name_to_decorate='mean_occupation',
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            **kwargs)


class PreservingNgalAssembiasZheng07Cens(Zheng07Cens, PreservingNgalHeavisideAssembias):
    r""" Assembly-biased modulation of `Zheng07Cens` that preserves N_gals.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Zheng07Cens.__init__(self, **kwargs)
        PreservingNgalHeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)
