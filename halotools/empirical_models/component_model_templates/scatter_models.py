"""
Module containing the `~halotools.empirical_models.LogNormalScatterModel` class
used to model stochasticity in the mapping between stellar mass and halo properties.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .. import model_defaults
from .. import model_helpers as model_helpers

from ...utils.array_utils import custom_len


__all__ = ('LogNormalScatterModel', )
__author__ = ('Andrew Hearin', )


class LogNormalScatterModel(object):
    """ Simple model used to generate log-normal scatter
    in a stellar-to-halo-mass type relation.

    """

    def __init__(self,
            prim_haloprop_key=model_defaults.default_smhm_haloprop,
            **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the level of scatter.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        scatter_abscissa : array_like, optional
            Array of values giving the abscissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        scatter_ordinates : array_like, optional
            Array of values defining the level of scatter at the input abscissa.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        Examples
        ---------
        >>> scatter_model = LogNormalScatterModel()
        >>> scatter_model = LogNormalScatterModel(prim_haloprop_key='halo_mvir')

        To implement variable scatter, we need to define the level
        of log-normal scatter at a set of control values
        of the primary halo property. Here we give an example of a model
        in which the scatter is 0.3 dex for Milky Way table and 0.1 dex in cluster table:

        >>> scatter_abscissa = [12, 15]
        >>> scatter_ordinates = [0.3, 0.1]
        >>> scatter_model = LogNormalScatterModel(scatter_abscissa=scatter_abscissa, scatter_ordinates=scatter_ordinates)

        """

        default_scatter = model_defaults.default_smhm_scatter
        self.prim_haloprop_key = prim_haloprop_key

        if ('scatter_abscissa' in list(kwargs.keys())) and ('scatter_ordinates' in list(kwargs.keys())):
            self.abscissa = np.atleast_1d(kwargs['scatter_abscissa'])
            self.ordinates = np.atleast_1d(kwargs['scatter_ordinates'])
        else:
            self.abscissa = [12]
            self.ordinates = [default_scatter]

        self._initialize_param_dict()

        self._update_interpol()

    def mean_scatter(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added
        to the galaxy property as a function of the input table.

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
        scatter : array_like
            Array containing the amount of log-normal scatter evaluated
            at the input table.
        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_scatter:\n"
                "``table`` or ``prim_haloprop``")

        self._update_interpol()

        return self.spline_function(np.log10(mass))

    def scatter_realization(self, seed=None, **kwargs):
        """ Return the amount of log-normal scatter that should be added
        to the galaxy property as a function of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        seed : int, optional
            Random number seed. Default is None.

        Returns
        -------
        scatter : array_like
            Array containing a random variable realization that should be summed
            with the galaxy property to add scatter.
        """

        scatter_scale = self.mean_scatter(**kwargs)

        # initialize result with zero scatter result
        result = np.zeros(len(scatter_scale))

        # only draw from a normal distribution for non-zero values of scatter
        mask = (scatter_scale > 0.0)
        with NumpyRNGContext(seed):
            result[mask] = np.random.normal(loc=0, scale=scatter_scale[mask])

        return result

    def _update_interpol(self):
        """ Private method that updates the interpolating functon used to
        define the level of scatter as a function of the input table.
        If this method is not called after updating ``self.param_dict``,
        changes in ``self.param_dict`` will not alter the model behavior.
        """

        scipy_maxdegree = 5
        degree_list = [scipy_maxdegree, custom_len(self.abscissa)-1]
        self.spline_degree = np.min(degree_list)

        self.ordinates = [self.param_dict[self._get_param_key(i)] for i in range(len(self.abscissa))]

        self.spline_function = model_helpers.custom_spline(
            self.abscissa, self.ordinates, k=self.spline_degree)

    def _initialize_param_dict(self):
        """ Private method used to initialize ``self.param_dict``.
        """
        self.param_dict = {}
        for ipar, val in enumerate(self.ordinates):
            key = self._get_param_key(ipar)
            self.param_dict[key] = val

    def _get_param_key(self, ipar):
        """ Private method used to retrieve the key of self.param_dict
        that corresponds to the appropriately selected i^th ordinate
        defining the behavior of the scatter model.
        """
        return 'scatter_model_param'+str(ipar+1)
