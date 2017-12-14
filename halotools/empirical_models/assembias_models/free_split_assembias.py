r"""
This module contains the `~halotools.empirical_models.FreeSplitAssembias` class.
The purpose of this class is to modify the behavior of the "split" parameter in HeavisideAssembias, so it
is a free parameter.  It subclasses `HeavisideAssembias` and extends its features. Details can be found in
`McLaughlin et al 2017 (in prep)`_.
"""

import numpy as np

from . import HeavisideAssembias
from .. import model_helpers
from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import custom_len

__all__ = ('FreeSplitAssembias', )
__author__ = ('Sean McLaughlin', )

class FreeSplitAssembias(HeavisideAssembias):

    def _set_percentile_splitting(self, split=0.5, **kwargs):
        """
        Interpret constructor arguments and set up interpolation scheme

        In this subclass, add "split" as a free parameter, similar to assembias.
        """
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}

        if 'splitting_model' in kwargs:
            #self.splitting_model = kwargs['splitting_model']
            #func = getattr(self.splitting_model, kwargs['splitting_method_name'])
            #if isinstance(func, collections.Callable):
            #    self._input_split_func = func
            #else:
            raise HalotoolsError("Input ``splitting_model`` has not yet been implemented for the "
                                 "FreeSplitAssembias subclass.")

        # Make sure the code behaves properly whether or not we were passed an iterable
        try:
            iterator = iter(split)
            split = list(split)
        except TypeError:
            split = [split]

        if 'assembias_strength_abscissa' in kwargs:
            abscissa = kwargs['assembias_strength_abscissa']
            try:
                iterator = iter(abscissa)
                abscissa = list(abscissa)
            except TypeError:
                abscissa = [abscissa]
        else:
            abscissa = [2]

        if custom_len(abscissa) != custom_len(split):
            raise HalotoolsError("``assembias_strength`` and ``assembias_strength_abscissa`` "
                                 "must have the same length")

        self._split_abscissa = abscissa
        for ipar, val in enumerate(split):
            self.param_dict[self._get_free_split_assembias_param_dict_key(ipar)] = val

    def _get_free_split_assembias_param_dict_key(self, ipar):
        return self._method_name_to_decorate + '_' + self.gal_type + '_assembias_split' + str(ipar + 1)

    @model_helpers.bounds_enforcing_decorator_factory(0, 1)
    def percentile_splitting_function(self, prim_haloprop):
        """
        Method returns the fraction of halos that are ``type-2``
        as a function of the input primary halo property.

        Parameters
        -----------
        prim_haloprop : array_like
            Array storing the primary halo property.

        Returns
        -------
        split : float
            Fraction of ``type2`` halos at the input primary halo property.
        """

        #retrieve ordinates from our dictionary
        split_ordinates = np.array([self.param_dict[self._get_free_split_assembias_param_dict_key(ipar)]
                           for ipar in xrange(len(self._split_abscissa))])

        if self._loginterp:
            spline_function = model_helpers.custom_spline(
                np.log10(self._split_abscissa), split_ordinates, k=3)
            result = spline_function(np.log10(prim_haloprop))
        else:
            spline_function = model_helpers.custom_spline(
                self._split_abscissa, split_ordinates, k=3)
            result = spline_function(prim_haloprop)

        return result
