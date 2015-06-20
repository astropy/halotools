# -*- coding: utf-8 -*-
"""
Module containing classes used to perform abundance matching (SHAM)
and conditional abundance matching (CAM). 

"""
import numpy as np

from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import model_defaults
from . import model_helpers as model_helpers
from .smhm_components import PrimGalpropModel

from ..utils.array_utils import array_like_length as custom_len
from ..sim_manager import sim_defaults 
from ..utils import array_utils 

from warnings import warn
from functools import partial

class AbunMatchSmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on traditional abundance matching. 
    """

    def __init__(self, galaxy_abundance_abcissa, galaxy_abundance_ordinates, 
        scatter_level = 0.2, **kwargs):
        """
        Parameters 
        ----------
        galprop_key : string, optional keyword argument 
            Name of the galaxy property being assigned. Default is ``stellar mass``, 
            though another common case may be ``luminosity``. 

        galaxy_abundance_ordinates : array_like
            Length-Ng array storing the comoving number density of galaxies 
            The value ``galaxy_abundance_ordinates[i]`` gives the comoving number density 
            of galaxies evaluated at the galaxy property stored in ``galaxy_abundance_abcissa[i]``. 
            The most common two cases are where ``galaxy_abundance_abcissa`` stores either 
            stellar mass or luminosity, in which case ``galaxy_abundance_ordinates`` would 
            simply be the stellar mass function or the luminosity function, respectively. 

        galaxy_abundance_abcissa : array_like
            Length-Ng array storing the property of the galaxies for which the 
            abundance has been tabulated. 
             The value ``galaxy_abundance_ordinates[i]`` gives the comoving number density 
            of galaxies evaluated at the galaxy property stored in ``galaxy_abundance_abcissa[i]``. 
            The most common two cases are where ``galaxy_abundance_abcissa`` stores either 
            stellar mass or luminosity, in which case ``galaxy_abundance_ordinates`` would 
            simply be the stellar mass function or the luminosity function, respectively. 

        subhalo_abundance_ordinates : array_like, optional keyword argument 
            Length-Nh array storing the comoving number density of subhalos.
            The value ``subhalo_abundance_ordinates[i]`` gives the comoving number density 
            of subhalos of property ``subhalo_abundance_abcissa[i]``. 
            If keyword arguments ``subhalo_abundance_ordinates`` 
            and ``subhalo_abundance_abcissa`` are not passed, 
            then keyword arguments ``prim_haloprop_key`` and ``halos`` must be passed. 

        subhalo_abundance_abcissa : array_like, optional keyword argument 
            Length-Nh array storing the stellar mass of subhalos. 
            The value ``subhalo_abundance_ordinates[i]`` gives the comoving number density 
            of subhalos of property ``subhalo_abundance_abcissa[i]``. 
            If keyword arguments ``subhalo_abundance_ordinates`` 
            and ``subhalo_abundance_abcissa`` are not passed, 
            then keyword arguments ``prim_haloprop_key`` and ``halos`` must be passed. 

        scatter_level : float, optional keyword argument 
            Level of constant scatter in dex. Default is 0.2. 

        input_param_dict : dict, optional keyword argument
            Dictionary containing values for the parameters specifying the model.
            If none is passed, the `Moster13SmHm` instance will be initialized to 
            the best-fit values taken from Moster et al. (2013). 
        """

        kwargs['scatter_model'] = LogNormalScatterModel
        kwargs['scatter_abcissa'] = [12]
        kwargs['scatter_ordinates'] = [scatter_level]

        super(AbunMatchSmHm, self).__init__(**kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        return None

class ConditionalAbunMatch(model_helpers.GalPropModel):
    """ Class to produce any CAM-style model of a galaxy property, such as age matching.  
    """
    def __init__(self, minimum_sampling_requirement=100, tol=0.01, 
        **kwargs):
        """ 
        Parameters 
        ----------
        galprop_key : string, keyword argument
            Column name of the galaxy property being modeled, e.g., color. 

        prim_galprop_key : string, keyword argument 
            Column name storing the galaxy property used to define the 
            conditional one-point statistics, e.g., stellar mass 
            or luminosity. 

        sec_haloprop_key : string, keyword argument 
            Column name storing the halo property that will be correlated 
            with galprop at fixed prim_galprop

        input_galaxy_table : data table 
            Astropy Table object storing the input galaxy population.  
            The conditional one-point functions of this population 
            will be used as inputs when building the primary behavior 
            of the `ConditionalAbunMatch` model. 

        prim_galprop_bins : array 
            Array used to bin the input galaxy population by the 
            prim_galprop of the model. 

        correlation_strength : float or array, optional keyword argument 
            Specifies the absolute value of the desired 
            Spearmann rank-order correlation coefficient 
            between the secondary halo property and the galprop. 
            If a float, the correlation strength will be assumed constant 
            for all values of the prim_galprop. If an array, the i^th entry 
            specifies the correlation strength when prim_galprop equals  
            ``prim_galprop_bins[i]``. 
            Default is None, in which case zero scatter is assumed. 

        correlation_strength_abcissa : float or array, optional keyword argument 
            Specifies the value if the primary galaxy property at which 
            the input correlation_strength applies. ``correlation_strength_abcissa`` 
            need only be specified if a ``correlation_strength`` array is passed. 

        tol : float, optional keyword argument 
            Tolerance for the difference between the actual and desired 
            correlation strength in each prim_galprop bin. Default is 0.01. 

        minimum_sampling_requirement : int, optional
            Minimum number of galaxies in the prim_galprop bin required to 
            adequately sample the galprop PDF. Default is 100. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Dictionary of function objects used by the mock factory 
            to create additional halo properties during a halo catalog pre-processing 
            phase. Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog that will be passed 
            to methods of `ConditionalAbunMatch`; each dict value 
            of ``new_haloprop_func_dict`` is a function object that returns 
            a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
        """

        self.minimum_sampling = minimum_sampling_requirement
        self.tol = tol

        required_kwargs = (
            ['galprop_key', 'prim_galprop_key', 'prim_galprop_bins', 'sec_haloprop_key'])
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)
        setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)
        super(ConditionalAbunMatch, self).__init__(galprop_key=self.galprop_key)

        self._build_param_dict(**kwargs)

        self.build_one_point_lookup_table(**kwargs)

        if 'new_haloprop_func_dict' in kwargs.keys():
            self.new_haloprop_func_dict = kwargs['new_haloprop_func_dict']

    def _mc_galprop(self, seed=None, **kwargs):
        """
        Parameters 
        ----------
        galaxy_table : data table 
            Astropy Table object storing the mock galaxy population.  

        galaxy_table_slice_array : array, optional keyword argument 
            Array of slice objects. The i^th entry of 
            ``galaxy_table_slice_array`` stores the slice of 
            the halo catalog which falls into the i^th 
            stellar mass bin. Useful if exploring models 
            with fixed stellar masses. Default is None, 
            in which case the `zero_scatter_relation` determines 
            this information for itself (at a performance cost). 

        Returns 
        -------
        output_galprop : array 
            Numpy array storing a Monte Carlo realization of 
            the modeled galaxy property. The `zero_scatter_relation` 
            assumes there is no scatter between the sec_haloprop 
            and the galprop. 
        """
        galaxy_table = kwargs['galaxy_table']

        if 'galaxy_table_slice_array' not in kwargs.keys():
            binned_prim_galprop = np.digitize(
                galaxy_table[self.prim_galprop_key], 
                self.prim_galprop_bins)

        # All at once, draw all the randoms we will need
        np.random.seed(seed=seed)
        all_randoms = np.random.random(len(galaxy_table)*2)
        galprop_cumprob = all_randoms[0:len(galaxy_table)]
        galprop_scatter = all_randoms[len(galaxy_table):]

        # Initialize the output array
        output_galprop = np.zeros(len(galaxy_table))

        for i in range(len(self.one_point_lookup_table)):

            # Determine the slice corresponding to the i^th prim_galprop bin
            if 'galaxy_table_slice_array' not in kwargs.keys():
                idx_bini = np.where(binned_prim_galprop==i)[0]
            else:
                idx_bini = kwargs['galaxy_table_slice_array'][i]

            if len(idx_bini) > 0:
                # Fetch the appropriate number of randoms
                # for the i^th prim_galprop bin
                galprop_cumprob_bini = galprop_cumprob[idx_bini]
                galprop_scatter_bini = galprop_scatter[idx_bini]

                # Fetch the halos in the i^th prim_galprop bin, 
                # and determine how they are sorted
                haloprop_bini = galaxy_table[idx_bini][self.sec_haloprop_key]
                idx_sorted_haloprop_bini = np.argsort(haloprop_bini)

                galprop_bini = self._condition_matched_galprop(
                    haloprop_bini[idx_sorted_haloprop_bini], 
                    galprop_cumprob_bini, i, 
                    0.75, galprop_scatter_bini, self.tol)

                # Assign the final values to the 
                # appropriately sorted subarray of output_galprop
                output_galprop[idx_bini[idx_sorted_haloprop_bini]] = galprop_bini

        return output_galprop

    def _condition_matched_galprop(self, sorted_haloprop, galprop_cumprob, ibin, 
        desired_correlation, randoms, tol):

        additional_noise = np.random.random(len(galprop_cumprob))

        def compute_pearson_difference(r):
            new_randoms = galprop_cumprob + r*randoms
            idx_sorted = np.argsort(new_randoms)
            galprop = (
                self.one_point_lookup_table[ibin](galprop_cumprob[idx_sorted]))
            return abs(pearsonr(galprop, sorted_haloprop)[0]-desired_correlation)

        result = minimize_scalar(compute_pearson_difference, tol=tol)
        new_randoms = galprop_cumprob + result.x*randoms
        idx_sorted = np.argsort(new_randoms)
        galprop = (
            self.one_point_lookup_table[ibin](galprop_cumprob[idx_sorted]))

        return galprop


    def build_one_point_lookup_table(self, **kwargs):
        """
        Parameters 
        ----------
        input_galaxy_table : data table 
            Astropy Table object storing the input galaxy population.  
            The conditional one-point functions of this population 
            will be used as inputs when building the primary behavior 
            of the `ConditionalAbunMatch` model. 

        prim_galprop_bins : array 
            Array used to bin the input galaxy population by the 
            prim_galprop of the model. 

        """
        galaxy_table = kwargs['input_galaxy_table']
        prim_galprop_bins = kwargs['prim_galprop_bins']

        self.one_point_lookup_table = np.zeros(
            len(prim_galprop_bins)+1, dtype=object)

        binned_prim_galprop = np.digitize(
            galaxy_table[self.prim_galprop_key], 
            self.prim_galprop_bins)

        for i in range(len(self.one_point_lookup_table)):
            idx_bini = np.where(binned_prim_galprop == i)[0]
            if model_helpers.custom_len(idx_bini) > self.minimum_sampling:
                gals_bini = galaxy_table[idx_bini]
                abcissa = np.arange(len(gals_bini))/float(len(gals_bini)-1)
                ordinates = np.sort(gals_bini[self.galprop_key])
                self.one_point_lookup_table[i] = (
                    model_helpers.custom_spline(abcissa, ordinates, k=2)
                    )

        # For all empty lookup tables, fill them with the nearest lookup table
        unfilled_lookup_table_idx = np.where(
            self.one_point_lookup_table == 0)[0]
        filled_lookup_table_idx = np.where(
            self.one_point_lookup_table != 0)[0]

        if len(unfilled_lookup_table_idx) > 0:
            msg = ("When building the one-point lookup table from input_galaxy_table, " + 
                "there were some bins of prim_galprop_bins that contained fewer than " + 
                str(self.minimum_sampling)+ " galaxies. In such cases, the lookup table " + 
                "of the nearest sufficiently populated bin will be chosen.")
            warn(msg)
        for idx in unfilled_lookup_table_idx:
            closest_filled_idx_idx = array_utils.find_idx_nearest_val(
                filled_lookup_table_idx, idx)
            closest_filled_idx = filled_lookup_table_idx[closest_filled_idx_idx]
            self.one_point_lookup_table[idx] = (
                self.one_point_lookup_table[closest_filled_idx])

    def _build_param_dict(self, **kwargs):
        
        if 'correlation_strength' in kwargs.keys():

            correlation_strength = kwargs['correlation_strength']
            if custom_len(correlation_strength) > 1:
                try:
                    self.correlation_strength_abcissa = kwargs['correlation_strength_abcissa']
                except KeyError:
                    msg = ("If correlation_strength keyword is passed to the constructor, \n" + 
                        "you must also pass a correlation_strength_abcissa keyword argument " + 
                        "storing an array of the same length as correlation_strength.")
                    raise(msg)
            else:
                self.correlation_strength_abcissa = [0]
                correlation_strength = [correlation_strength]

            self._param_dict_keys = ['correlation_param' + str(i+1) for i in range(len(correlation_strength))]
            self.param_dict = {key:value for key, value in zip(self._param_dict_keys, correlation_strength)}

            self._set_correlation_strength()

    def _set_correlation_strength(self, **kwargs):

        model_helpers.update_param_dict(self, **kwargs)
        abcissa = self.correlation_strength_abcissa
        ordinates = [self.param_dict['correlation_param'+str(i+1)] for i in range(len(abcissa))]
        correlation_strength_spline = model_helpers.custom_spline(abcissa, ordinates)
        self.correlation_strength = correlation_strength_spline(self.prim_galprop_bins)

























