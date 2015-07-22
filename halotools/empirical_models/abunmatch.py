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
from . import model_helpers
from .smhm_components import PrimGalpropModel
from . import smhm_components

from ..utils.array_utils import array_like_length as custom_len
from ..sim_manager import sim_defaults
from .. import sim_manager

from ..utils import array_utils 

from warnings import warn
from functools import partial

__all__ = ['ConditionalAbunMatch']

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
            Length-Nh array storing the comoving number density of subhalo_table.
            The value ``subhalo_abundance_ordinates[i]`` gives the comoving number density 
            of subhalo_table of property ``subhalo_abundance_abcissa[i]``. 
            If keyword arguments ``subhalo_abundance_ordinates`` 
            and ``subhalo_abundance_abcissa`` are not passed, 
            then keyword arguments ``prim_haloprop_key`` and ``halo_table`` must be passed. 

        subhalo_abundance_abcissa : array_like, optional keyword argument 
            Length-Nh array storing the stellar mass of subhalo_table. 
            The value ``subhalo_abundance_ordinates[i]`` gives the comoving number density 
            of subhalo_table of property ``subhalo_abundance_abcissa[i]``. 
            If keyword arguments ``subhalo_abundance_ordinates`` 
            and ``subhalo_abundance_abcissa`` are not passed, 
            then keyword arguments ``prim_haloprop_key`` and ``halo_table`` must be passed. 

        scatter_level : float, optional keyword argument 
            Level of constant scatter in dex. Default is 0.2. 

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
        input_galaxy_table : data table, required keyword argument 
            Astropy Table object storing the input galaxy population 
            upon which the CAM model is based.  

        prim_galprop_key : string, required keyword argument 
            Column name such as ``stellar_mass`` or ``luminosity`` 
            where the primary galaxy property is stored in 
            ``input_galaxy_table``. 

        galprop_key : string, required keyword argument 
            Column name such as ``gr_color`` or ``ssfr`` 
            of the secondary galaxy property being modeled. 
            Can be any column of ``input_galaxy_table`` other than 
            ``prim_galprop_key``. 

        sec_haloprop_key : string, required keyword argument 
            Column name of the subhalo property that CAM models as 
            being correlated with ``galprop_key`` at fixed ``prim_galprop_key``. 

        prim_galprop_bins : array, required keyword argument 
            Array used to bin ``input_galaxy_table`` by ``prim_galprop_key``. 

        correlation_strength : float or array, optional keyword argument 
            Specifies the absolute value of the desired 
            Spearman rank-order correlation coefficient 
            between ``sec_haloprop_key`` and ``galprop_key``. 
            If a float, the correlation strength will be assumed constant 
            for all values of ``prim_galprop_key``. If an array, the i^th entry 
            specifies the correlation strength when ``prim_galprop_key`` equals  
            ``prim_galprop_bins[i]``. Entries must be in the range [-1, 1], 
            with negative values corresponding to anti-correlations; 
            the endpoints signify maximum correlation, zero signifies 
            that ``sec_haloprop_key`` and ``galprop_key`` are uncorrelated. 
            Default is maximum (positive) correlation strength of 1. 

        correlation_strength_abcissa : float or array, optional keyword argument 
            Specifies the value of ``prim_galprop_key`` at which 
            the input ``correlation_strength`` applies. ``correlation_strength_abcissa`` 
            need only be specified if a ``correlation_strength`` array is passed. 
            Intermediary values of the correlation strength at values 
            between the abcissa are solved for by spline interpolation. 

        tol : float, optional keyword argument 
            Tolerance for the difference between the actual and desired 
            correlation strength. Default is 0.01. 

        minimum_sampling_requirement : int, optional keyword argument 
            Minimum number of galaxies in the ``prim_galprop_key`` bin required to 
            adequately sample the probability distribution of ``galprop_key`` at 
            fixed ``prim_galprop_key``. Default is 100. 
            For ``prim_galprop_key`` bins not meeting this requirement, 
            the nearest ``prim_galprop_key`` bin of sufficient size 
            will be used  to determine the conditional PDF. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Can be used to create new halo properties with which to introduce 
            correlations. This is useful for cases where the desired ``sec_haloprop_key`` 
            does not appear in the input subhalo catalog. 
            ``new_haloprop_func_dict`` is a dictionary of 
            function objects. Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog that will be 
            created. Each dict value of ``new_haloprop_func_dict`` is a function object 
            that returns a length-N numpy array when passed a 
            length-N Astropy table via the ``halo_table`` keyword argument. 
            By passing the ``new_haloprop_func_dict`` keyword argument, 
            your newly created halo property will be guaranteed to exist, 
            and can thus be optionally used as the ``sec_haloprop_key``. 
            This is the device used by 
            `~halotools.empirical_models.SubhaloMockFactory`  
            to create new halo properties during 
            a halo catalog pre-processing phase. 

        Examples 
        --------
        Any `ConditionalAbunMatch` model is based on an input galaxy population, so to build a CAM model 
        we must start with some data that defines our conditional one-point distributions. 
        For demonstration purposes we can use fake data generated by 
        `~halotools.sim_manager.FakeMock`:

        >>> fake_data = sim_manager.FakeMock()

        We now must specify how to bin our data according to the primary galaxy property, 
        which for this example will be stellar mass:

        >>> sm_min = fake_data.galaxy_table['stellar_mass'].min()
        >>> sm_max = fake_data.galaxy_table['stellar_mass'].max()
        >>> sm_bins = np.logspace(np.log10(sm_min)-0.01, np.log10(sm_max)+0.01, 15)

        Now let's build a `ConditionalAbunMatch` model in which 
        ``gr_color`` and halo formation time ``zhalf`` are 
        in monotonic correspondence at fixed stellar mass:

        >>> cam_noscatter = ConditionalAbunMatch(galprop_key='gr_color', prim_galprop_key = 'stellar_mass', sec_haloprop_key = 'halo_zhalf', input_galaxy_table = fake_data.galaxy_table, prim_galprop_bins = sm_bins)

        ``cam_noscatter`` is our model object that can now be used to generate mock galaxy 
        populations. To do so, we need only pass ``cam_noscatter`` a halo catalog. For 
        demonstration purposes, we'll use a fake halo catalog generated by 
        `~halotools.sim_manager.FakeSim`:

        >>> fake_sim = sim_manager.FakeSim()
        >>> halo_table = fake_sim.halo_table

        CAM models do not assign values of ``prim_galprop``, 
        and so our halo_table must have stellar masses assigned by 
        some other means before calling `ConditionalAbunMatch`. Any 
        model for stellar mass can work with CAM, the following example is based on 
        `~halotools.empirical_models.Moster13SmHm`:

        >>> moster_model = smhm_components.Moster13SmHm(redshift=0)
        >>> halo_table['stellar_mass'] = moster_model.mc_stellar_mass(halo_table=halo_table)

        To assign values of ``gr_color`` to our halo_table, we call the ``mc_gr_color`` method 
        of our model object:

        >>> halo_table['gr_color'] = cam_noscatter.mc_gr_color(halo_table=halo_table)

        The `ConditionalAbunMatch` class can also produce models in which there is scatter 
        between the galaxy and halo property. The level of scatter is defined by the 
        Pearson rank-order correlation coefficient, which can take on values between [-1, 1]. 
        In the example below, we will construct a CAM model for specific star-formation rate 
        implementing a positive correlation of 50% correlation strength 
        between ``zhalf`` and ``ssfr`` at fixed ``stellar_mass``:

        >>> cam_constant_scatter = ConditionalAbunMatch(galprop_key='ssfr', prim_galprop_key = 'stellar_mass', sec_haloprop_key = 'halo_zhalf', input_galaxy_table = fake_data.galaxy_table, prim_galprop_bins = sm_bins, correlation_strength = 0.5)
        >>> halo_table['ssfr'] = cam_constant_scatter.mc_ssfr(halo_table=halo_table)

        After building a CAM model, you can vary the strength of the amount of scatter 
        by the model's parameter dictionary:

        >>> cam_constant_scatter.param_dict['correlation_strength_param1'] = 0.75

        Calling the ``mc_ssfr`` method will now implement a 75% correlation strength:

        >>> halo_table['ssfr'] = cam_constant_scatter.mc_ssfr(halo_table=halo_table)

        The `ConditionalAbunMatch` class also has support for levels of scatter that 
        vary with the primary galaxy property. To construct such a model, you provide 
        an array for the ``correlation_strength`` keyword argument, as well as an array 
        of the same length for the ``correlation_strength_abcissa`` argument. The latter 
        specifies the values of the primary galaxy property at which the correlation 
        strength attains the values given by the ``correlation_strength`` array. 

        For example, suppose we wish to have 75% correlation strength at a stellar mass 
        of :math:`10^{10}M_{\odot}`, and a 25% correlation strength at a stellar mass of :math:`10^{11}M_{\odot}` :

        >>> cam_variable_scatter = ConditionalAbunMatch(galprop_key='ssfr', prim_galprop_key = 'stellar_mass', sec_haloprop_key = 'halo_zhalf', input_galaxy_table = fake_data.galaxy_table, prim_galprop_bins = sm_bins, correlation_strength = [0.75, 0.25], correlation_strength_abcissa = [1.e10, 1.e11])
        >>> halo_table['ssfr'] = cam_variable_scatter.mc_ssfr(halo_table=halo_table)

        Now, there are multiple parameters governing the scatter, one for the strength at 
        each value of ``correlation_strength_abcissa``. We can modulate the scatter level 
        independently at each value of the abcissa by changing the values of ``param_dict``. 
        Here is an example of how to change the correlation strength at the second 
        abcissa value (in our case a stellar mass of :math:`10^{11}M_{\odot}`):

        >>> cam_variable_scatter.param_dict['correlation_param2'] = 0.5
        >>> halo_table['ssfr'] = cam_variable_scatter.mc_ssfr(halo_table=halo_table)

        .. automethod:: _mc_galprop
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
        Private method controlling the primary algorithm behind the  
        implementation of conditional abundance matching. 
        This method will be renamed according to ``self.galprop_key`` in the 
        instance of `ConditionalAbunMatch`. For example, 
        if the property being modeled is ``gr_color``, then the 
        `_mc_galprop` function would instead be named ``mc_gr_color``, 
        a bound method to the `ConditionalAbunMatch` class instance. 

        Parameters 
        ----------
        halo_table : data table
            Astropy Table object storing the halo population 
            onto which values of ``self.galprop_key`` will be painted. 

        galaxy_table_slice_array : array, optional keyword argument 
            Array of slice objects. The i^th entry of 
            ``galaxy_table_slice_array`` stores the slice of 
            the halo catalog which falls into the i^th 
            stellar mass bin. Useful if exploring models 
            with fixed stellar masses. Default is None, 
            in which case the `_mc_galprop` method determines 
            this information for itself (at a performance cost). 

        Returns 
        -------
        output_galprop : array 
            Numpy array storing a Monte Carlo realization of 
            the modeled galaxy property. 
        """
        self._set_correlation_strength()

        galaxy_table = kwargs['halo_table']

        self.add_new_haloprops(galaxy_table)

        # All at once, draw all the randoms we will need
        np.random.seed(seed=seed)
        all_randoms = np.random.random(len(galaxy_table)*2)
        galprop_cumprob = all_randoms[0:len(galaxy_table)]
        galprop_scatter = all_randoms[len(galaxy_table):]

        # Initialize the output array
        output_galprop = np.zeros(len(galaxy_table))

        # Determine binning and loop range
        if 'galaxy_table_slice_array' not in kwargs.keys():
            binned_prim_galprop = np.digitize(
                galaxy_table[self.prim_galprop_key], 
                self.prim_galprop_bins)
            prim_galprop_loop_range = set(binned_prim_galprop)
        else:
            prim_galprop_loop_range = range(len(self.one_point_lookup_table))

        for i in prim_galprop_loop_range:

            # Determine the slice corresponding to the i^th prim_galprop bin
            if 'galaxy_table_slice_array' not in kwargs.keys():
                idx_bini = np.where(binned_prim_galprop==i)[0]
                num_bini = len(idx_bini)
            else:
                idx_bini = kwargs['galaxy_table_slice_array'][i]
                num_bini = len(galaxy_table[idx_bini])

            if len(idx_bini) > 0:
                # Fetch the appropriate number of randoms
                # for the i^th prim_galprop bin
                galprop_cumprob_bini = galprop_cumprob[idx_bini]
                galprop_scatter_bini = galprop_scatter[idx_bini]

                # Fetch the halo_table in the i^th prim_galprop bin, 
                # and determine how they are sorted
                haloprop_bini = galaxy_table[idx_bini][self.sec_haloprop_key]
                idx_sorted_haloprop_bini = np.argsort(haloprop_bini)

                galprop_bini = self._condition_matched_galprop(
                    haloprop_bini[idx_sorted_haloprop_bini], 
                    galprop_cumprob_bini, i, galprop_scatter_bini, self.tol)

                # Assign the final values to the 
                # appropriately sorted subarray of output_galprop
                output_galprop[idx_bini[idx_sorted_haloprop_bini]] = galprop_bini

        return output_galprop

    def _condition_matched_galprop(self, sorted_haloprop, galprop_cumprob, 
        ibin, randoms, tol):

        def compute_pearson_difference(r):
            new_randoms = galprop_cumprob + r*randoms
            idx_sorted = np.argsort(new_randoms)
            galprop = (
                self.one_point_lookup_table[ibin](galprop_cumprob[idx_sorted]))
            return abs(pearsonr(galprop, sorted_haloprop)[0]-abs(self.correlation_strength[ibin]))

        if hasattr(self, 'correlation_strength'):
            result = minimize_scalar(compute_pearson_difference, tol=tol)
            new_randoms = galprop_cumprob + result.x*randoms
            idx_sorted = np.argsort(new_randoms)
            galprop = (
                self.one_point_lookup_table[ibin](galprop_cumprob[idx_sorted]))
        else:
            # Zero scatter case
            idx_sorted = np.argsort(galprop_cumprob)
            galprop = (
                self.one_point_lookup_table[ibin](galprop_cumprob[idx_sorted]))

        if self.correlation_strength[ibin] < 0:
            return galprop[::-1]
        else:
            return galprop


    def build_one_point_lookup_table(self, **kwargs):
        """
        Method computes lookup tables of the cumulative ``galprop`` PDF 
        defined by ``input_galaxy_table``. 

        Parameters 
        ----------
        input_galaxy_table : data table, required keyword argument 
            Astropy Table object storing the input galaxy population.  
            The conditional one-point functions of this population 
            will be used as inputs when building the primary behavior 
            of the `ConditionalAbunMatch` model. 

        prim_galprop_bins : array, required keyword argument 
            Array used to bin ``input_galaxy_table`` by ``prim_galprop_key``. 

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
        """ Method creates ``self.param_dict`` regulating the strength of 
        the correlation between sec_haloprop and galprop at each value of prim_galprop.  
        """
        
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
        else:
            self.param_dict = {'correlation_param1': 1.0}
        self._set_correlation_strength()

    def _set_correlation_strength(self):
        """ Method uses the current values in the param_dict to update the strength 
        of the correlation between sec_haloprop and galprop at each value of prim_galprop.  
        """

        if hasattr(self, 'correlation_strength_abcissa'):
            abcissa = self.correlation_strength_abcissa
            ordinates = [self.param_dict['correlation_param'+str(i+1)] for i in range(len(abcissa))]
            correlation_strength_spline = model_helpers.custom_spline(abcissa, ordinates, k=custom_len(abcissa)-1)
            self.correlation_strength = correlation_strength_spline(self.prim_galprop_bins)
        else:
            self.correlation_strength = np.repeat(self.param_dict['correlation_param1'], len(self.prim_galprop_bins))

        self.correlation_strength[self.correlation_strength > 1] = 1
        self.correlation_strength[self.correlation_strength <- 1] = -1

        self.correlation_strength = np.append(
            self.correlation_strength, self.correlation_strength[-1])

    def add_new_haloprops(self, galaxy_table):
        """ Method calls ``new_haloprop_func_dict`` to create new 
        halo properties as columns to the mock catalog, if applicable. 
        """
        if hasattr(self, 'new_haloprop_func_dict'):
            d = self.new_haloprop_func_dict
            for key, func in d.iteritems():
                if key not in galaxy_table.keys():
                    galaxy_table[key] = func(galaxy_table=galaxy_table)

























