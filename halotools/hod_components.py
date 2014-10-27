# -*- coding: utf-8 -*-
"""

This module contains the model components used 
by hod_designer to build composite HOD models 
by composing the behavior of the components. 

"""

__all__ = ['Zheng07_centrals']


import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

import occupation_helpers as occuhelp




class Zheng07_centrals(object):
    """ Model for the occupation statistics of central galaxies, 
    taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys are 'logMmin_cen' and 'sigma_logM'

        Their best-fit parameter values provided in Table 1 of 
        Zheng et al. (2007) are pre-loaded into this class, and 
        can be accessed via the `published_parameters` method.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
        Default value is specified in the `~halotools.defaults` module.

    gal_type : string, optional
        Sets the key value used by `~halotools.hod_designer` and 
        `~halotools.hod_factory` to access the behavior of the methods 
        of this class. 

    """

    def __init__(self,parameter_dict=None,
        threshold=defaults.default_luminosity_threshold,
        gal_type='centrals'):

        self.gal_type = gal_type

        self.threshold = threshold
        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(self.threshold)
        else:
            self.parameter_dict = parameter_dict
        # Put parameter_dict keys in standard form
        correct_keys = self.published_parameters(self.threshold).keys()
        self.parameter_dict = occuhelp.format_parameter_keys(
            self.parameter_dict,correct_keys,self.gal_type)
        # get the new keys so that the methods know 
        # how to evaluate their functions
        self.logMmin_key = 'logMmin_'+self.gal_type
        self.sigma_logM_key = 'sigma_logM_'+self.gal_type


    def mean_occupation(self,logM):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        Returns
        -------
        mean_ncen : array
    
        Notes
        -------
        Mean number of central galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{cen} \\rangle_{M} = 
        \\frac{1}{2}\\left( 1 + 
        erf\\left( \\frac{log_{10}M - 
        log_{10}M_{min}}{\\sigma_{log_{10}M}} \\right) \\right)`

        """
        logM = np.array(logM)

        mean_ncen = 0.5*(1.0 + erf(
            (logM - self.parameter_dict[self.logMmin_key])
            /self.parameter_dict[self.sigma_logM_key]))

        return mean_ncen


    def published_parameters(self,threshold):
        """
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters 
        ----------

        threshold : float
            Luminosity threshold defining the SDSS sample to which Zheng et al. 
            fit their HOD model. Must be agree with one of the published values: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        Returns 
        -------

        parameter_dict : dict
            Dictionary of model parameters whose values have been set to 
            agree with the values taken from Table 1 of Zheng et al. 2007.

        """

        #Load tabulated data from Zheng et al. 2007, Table 1
        logMmin_array = [11.35,11.46,11.6,11.75,12.02,12.3,12.79,13.38,14.22]
        sigma_logM_array = [0.25,0.24,0.26,0.28,0.26,0.21,0.39,0.51,0.77]
        # define the luminosity thresholds corresponding to the above data
        threshold_array = np.arange(-22,-17.5,0.5)
        threshold_array = threshold_array[::-1]

        threshold_index = np.where(threshold_array==threshold)[0]
        if len(threshold_index)==1:
            parameter_dict = {
            'logMmin' : logMmin_array[threshold_index[0]],
            'sigma_logM' : sigma_logM_array[threshold_index[0]]
            }
        else:
            raise ValueError("Input luminosity threshold "
                "does not match any of the Table 1 values of Zheng et al. 2007 (arXiv:0703457).")

        return parameter_dict






















