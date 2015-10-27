# -*- coding: utf-8 -*-

"""
Module containing classes used to perform abundance matching (SHAM)

"""

import numpy as np

from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from warnings import warn
from functools import partial

from . import model_defaults
from . import model_helpers
from .smhm_models import PrimGalpropModel, LogNormalScatterModel, Moster13SmHm

from ..utils.array_utils import custom_len
from ..sim_manager import sim_defaults
from .. import sim_manager
from ..utils import array_utils 


__all__ = ['AbunMatchSmHm']

class AbunMatchSmHm(PrimGalpropModel):
    """
    """

    def __init__(self, galprop_name, galaxy_abundance_abcissa, galaxy_abundance_ordinates, 
        scatter_level = 0.2, **kwargs):
    
        kwargs['scatter_model'] = LogNormalScatterModel
        kwargs['scatter_abcissa'] = [12]
        kwargs['scatter_ordinates'] = [scatter_level]

        super(AbunMatchSmHm, self).__init__(galprop_name, **kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        return None







