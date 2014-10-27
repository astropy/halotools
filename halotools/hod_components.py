# -*- coding: utf-8 -*-
"""

This module contains the model components used 
by hod_designer to build composite HOD models 
by composing the behavior of the components. 

"""

import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings



