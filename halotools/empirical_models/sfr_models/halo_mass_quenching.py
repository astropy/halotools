# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
    division, print_function, absolute_import)

__all__ = ['HaloMassInterpolQuenching']

from functools import partial
from copy import copy
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

from .. import model_defaults
from .. import model_helpers
from ..component_model_templates import BinaryGalpropInterpolModel

from ...utils.array_utils import custom_len
from ...custom_exceptions import HalotoolsError

class HaloMassInterpolQuenching(BinaryGalpropInterpolModel):
	pass