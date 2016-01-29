#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..halo_mass_quenching import HaloMassInterpolQuenching

from ... import model_defaults

from ....custom_exceptions import HalotoolsError

__all__ = ['TestHaloMassInterpolQuenching']


class TestHaloMassInterpolQuenching(TestCase):
	pass
