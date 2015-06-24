#!/usr/bin/env python
import numpy as np

from ..hod_components import Leauthaud11Cens

from .. import model_defaults

from astropy.table import Table
from copy import copy

__all__ = ['test_Leauthaud11Cens']

def test_Leauthaud11Cens():
	""" Function to test 
	`~halotools.empirical_models.Leauthaud11Cens`. 
	"""

	model = Leauthaud11Cens()






