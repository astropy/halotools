#!/usr/bin/env python

import numpy as np 
from unittest import TestCase
from functools import partial

from ..table_utils import SampleSelector
from astropy.table import Table

class TestSampleSelector(TestCase):
	"""
	"""

	def test_split_sample(self):
		Npts = 10
		x = np.linspace(0, 9, Npts)
		d = {'x':x}
		self.t = Table(d)

		percentiles = 0.5
		result = SampleSelector.split_sample(table=self.t, key='x', percentiles = percentiles)

		assert len(result) == 2
		assert len(result[0]) == 5
		assert len(result[1]) == 5

		result0_sum = result[0]['x'].sum()
		correct_sum = np.sum([0, 1, 2, 3, 4])
		assert result0_sum == correct_sum

		result1_sum = result[1]['x'].sum()
		correct_sum = np.sum([5, 6, 7, 8, 9])
		assert result1_sum == correct_sum

		f = partial(SampleSelector.split_sample, table=self.t[0:3], key='x', 
			percentiles=[0.1, 0.2, 0.3, 0.4, 0.5])

		self.assertRaises(ValueError, f)



