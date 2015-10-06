# -*- coding: utf-8 -*-
"""

Common functions applied to halo catalogs. 

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from warnings import warn

from .match import crossmatch

def host_status(halos):
	"""
	Function divides input ``halos`` into a few categories:

	* ``true_host_halo``: a halo whose center is not presently found within the halo-finder-determined boundary of a more massive parent halo, and where this condition has always held true throughout its main progenitor history. 

	* ``ejected_host_halo``: a halo whose center is not presently found within the halo-finder-determined boundary of a more massive parent halo, but where this condition has been violated at some point in its main progenitor history. 

	* ``subhalo``: a halo whose center is presently found within the halo-finder-determined boundary of a more massive parent halo, and where this more massive parent halo is a host halo (either true or ejected) that appears within the input ``halos`` catalog that was passed to the `host_status` function.

	* ``subhalo_nohost``: a halo whose center is presently found within the halo-finder-determined boundary of a more massive parent halo, and where this more massive parent halo does not appear within the input ``halos`` catalog that was passed to the `host_status` function.

	* ``merging_subhalo``: a halo whose center is presently found within the halo-finder-determined boundary of a more massive parent halo, and where this more massive parent halo is itself a ``subhalo``. Only happens in rare cases where the more massive parent halo resides inside, but in the outskirts, of another parent. 

	Parameters 
	----------
	halos : `~astropy.table.Table`
		Catalog of halos with the following keys: ``halo_id``, ``halo_hostid``. 

	Returns 
	--------
	output : array 
		Array of strings giving the halo designation. 

	"""

	output = np.zeros(len(halos), dtype=object)

	host_halo_mask = halos['halo_hostid'] == halos['halo_id']
	host_halos = halos[host_halo_mask]

	if ('halo_scale_factor_firstacc' in halos.keys()) & ('halo_scale_factor' in halos.keys()):
		# implement workaround of the fact that 'halo_scale_factor_firstacc' was 
		# artificially truncated at 3 decimals in the original ascii, but not 'halo_scale_factor'
		scale_factor_diff = host_halos['halo_scale_factor'] - host_halos['halo_scale_factor_firstacc'] 
		ejected_mask = scale_factor_diff > 0.01
		output[host_halo_mask] = np.where(ejected_mask, 'ejected_host_halo', 'true_host_halo')
	else:
		warn("Unable to determine whether host halo was ejected because "
			"either ``halo_scale_factor_firstacc`` or ``halo_scale_factor`` "
			"key is missing from input halo catalog")
		output[host_halo_mask] = 'true_host_halo'

	sub_halo_mask = np.invert(host_halo_mask)
	sub_halos = halos[sub_halo_mask]
	output[sub_halo_mask] = 'subhalo_nohost'
	output_subhalo_subarray = output[sub_halo_mask]

	a_ind, b_ind = crossmatch(sub_halos['halo_hostid'], host_halos['halo_id'])
	output_subhalo_subarray[a_ind] = 'subhalo'

	a_ind, b_ind = crossmatch(sub_halos['halo_hostid'], sub_halos['halo_id'])
	output_subhalo_subarray[a_ind] = 'merging_subhalo'
	output[sub_halo_mask] = output_subhalo_subarray

	return output.astype(str)






