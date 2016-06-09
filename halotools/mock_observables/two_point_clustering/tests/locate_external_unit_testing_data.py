""" This module is used to search the user's disk to see whether
data files are present to conduct unit-tests in which Halotools results
are compared against results obtained from independently-written code bases.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from astropy.config.paths import _find_home
halotools_cache_dirname = os.path.join(_find_home(), '.astropy', 'cache', 'halotools')
halotool_unit_testing_dirname = os.path.join(halotools_cache_dirname, 'unit_testing_files')


__all__ = ('tpcf_corrfunc_comparison_files_exist', 'wp_corrfunc_comparison_files_exist')


def tpcf_corrfunc_comparison_files_exist(return_fnames=False):
    """
    """
    aph_fname1 = os.path.join(halotool_unit_testing_dirname, 'sample1_position_array.npy')
    aph_fname2 = os.path.join(halotool_unit_testing_dirname, 'sample2_position_array.npy')
    aph_fname3 = os.path.join(halotool_unit_testing_dirname, 'rp_bins_array.npy')

    deep_fname1 = os.path.join(halotool_unit_testing_dirname,
        'sinha_corrfunc_results', 'sample1_position_array_xi.npy')
    deep_fname2 = os.path.join(halotool_unit_testing_dirname,
        'sinha_corrfunc_results', 'sample2_position_array_xi.npy')

    all_files_exist = (
        os.path.isfile(aph_fname1) & os.path.isfile(aph_fname2) & os.path.isfile(aph_fname3) &
        os.path.isfile(deep_fname1) & os.path.isfile(deep_fname2))

    if return_fnames is False:
        return all_files_exist
    else:
        return all_files_exist, aph_fname1, aph_fname2, aph_fname3, deep_fname1, deep_fname2


def wp_corrfunc_comparison_files_exist(return_fnames=False):
    """
    """
    aph_fname1 = os.path.join(halotool_unit_testing_dirname, 'sample1_position_array.npy')
    aph_fname2 = os.path.join(halotool_unit_testing_dirname, 'sample2_position_array.npy')
    aph_fname3 = os.path.join(halotool_unit_testing_dirname, 'rp_bins_array.npy')

    deep_fname1 = os.path.join(halotool_unit_testing_dirname,
        'sinha_corrfunc_results', 'sample1_position_array_wp.npy')
    deep_fname2 = os.path.join(halotool_unit_testing_dirname,
        'sinha_corrfunc_results', 'sample2_position_array_wp.npy')

    all_files_exist = (
        os.path.isfile(aph_fname1) & os.path.isfile(aph_fname2) & os.path.isfile(aph_fname3) &
        os.path.isfile(deep_fname1) & os.path.isfile(deep_fname2))

    if return_fnames is False:
        return all_files_exist
    else:
        return all_files_exist, aph_fname1, aph_fname2, aph_fname3, deep_fname1, deep_fname2
