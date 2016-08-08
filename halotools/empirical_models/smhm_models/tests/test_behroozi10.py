"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from ...smhm_models import Behroozi10SmHm

__all__ = ('test_behroozi10_smhm_z01', 'test_behroozi10_smhm_z05',
    'test_behroozi10_smhm_z1')


def test_behroozi10_smhm_z01():
    """ The arrays ``logmh_z01`` and ``logmratio_z01`` were provided by Peter Behroozi
    via private communication. These quantities are in the h=0.7 units adopted in
    Behroozi+10. This test function treats these arrays as truth,
    and enforces that the result computed by Halotools agrees with them.
    """
    model = Behroozi10SmHm()

    logmh_z01 = np.array(
        [10.832612, 10.957612, 11.082612, 11.207612,
        11.332612, 11.457612, 11.582612, 11.707612,
        11.832612, 11.957612, 12.082612, 12.207612,
        12.332612, 12.457612, 12.582612, 12.707612,
        12.832612, 12.957612, 13.082612, 13.207612,
        13.332612, 13.457612, 13.582612, 13.707612,
        13.832612, 13.957612, 14.082612, 14.207612,
        14.332612, 14.457612, 14.582612, 14.707612,
        14.832612, 14.957612, 15.082612, 15.207612])

    logmratio_z01 = np.array(
        [-2.532613, -2.358159, -2.184308, -2.012586,
        -1.847878, -1.702718, -1.596036, -1.537164,
        -1.518895, -1.529237, -1.558904, -1.601876,
        -1.654355, -1.713868, -1.778768, -1.84792,
        -1.920522, -1.995988, -2.07388, -2.153878,
        -2.235734, -2.319242, -2.404256, -2.490647,
        -2.578321, -2.66718, -2.757161, -2.848199,
        -2.94024, -3.033235, -3.127133, -3.221902,
        -3.317498, -3.413892, -3.511041, -3.608918])

    halo_mass_z01 = (10.**logmh_z01)*model.littleh
    logmratio_z01 = np.log10((10.**logmratio_z01)*model.littleh)

    z01_sm = model.mean_stellar_mass(prim_haloprop=halo_mass_z01, redshift=0.1)
    z01_ratio = z01_sm / halo_mass_z01
    z01_result = np.log10(z01_ratio)
    assert np.allclose(z01_result, logmratio_z01, rtol=0.02)


def test_behroozi10_smhm_z05():
    """ The arrays ``logmh_z01`` and ``logmratio_z01`` were provided by Peter Behroozi
    via private communication. These quantities are in the h=0.7 units adopted in
    Behroozi+10. This test function treats these arrays as truth,
    and enforces that the result computed by Halotools agrees with them.
    """
    model = Behroozi10SmHm()

    logmh_z05 = np.array([
        11.066248, 11.191248, 11.316248, 11.441248,
        11.566248, 11.691248, 11.816248, 11.941248,
        12.066248, 12.191248, 12.316248, 12.441248,
        12.566248, 12.691248, 12.816248, 12.941248,
        13.066248, 13.191248, 13.316248, 13.441248,
        13.566248, 13.691248, 13.816248, 13.941248,
        14.066248, 14.191248, 14.316248, 14.441248,
        14.566248, 14.691248]
        )

    logmratio_z05 = np.array([
        -2.375180, -2.183537, -2.015065, -1.879960,
        -1.782708, -1.720799, -1.688169, -1.678521,
        -1.686669, -1.708703, -1.741731, -1.783616,
        -1.832761, -1.887952, -1.948255, -2.012940,
        -2.081414, -2.153203, -2.227921, -2.305249,
        -2.384912, -2.466680, -2.550359, -2.635785,
        -2.722806, -2.811296, -2.901139, -2.992246,
        -3.084516, -3.177873]
        )

    halo_mass_z05 = (10.**logmh_z05)*model.littleh
    logmratio_z05 = np.log10((10.**logmratio_z05)*model.littleh)

    z05_sm = model.mean_stellar_mass(prim_haloprop=halo_mass_z05, redshift=0.5)
    z05_ratio = z05_sm / halo_mass_z05
    z05_result = np.log10(z05_ratio)
    assert np.allclose(z05_result, logmratio_z05, rtol=0.02)


def test_behroozi10_smhm_z1():
    """ The arrays ``logmh_z01`` and ``logmratio_z01`` were provided by Peter Behroozi
    via private communication. These quantities are in the h=0.7 units adopted in
    Behroozi+10. This test function treats these arrays as truth,
    and enforces that the result computed by Halotools agrees with them.
    """
    model = Behroozi10SmHm()

    logmh_z1 = np.array(
        [11.368958, 11.493958, 11.618958, 11.743958,
        11.868958, 11.993958, 12.118958, 12.243958,
        12.368958, 12.493958, 12.618958, 12.743958,
        12.868958, 12.993958, 13.118958, 13.243958,
        13.368958, 13.493958, 13.618958, 13.743958,
        13.868958, 13.993958, 14.118958, 14.243958]
        )

    logmratio_z1 = np.array(
        [-2.145909, -2.020974, -1.924020, -1.852937,
        -1.804730, -1.776231, -1.764455, -1.766820,
        -1.781140, -1.805604, -1.838727, -1.879292,
        -1.926290, -1.978890, -2.036405, -2.098245,
        -2.163930, -2.233045, -2.305230, -2.380185,
        -2.457643, -2.537377, -2.619191, -2.702901]
        )

    halo_mass_z1 = (10.**logmh_z1)*model.littleh
    logmratio_z1 = np.log10((10.**logmratio_z1)*model.littleh)

    z1_sm = model.mean_stellar_mass(prim_haloprop=halo_mass_z1, redshift=1.0)
    z1_ratio = z1_sm / halo_mass_z1
    z1_result = np.log10(z1_ratio)
    assert np.allclose(z1_result, logmratio_z1, rtol=0.02)
