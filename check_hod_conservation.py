#!/usr/bin/env python

import numpy as np
from halotools.empirical_models import Hearin15, Leauthaud11

def mean_mock_occupation(halo_table, galaxy_table, bins, key, gal_type):
    halo_counts = np.histogram(halo_table[key], bins=bins)[0].astype(float)
    galaxy_sample = galaxy_table[galaxy_table['gal_type'] == gal_type]
    galaxy_counts = np.histogram(galaxy_sample[key], bins=bins)[0].astype(float)
    return galaxy_counts/halo_counts




alexie = Leauthaud11()
aph_5050 = Hearin15()
aph_2080 = Hearin15(split = 0.2)

alexie.populate_mock()
aph_5050.populate_mock()
aph_2080.populate_mock()

bins = np.logspace(12., 14., 10)
mass = (bins[1:] + bins[:-1])/2.
key = 'halo_mvir'
gal_type = 'satellites'
alexie_nsat = mean_mock_occupation(alexie.mock.halo_table, alexie.mock.galaxy_table, bins, key, gal_type)
aph_5050_nsat = mean_mock_occupation(aph_5050.mock.halo_table, aph_5050.mock.galaxy_table, bins, key, gal_type)
aph_2080_nsat = mean_mock_occupation(aph_2080.mock.halo_table, aph_2080.mock.galaxy_table, bins, key, gal_type)

fracdiff_5050 = (aph_5050_nsat - alexie_nsat)/alexie_nsat
fracdiff_2080 = (aph_2080_nsat - alexie_nsat)/alexie_nsat

maxerr_5050 = np.abs(fracdiff_5050).max()
maxerr_2080 = np.abs(fracdiff_2080).max()

print("\n\nmaximum fractional difference for 50/50 case = %.3f" % maxerr_5050)
print("maximum fractional difference for 20/80 case = %.3f\n\n" % maxerr_2080)

if maxerr_2080 < 0.2:
    print("\a\a\a\a\a")
else:
    print("\a\a")

