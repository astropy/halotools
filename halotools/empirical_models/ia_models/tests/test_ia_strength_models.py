from ..ia_strength_models import RadialSatelliteAlignmentStrength, HaloMassCentralAlignmentStrength

import numpy as np
from astropy.table import Table

__all__ = ('test_alignment_strength_mass_dependence',
           'test_assign_central_alignment_strength',
           'test_alignment_strength_radial_dependence',
           'test_assign_satellite_alignment_strength')

# HaloMassCentralAlignmentStrength Tests
def test_alignment_strength_mass_dependence():
    """
    Almost a trivial test. The function is just a calculation with masking at the upper and lower bounds of the defined region.
    Test to see that the function returns expected values.
    Test in three pieces:
        1: all values are within (-0.99, 0.99)
        2: all values are >= 0.99 and get truncated to 0.99
        3: all values are <= -0.99 and get truncated to -0.99
    """
    strength_model = HaloMassCentralAlignmentStrength()
    m = 1e3 * np.linspace(1, 10, 20)

    # Test no truncation
    a = 0.02
    gamma = 0.7
    calculated = a*np.log10(m) + gamma
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    result = strength_model.alignment_strength_mass_dependence(m)
    assert( (result == calculated).all() )

    # Test all >= 0.99 and get returned as 0.99
    a = 0.2
    gamma = 0.7
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    result = strength_model.alignment_strength_mass_dependence(m)
    assert( (result == 0.99).all() )

    # Test all <= -0.99 and get returned as 0.99
    a = -0.5
    gamma = 0.01
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    result = strength_model.alignment_strength_mass_dependence(m)
    assert( (result == -0.99).all() )

def test_assign_central_alignment_strength():
    """
    Essentially a trivial test. Test that the masses given in a table for the centrals get calculated and stored in the table.
    """
    # Give the list of masses enough range to span past both bounds
    m = 1e3 * np.linspace(1,100,30)
    strength_model = HaloMassCentralAlignmentStrength()
    strength_model.param_dict['a'] = -1
    strength_model.param_dict['gamma'] = 4
    tab = Table([m], names=["halo_mvir"])
    tab["gal_type"] = "centrals"

    results = strength_model.alignment_strength_mass_dependence(tab["halo_mvir"])
    tab = strength_model.assign_central_alignment_strength(table=tab)
    assert( "central_alignment_strength" in tab.columns )
    assert( (results == tab["central_alignment_strength"]).all() )


# RadialSatellitesAlignmentStrength Tests
def test_alignment_strength_radial_dependence():
    """
    Test the alignment_strength_radial_dependence method in RadialSatelliteAlignmentStrength
    Not a very complex test since the function is just doing one equation and adjusting values that fall outside a given range
    Test in three pieces:
        1: all values are within (-0.99, 0.99)
        2: all values are >= 0.99 and get truncated to 0.99
        3: all values are <= -0.99 and get truncated to -0.99
    """

    strength_model = RadialSatelliteAlignmentStrength()

    r = np.linspace(0.1,1,20)

    # All of these values should be exactly what is calculated by a*(r**gamma)
    a = 0.5
    gamma = 0.7
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    results = strength_model.alignment_strength_radial_dependence(r)
    assert( ( results == ( a * (r**gamma) ) ).all() )

    # Now a test where all values evaluate to >= 0.99 and become 0.99
    a = 1
    gamma = -0.3
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    results = strength_model.alignment_strength_radial_dependence(r)
    assert( ( results == 0.99 ).all() )

    # Now a test where all values evaluate to <= -0.99 and become -0.99
    a = -1
    gamma = -0.3
    strength_model.param_dict['a'] = a
    strength_model.param_dict['gamma'] = gamma
    results = strength_model.alignment_strength_radial_dependence(r)
    assert( ( results == -0.99 ).all() )

def test_assign_satellite_alignment_strength():
    """
    Test if, given a table with the proper columns, the model adds a column with the calculated radially dependent alignment strengths.
    The alignment_Strenth_radial_dependence has a separate test to ensure it calculates properly, so we will use the result of that to test the added column
    """
    strength_model = RadialSatelliteAlignmentStrength()
    strength_model.param_dict['a'] = 0.6
    strength_model.param_dict['gamma'] = -0.1

    Lbox = [250., 250., 250.]
    halo_x = np.linspace(0,100,10)
    halo_y = np.linspace(0,100,10)
    halo_z = np.linspace(0,100,10)
    halo_x, halo_y, halo_z = np.meshgrid(halo_x,halo_y,halo_z)
    halo_x = halo_x.flatten()
    halo_y = halo_y.flatten()
    halo_z = halo_z.flatten()

    additions = np.linspace(1,20,10)
    plus_x, plus_y, plus_z = np.meshgrid(additions, additions, additions)
    plus_x = plus_x.flatten()
    plus_y = plus_y.flatten()
    plus_z = plus_z.flatten()

    x, y, z = np.array( [ halo_x, halo_y, halo_z ] ) + np.array( [ plus_x, plus_y, plus_z ] )

    r_vir = np.linspace( 2, 20, 20 )
    r_vir = np.tile(r_vir, int(len(x)/len(r_vir)))

    tab = Table(data=[halo_x, halo_y, halo_z, x, y, z, r_vir], names=["halo_x", "halo_y", "halo_z", "x", "y", "z", "halo_rvir"])
    tab["gal_type"] = "satellites"
    tab = strength_model.assign_satellite_alignment_strength(table=tab, Lbox=Lbox)

    assert( "satellite_alignment_strength" in tab.columns )

    # Since we have a separate test to show that the alignment_Strength_radial_dependence works, use that to generate truth values
    halo_pos = np.array( [ tab["halo_x"], tab["halo_y"], tab["halo_z"] ] ).T
    pos = np.array( [ tab["x"], tab["y"], tab["z"] ] ).T
    r = np.sqrt( np.sum( (halo_pos - pos)**2, axis=1 ) )
    compare = strength_model.alignment_strength_radial_dependence( r / tab["halo_rvir"] )
    assert( (compare == tab["satellite_alignment_strength"]).all() )

    # Now check that it still works if we wrap around the boundary (using PBC)
    tab = Table( data=[ [249], [249], [249], [1], [1], [1], [4], ["satellites"] ], names=["halo_x", "halo_y", "halo_z", "x", "y", "z", "halo_rvir", "gal_type"] )
    r = np.array([np.sqrt(4+4+4)])
    tab = strength_model.assign_satellite_alignment_strength(table=tab, Lbox=Lbox)
    compare = strength_model.alignment_strength_radial_dependence( r / tab["halo_rvir"] )
    assert( (compare == tab["satellite_alignment_strength"]).all() )
