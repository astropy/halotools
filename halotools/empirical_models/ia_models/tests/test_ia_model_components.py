import numpy as np
from ..ia_model_components import RandomAlignment, CentralAlignment, SatelliteAlignment, RadialSatelliteAlignment, SubhaloAlignment
from ..ia_model_components import axes_correlated_with_z, axes_correlated_with_input_vector, is_z
from astropy.table import Table
from ....utils.mcrotations import rotate_vector_collection, random_unit_vectors_3d
from ....utils import normalized_vectors, angles_between_list_of_vectors
from ....utils.vector_utilities import elementwise_dot

__all__ = ('test_random_alignment_assign_orientation',
           'test_central_alignment_assign_central_orientation',
           'test_satellite_alignment_assign_satellite_orientation',
           'test_radial_satellite_alignment_assign_satellite_orientation',
           'test_radial_satellite_alignment_get_radial_vector',
           'test_bad_axis_radial_alignment',
           'test_subhalo_alignment_orient_false_subhalo',
           'test_subhalo_alignment_get_rotation_matrix',
           'test_subhalo_alignment_assign_satellite_orientation',
           'test_axes_correlated_with_z',
           'test_axes_correlated_with_input_vector',
           'test_is_z')

# RandomAlignment Tests
# assign_orientation
def test_random_alignment_assign_orientation():
    """
    The randomness of the vectors is taken care of by other functions, so this function tests whether or not the three random axes get placed properly in the table
    """
    alignment = RandomAlignment()
    axis_labels = [ "galaxy_axisA_x", "galaxy_axisA_y", "galaxy_axisA_z", "galaxy_axisB_x", "galaxy_axisB_y", "galaxy_axisB_z", "galaxy_axisC_x", "galaxy_axisC_y", "galaxy_axisC_z" ]

    tab = Table( [["centrals","centrals","centrals","satellites","satellites","satellites","satellites","satellites"]], names=["gal_type"] )
    tab = alignment.assign_orientation(table=tab)
    for label in axis_labels:
        # Make sure each columns has been added
        assert( label in tab.columns )
    
    satellites =  tab[ tab["gal_type"] == "satellites" ]

    # Assigning orientations should assign to centrals only by default
    for label in axis_labels:
        assert( ( satellites[label] == 0.0 ).all() )

    # Refresh the table and assign only to satellites now
    alignment.gal_type = "satellites"
    tab = Table( [["centrals","centrals","centrals","satellites","satellites","satellites","satellites","satellites"]], names=["gal_type"] )
    tab = alignment.assign_orientation(table=tab)
    centrals = tab[ tab["gal_type"] == "centrals" ]
    for label in axis_labels:
        # Make sure each columns has been added
        assert( label in tab.columns )
        assert( (centrals[label] == 0.0).all() )

    # Test that, even without gal_type, columns get added
    tab = Table( [["A","B","B","B","A"]], names=["Stuff"] )
    tab = alignment.assign_orientation(table=tab)
    for label in axis_labels:
        # Make sure each columns has been added
        assert( label in tab.columns )

# CentralAlignment Tests
# assign_central_orientation
def test_central_alignment_assign_central_orientation():
    """
    Test central_alignment. Test that aligning this way will align central galaxies with respect to the host halo major axis.
    Satellite galaxies should be left untouched.
    """
    alignment = CentralAlignment()
    axis_labels = [ "galaxy_axisA_x", "galaxy_axisA_y", "galaxy_axisA_z", "galaxy_axisB_x", "galaxy_axisB_y", "galaxy_axisB_z", "galaxy_axisC_x", "galaxy_axisC_y", "galaxy_axisC_z" ]

    halo_axes = np.array( [ [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1], [0,0,0], [0,0,0], [0,0,0] ] ).astype(float)
    # Normalize the axes
    mags = np.sqrt( np.sum( halo_axes**2, axis=1 ) )
    halo_axes /= np.array( [mags, mags, mags] ).T
    halo_axes = np.nan_to_num(halo_axes)
    
    gal_types = [ "centrals", "centrals", "centrals", "centrals", "centrals", "centrals", "centrals", "satellites", "satellites", "satellites" ]
    tab = Table( [ gal_types, halo_axes[:,0], halo_axes[:,1], halo_axes[:,2] ], names=["gal_type", "halo_axisA_x", "halo_axisA_y", "halo_axisA_z"] )
    tab = alignment.assign_central_orientation(table=tab)

    centrals =  tab[ tab["gal_type"] == "centrals" ]
    satellites =  tab[ tab["gal_type"] == "satellites" ]

    for label in axis_labels:
        # Test that all columns have been added and that the satellites have not been given alignments
        assert( label in tab.columns )
        assert( ( satellites[ label ] == 0.0 ).all() )

    # Test that the centrals are perfectly aligned with the host
    galaxy_orientations = np.array( [ centrals["galaxy_axisA_x"], centrals["galaxy_axisA_y"], centrals["galaxy_axisA_z"] ] ).T
    halo_orientations = np.array( [ centrals["halo_axisA_x"], centrals["halo_axisA_y"], centrals["halo_axisA_z"] ] ).T

    # Because of numerical artefacts, consider it perfect if the difference is less than 1e-14
    # Also, the orientations are the same if rotated 180 degrees, so take (halo_orientation) - (relative sign)*(galaxy_orientation)
    galaxy_signs = np.sign( galaxy_orientations )
    halo_signs = np.sign( halo_orientations )
    relative_signs = galaxy_signs*halo_signs

    diffs = halo_orientations - ( relative_signs * galaxy_orientations )
    assert( ( abs(diffs) < 1e-14 ).all() )

# SatelliteAlignment Tests
# assign_satellite_orientation
def test_satellite_alignment_assign_satellite_orientation():
    """
    Test that this will align satellite galaxies with respect to the host halo major axis
    Central galaxies should be left untouched
    """
    alignment = SatelliteAlignment()
    axis_labels = [ "galaxy_axisA_x", "galaxy_axisA_y", "galaxy_axisA_z", "galaxy_axisB_x", "galaxy_axisB_y", "galaxy_axisB_z", "galaxy_axisC_x", "galaxy_axisC_y", "galaxy_axisC_z" ]

    halo_axes = np.array( [ [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1], [0,0,0], [0,0,0], [0,0,0] ] ).astype(float)
    # Normalize the axes
    mags = np.sqrt( np.sum( halo_axes**2, axis=1 ) )
    halo_axes /= np.array( [mags, mags, mags] ).T
    halo_axes = np.nan_to_num(halo_axes)
    
    gal_types = [ "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "centrals", "centrals", "centrals" ]
    tab = Table( [ gal_types, halo_axes[:,0], halo_axes[:,1], halo_axes[:,2] ], names=["gal_type", "halo_axisA_x", "halo_axisA_y", "halo_axisA_z"] )
    tab = alignment.assign_satellite_orientation(table=tab)

    centrals =  tab[ tab["gal_type"] == "centrals" ]
    satellites =  tab[ tab["gal_type"] == "satellites" ]

    for label in axis_labels:
        # Test that all columns have been added and that the satellites have not been given alignments
        assert( label in tab.columns )
        assert( ( centrals[ label ] == 0.0 ).all() )

    # Test that the centrals are perfectly aligned with the host
    galaxy_orientations = np.array( [ satellites["galaxy_axisA_x"], satellites["galaxy_axisA_y"], satellites["galaxy_axisA_z"] ] ).T
    halo_orientations = np.array( [ satellites["halo_axisA_x"], satellites["halo_axisA_y"], satellites["halo_axisA_z"] ] ).T

    # Because of numerical artefacts, consider it perfect if the difference is less than 1e-14
    # Also, the orientations are the same if rotated 180 degrees, so take (halo_orientation) - (relative sign)*(galaxy_orientation)
    galaxy_signs = np.sign( galaxy_orientations )
    halo_signs = np.sign( halo_orientations )
    relative_signs = galaxy_signs*halo_signs

    diffs = halo_orientations - ( relative_signs * galaxy_orientations )
    assert( ( abs(diffs) < 1e-14 ).all() )

# RadialSatelliteAlignment Tests
# assign_satellite_orientation
# get_radial_vector
def test_radial_satellite_alignment_assign_satellite_orientation():
    """
    Test that assign_satellite_orientation will place new axes in the table that have been aligned with respect to the radial vectors between two positions
    """
    Lbox = [250., 250., 250.]
    alignment = RadialSatelliteAlignment(satellite_alignment_strength=1.0, Lbox=Lbox)
    axes = [ "galaxy_axisA_x", "galaxy_axisA_y", "galaxy_axisA_z" ]

    # Positions including two that wrap around the given Lbox
    galaxy_positions = np.array( [ [0,0.5,1], [0,1,0], [1,0,0], [1,1,1], [1,0,1], [2,1,3], [5,6,7], [1,1,1], [249, 249, 249], [3,3,3] ] ).astype(float)
    host_positions = np.array( [ [0,0,0], [0,0,0], [0,0,0], [0,0,0], [1,1,1], [3,3,1], [1,0,2], [249, 249, 249], [1,1,1], [3,3,3] ] ).astype(float)
    gal_types = np.array( [ "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "satellites", "centrals" ] )

    # These will be the normalized unit vectors along the radial direction for just the satellite galaxies
    #true_r_vecs = np.array( [ [0,0,1], [0,1,0], [1,0,0], [1,1,1], [0,0,-1], [-1,-2,2], [4,6,5], [2,2,2], [-2,-2,-2] ] ).astype(float)
    #true_r_mags = np.array( [ 1., 1., 1., np.sqrt(3), 1, 3., np.sqrt(77), np.sqrt(12), np.sqrt(12) ] ).astype(float)
    #true_r_vecs /= np.array( [ true_r_mags, true_r_mags, true_r_mags ] ).T

    tab = Table( np.hstack((galaxy_positions, host_positions)), names=["x","y","z","halo_x","halo_y","halo_z"] )
    tab["gal_type"] = gal_types

    tab = alignment.assign_satellite_orientation(table=tab)

    centrals = tab[ tab["gal_type"] == "centrals" ]
    satellites = tab[ tab["gal_type"] == "satellites" ]

    # The function used here is tested elsewhere
    true_r_vecs, true_r_mags = alignment.get_radial_vector(table=satellites)
    true_r_vecs /= np.array( [ true_r_mags, true_r_mags, true_r_mags ] ).T

    for axis in axes:
        assert( ( centrals[axis] == 0.0 ).all() )
    
    r_vecs = np.array( [ satellites["galaxy_axisA_x"], satellites["galaxy_axisA_y"], satellites["galaxy_axisA_z"] ] ).T
    # Because of numerical issues, the agreement won't be perfect
    # And we don't care about the relative negative sign since it's symmetric about 180 degree
    relative_sign = np.sign(r_vecs)*np.sign(true_r_vecs)
    diff = true_r_vecs - (relative_sign * r_vecs)
    assert( ( diff < 1e-14 ).all() )

def test_radial_satellite_alignment_get_radial_vector():
    """
    Test that the RadialSatelliteAlignment module can generate the proper radial vectors and magnitudes given two coordinates.
    Include the possibility of wrapping around the Lbox
    """
    alignment = RadialSatelliteAlignment()
    Lbox = [250., 250., 250.]
    # Positions including two that wrap around the given Lbox
    galaxy_positions = np.array( [ [0,0,1], [0,1,0], [1,0,0], [1,1,1], [1,1,0], [2,1,3], [5,6,7], [1,1,1], [249, 249, 249] ] ).astype(float)
    host_positions = np.array( [ [0,0,0], [0,0,0], [0,0,0], [0,0,0], [1,1,1], [3,3,1], [1,0,2], [249, 249, 249], [1,1,1] ] ).astype(float)

    # True values that should return from the function being tested
    true_r_vecs = np.array( [ [0,0,1], [0,1,0], [1,0,0], [1,1,1], [0,0,-1], [-1,-2,2], [4,6,5], [2,2,2], [-2,-2,-2] ] ).astype(float)
    true_r_mags = np.array( [ 1., 1., 1., np.sqrt(3), 1, 3., np.sqrt(77), np.sqrt(12), np.sqrt(12) ] ).astype(float)

    tab = Table( np.hstack((galaxy_positions, host_positions)), names=["x","y","z","halo_x","halo_y","halo_z"] )

    r_vecs, r_mags = alignment.get_radial_vector( table=tab, Lbox=Lbox )
    assert( ( r_vecs == true_r_vecs ).all() )
    assert( ( r_mags == true_r_mags ).all() )

def test_bad_axis_radial_alignment():
    """
    Test the bug fix where a single bad axis used to force all axes to be the same random unit vector.
    This test is in reference to the bug fixed in commit with the git hash caac5c87af9951f29b71ba62c7a31d21f91bdeb5
    """

    # Here, the 2 index axis is bad while 0,1,3,4,5 are fine
    # with the mask, those good axes should remain unchanged while index 2 should be replaced
    axes = np.array( [ [1,1,1], [1,0,1], [np.nan, 0, 2], [0.5,0,2], [0,0,1], [2,1,1] ] )
    mags = np.sqrt( np.sum( axes**2, axis=1 ) )
    axes /= np.array( [ mags, mags, mags ] ).T

    # Each of these will undergo a different treatment
    # The first will use the old code, the second the new
    old_result = np.array(axes)
    new_result = np.array(axes)

    # Original method, the one with the bug
    mask = (~np.isfinite(np.sum(np.prod(old_result, axis=-1))))
    N_bad_axes = np.sum(mask)
    old_result[mask,:] = random_unit_vectors_3d(N_bad_axes)

    # New method without the np.sum call
    mask = (~np.isfinite(np.prod(new_result, axis=-1)))
    N_bad_axes = np.sum(mask)
    new_result[mask,:] = random_unit_vectors_3d(N_bad_axes)

    bad_mask = np.repeat(True, len(axes))
    bad_mask[2] = False
    assert( ( axes[bad_mask] == new_result[bad_mask] ).all() )

    assert( not np.isnan(new_result).any() )
    assert( not np.isnan(old_result).any() )

    # Show that all axes have been replaced with the same result using the old method
    first_row = old_result[0,:]
    for row in old_result:
        assert( (first_row == row).all() )

# SubhaloAlignment Tests
# _orient_false_subhalo
# _get_rotation_matrix
# assign_satellite_orientation
def test_subhalo_alignment_orient_false_subhalo():
    """
    Test that the false subhalos in a galaxy_table can be rotated (positions, orientations, and velocities) such that the rotated values are the same with respect to the new hst halo
    as they old values were with respect to the original host halo
    """
    class FakeHalocat:
        # Dummy class to give the same halocat.halo_table structure expected by the function being tested
        def __init__(self):
            halo_id = np.array( [ 1, 2, 10 ] )
            halo_hostid = np.array( [ 1, 2, 1 ] )
            halo_x = np.array( [ 0, 1, 0.5 ] ).astype(float)
            halo_y = np.array( [ 0, 1, 0.5 ] ).astype(float)
            halo_z = np.array( [ 0, 1, 0.5 ] ).astype(float)
            halo_axes = normalized_vectors( np.array( [ [1,0,1], [0,0,1], [-1,1,0] ] ).astype(float) )
            halo_axisA_x, halo_axisA_y, halo_axisA_z = halo_axes.T
            halo_vx = np.array( [ 0, 1, 1 ] ).astype(float)
            halo_vy = np.array( [ 1, 1, 0 ] ).astype(float)
            halo_vz = np.array( [ 1, 1, 0 ] ).astype(float)

            self.halo_table = Table( [ halo_id, halo_hostid, halo_x, halo_y, halo_z, halo_vx, halo_vy, halo_vz, halo_axisA_x, halo_axisA_y, halo_axisA_z ],
                                    names=[ "halo_id", "halo_hostid", "x", "y", "z", "vx", "vy", "vz", "halo_axisA_x", "halo_axisA_y", "halo_axisA_z" ] )
    
    halocat = FakeHalocat()

    # make an altered version of the halocat table where we place the subhalo (from host 1) into host 2
    tab = halocat.halo_table.copy()
    tab = tab[ tab["halo_id"] != 1 ]
    tab["halo_hostid"][ tab["halo_id"] == 10 ] = 2
    for label in set(tab.columns):
        if label == "halo_id" or label == "halo_hostid":
            continue
        if "halo" in label:
            tab[ "sub"+label ] = np.array( tab[label] )
    tab["real_subhalo"] = False
    tab["gal_type"] = np.array(["centrals","satellites"])
    
    original_host_row = halocat.halo_table[ halocat.halo_table["halo_id"] == 1 ]
    original_subhalo_row = tab[ tab["halo_id"] == 10 ]
    
    # Get the original angles between each property and the host major axis
    og_host_axis = np.array( [ original_host_row["halo_axisA_x"], original_host_row["halo_axisA_y"], original_host_row["halo_axisA_z"] ] ).T

    og_subhalo_pos = np.array( [ original_subhalo_row["x"], original_subhalo_row["y"], original_subhalo_row["z"] ] ).T
    og_subhalo_v = np.array( [ original_subhalo_row["vx"], original_subhalo_row["vy"], original_subhalo_row["vz"] ] ).T
    og_subhalo_axis = np.array( [ original_subhalo_row["subhalo_axisA_x"], original_subhalo_row["subhalo_axisA_y"], original_subhalo_row["subhalo_axisA_z"] ] ).T

    og_angles = np.hstack( [ angles_between_list_of_vectors(og_host_axis, og_subhalo_axis), angles_between_list_of_vectors(og_host_axis, og_subhalo_pos), 
                                angles_between_list_of_vectors(og_host_axis, og_subhalo_v) ] )
    
    # Get current misalignment angles with new host halo
    new_host_row = tab[ tab["halo_id"] == 2 ]
    new_subhalo_row = tab[ tab["halo_id"] == 10 ]

    new_host_axis = np.array( [ new_host_row["halo_axisA_x"], new_host_row["halo_axisA_y"], new_host_row["halo_axisA_z"] ] ).T

    new_subhalo_pos = np.array( [ new_subhalo_row["x"], new_subhalo_row["y"], new_subhalo_row["z"] ] ).T
    new_subhalo_v = np.array( [ new_subhalo_row["vx"], new_subhalo_row["vy"], new_subhalo_row["vz"] ] ).T
    new_subhalo_axis = np.array( [ new_subhalo_row["subhalo_axisA_x"], new_subhalo_row["subhalo_axisA_y"], new_subhalo_row["subhalo_axisA_z"] ] ).T

    new_angles = np.hstack( [ angles_between_list_of_vectors(new_host_axis, new_subhalo_axis), angles_between_list_of_vectors(new_host_axis, new_subhalo_pos), 
                                angles_between_list_of_vectors(new_host_axis, new_subhalo_v) ] )
    
    # The misalignment angles should be different
    # In this case, the difference in angle just has to be "large"
    assert( ~( abs(og_angles - new_angles) < np.pi/4 ).all() )

    # Now reorient false subhalos
    alignment = SubhaloAlignment(halocat=halocat)
    alignment._orient_false_subhalo(table=tab)

    # And check again
    new_host_row = tab[ tab["halo_id"] == 2 ]
    new_subhalo_row = tab[ tab["halo_id"] == 10 ]

    new_host_axis = np.array( [ new_host_row["halo_axisA_x"], new_host_row["halo_axisA_y"], new_host_row["halo_axisA_z"] ] ).T

    new_subhalo_pos = np.array( [ new_subhalo_row["x"], new_subhalo_row["y"], new_subhalo_row["z"] ] ).T
    new_subhalo_v = np.array( [ new_subhalo_row["vx"], new_subhalo_row["vy"], new_subhalo_row["vz"] ] ).T
    new_subhalo_axis = np.array( [ new_subhalo_row["subhalo_axisA_x"], new_subhalo_row["subhalo_axisA_y"], new_subhalo_row["subhalo_axisA_z"] ] ).T

    new_angles = np.hstack( [ angles_between_list_of_vectors(new_host_axis, new_subhalo_axis), angles_between_list_of_vectors(new_host_axis, new_subhalo_pos), 
                                angles_between_list_of_vectors(new_host_axis, new_subhalo_v) ] )
    
    # Check that the new angles are the same with respect to the new axis as they were with respect to the old
    assert( ( abs( og_angles - new_angles ) < 1e-14 ).all() )

def test_subhalo_alignment_get_rotation_matrix():
    """
    Test that proper rotation matrixes can be found to rotate one vector into another
    """
    alignment = SubhaloAlignment(halocat=1)
    axes_a = normalized_vectors( np.array( [ [1,5,2], [6,2,7], [1,1,1], [0,0,1] ] ) )
    axes_b = normalized_vectors( np.array( [ [0,0,1], [1,0,0], [3,2,1], [1,1,1] ] ) )

    matrices = [ alignment._get_rotation_matrix( axes_a[i], axes_b[i]) for i in range(len(axes_a) ) ]
    rotated = rotate_vector_collection(matrices, axes_a)

    for i in range(len(rotated)):
        assert( ( abs(rotated[i] - axes_b[i]) < 1e-14 ).all() )

def test_subhalo_alignment_assign_satellite_orientation():
    """
    Test that the SubhaloAlignment module adds columns for galaxy axes and that the major axis aligns with respect to the subhalo major axis.
    It should also leave centrals untouched.
    """
    # Use dummy entry for halocat since it's not needed here (I'm disabling rotate relative for simplicity as it has no effect on the galaxies' alignments to their subhalos)
    alignment = SubhaloAlignment(satellite_alignment_strength=1.0, halocat=1, rotate_relative=False)

    subhalo_axisA_x, subhalo_axisA_y, subhalo_axisA_z = normalized_vectors( [ [1,1,1], [1,0,1], [3,2,0], [7,3,4], [0,1,9], [1,3,3], [8,7,2] ] ).T
    gal_type = np.array( [ "centrals", "centrals", "satellites", "satellites", "satellites", "satellites", "satellites" ] )

    tab = Table( [ subhalo_axisA_x, subhalo_axisA_y, subhalo_axisA_z, gal_type ], names=[ "subhalo_axisA_x", "subhalo_axisA_y", "subhalo_axisA_z", "gal_type" ] )
    tab = alignment.assign_satellite_orientation(table=tab)

    centrals = tab[ tab["gal_type"] == "centrals" ]
    satellites = tab[ tab["gal_type"] == "satellites" ]

    # Centrals should be left untouched while satellites should perfectly match the subhalo
    cen_axes = np.array( [ centrals["galaxy_axisA_x"], centrals["galaxy_axisA_y"], centrals["galaxy_axisA_z"] ] ).T
    sat_galaxy_axes = np.array( [ satellites["galaxy_axisA_x"], satellites["galaxy_axisA_y"], satellites["galaxy_axisA_z"] ] ).T
    sat_halo_axes = np.array( [ satellites["subhalo_axisA_x"], satellites["subhalo_axisA_y"], satellites["subhalo_axisA_z"] ] ).T

    assert( ( cen_axes == 0.0 ).all() )

    # Rotation by 180 degrees is irrelevant as there's symmetry. So a relative sign flip between the two doesn't matter
    relative_sign = np.sign(sat_galaxy_axes) * np.sign(sat_halo_axes)
    assert( ( abs( sat_halo_axes - ( relative_sign * sat_galaxy_axes ) ) < 1e-14 ).all() )

# General Tests
def test_axes_correlated_with_z():
    N = 10000
    z = np.array([0,0,1])

    # All should be perfectly aligned with z
    results = abs( axes_correlated_with_z( np.ones(N) ) )
    assert( ( np.dot( results, z ) == 1 ).all() )

    # None should have any component aligned with z
    results = abs( axes_correlated_with_z( -np.ones(N) ) )
    assert( ( np.dot( results, z ) == 0 ).all() )

    # Higher alignment with higher value
    results = abs( axes_correlated_with_z( 0.8 * np.ones(N) ) )
    val1 = np.mean( np.dot( results,z ) )
    results = abs( axes_correlated_with_z( np.zeros(N) ) )
    val2 = np.mean( np.dot( results,z ) )

    # With this many values, the lower alignment strength should average a smaller dot product
    assert( val2 < val1 )

def test_axes_correlated_with_input_vector():
    N = 10000

    # Check that perfect alignment matches the input vectors
    # This test includes axes perfectly aligned with the x, y, and z axes
    axes = normalized_vectors( np.array( [ [1,0,0], [0,1,0], [0,0,1], [1,1,1], [1,0,1], [1,1,0], [0,1,1], [2,3,1], [0,0,1], [4,1,1] ] ) )
    p = np.ones(len(axes))
    results = axes_correlated_with_input_vector( axes, p )
    assert( ( abs( abs(elementwise_dot(axes, results)) - 1 ) < 1e-10 ).all() )

    # Check that perfectly anti-aligned gives 0 dot products
    p = -np.ones(len(axes))
    results = axes_correlated_with_input_vector( axes, p )
    assert( ( abs(elementwise_dot(axes, results)) < 1e-10 ).all() )

    # Check that dropping the alignment strength will result in a lower
    axes = np.random.uniform(0, 1, (N,3))
    p = 0.7 * np.ones( N )
    results = axes_correlated_with_input_vector( axes, p )
    val1 = np.mean( abs( elementwise_dot( axes, results ) ) )

    p = 0 * np.ones( N )
    results = axes_correlated_with_input_vector( axes, p )
    val2 = np.mean( abs( elementwise_dot( axes, results ) ) )

    assert( val1 > val2 )

def test_is_z():
    """
    Test that this helper function properly marks the z-axes
    """
    axes = np.array( [ [1,1,1], [0,0,1], [2,3,1], [0,0,4], [1,1,0], [0,0,1], [0,0,2], [3,2,2] ] )
    truth = np.array([1,3,5,6])
    results = is_z(axes)
    assert((results==truth).all())
