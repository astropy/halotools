r"""
halotools model components for modelling central and scatellite intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from .ia_strength_models import alignment_strength

# vector rotations
from ...utils import rotate_vector_collection
from ...utils.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from ...utils.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                              angles_between_list_of_vectors, vectors_normal_to_planes)
from ...utils.rotations3d import vectors_between_list_of_vectors, rotation_matrices_from_angles

# watson distribution
from .watson_distribution import DimrothWatson

# utilities
from warnings import warn
from astropy.utils.misc import NumpyRNGContext

from ...utils import crossmatch


__all__ = ('RandomAlignment',
           'CentralAlignment',
           'SatelliteAlignment',
           'RadialSatelliteAlignment',
           'SubhaloAlignment')
__author__ = ('Duncan Campbell',)


class RandomAlignment(object):
    """
    class to model random galaxy orientations
    """
    def __init__(self, gal_type='centrals', **kwargs):
        """
        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = []
        self._methods_to_inherit = ([])
        self.param_dict = ({})

    def assign_orientation(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
        else:
            N = kwargs['size']

        # assign random orientations
        major_v = random_unit_vectors_3d(N)
        inter_v = random_perpendicular_directions(major_v)
        minor_v = normalized_vectors(np.cross(major_v, inter_v))

        if 'table' in kwargs.keys():

            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*N)
                msg = ("Because `gal_type` is not indicated in `table`, the orientations",
                       "are being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class CentralAlignment(object):
    r"""
    alignment model for central galaxies in host-halos
    """
    def __init__(self, central_alignment_strength=1.0, prim_gal_axis='major', **kwargs):
        r"""
        Parameters
        ----------
        central_alignment_strength : float
            [-1,1] bounded number indicating alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`.

        alignment_keys : list
            A list of strings indicating the keywords for the x,y- and z components of the
            halo alignment vector. Deafult is ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z'].

        Notes
        -----
        If the kwargs or table contains a key "alignment_strength", when populating a mock,
        this will be used instead of the `central_alignment_stregth` parameter passed during intialization.
        This is how varying the alignment strength as a function of galaxy/halo properties is handeled.
        """

        self.gal_type = 'centrals'
        self._mock_generation_calling_sequence = (['assign_central_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        # specify the halo alignment vector
        if 'alignment_keys' in kwargs.keys():
            assert len(kwargs['alignment_keys'])==3
            self.list_of_haloprops_needed = kwargs['alignment_keys']
        else:
            self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_central_orientation'])
        self.param_dict = ({
            'central_alignment_strength': central_alignment_strength})

    def assign_central_orientation(self, **kwargs):
        r"""
        Assign a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ==========
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo alignment axis

        Returns
        =======
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axes
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table[self.list_of_haloprops_needed[0]]
            Ay = table[self.list_of_haloprops_needed[1]]
            Az = table[self.list_of_haloprops_needed[2]]
        else:
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['central_alignment_strength']
            except KeyError:
                msg = ('`central_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(Ax))*self.param_dict['central_alignment_strength']
        else:
            p = np.ones(len(Ax))*self.param_dict['central_alignment_strength']

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v
        else:
            msg = ('primary galaxy axis {0} is not recognized.'.format(self.prim_gal_axis))
            raise ValueError(msg)

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class SatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies in sub-halos
    """
    def __init__(self, satellite_alignment_strength=1.0, prim_gal_axis='major', **kwargs):
        r"""
        Parameters
        ----------
        satellite_alignment_strength : float
            [-1,1] bounded number indicating alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`.

        alignment_keys : list
            A list of strings indicating the keywords for the x,y- and z components of the
            halo alignment vector. Deafult is ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z'].

        Notes
        -----
        If the kwargs or table contains a key "alignment_strength", when populating a mock,
        this will be used instead of the `satellite_alignment_stregth` parameter passed during intialization.
        This is how varying the alignment strength as a function of galaxy/halo properties is handeled.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        # specify the halo alignment vector
        if 'alignment_keys' in kwargs.keys():
            assert len(kwargs['alignment_keys'])==3
            self.list_of_haloprops_needed = kwargs['alignment_keys']
        else:
            self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_satellite_orientation'])
        self.param_dict = ({
            'satellite_alignment_strength': satellite_alignment_strength})

    def assign_satellite_orientation(self, **kwargs):
        r"""
        Assign a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ==========
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo alignment axis

        Returns
        =======
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axes
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table[self.list_of_haloprops_needed[0]]
            Ay = table[self.list_of_haloprops_needed[1]]
            Az = table[self.list_of_haloprops_needed[2]]
        else:
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']
        else:
            p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v
        else:
            msg = ('primary galaxy axis {0} is not recognized.'.format(self.prim_gal_axis))
            raise ValueError(msg)

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class RadialSatelliteAlignment(object):
    r"""
    radial alignment model for satellite galaxies
    """

    def __init__(self, prim_gal_axis='major', **kwargs):
        """
        Parameters
        ----------
        satellite_alignment_strength : float
            parameter between [-1,1] that sets the alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, or `minor`.

        Notes
        -----
        If the kwargs or table contain a key "satellite_alignment_strength", this will be used instead.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties',
                                                   'assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.inf
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (
            ['assign_satellite_orientation', 'inherit_halocat_properties'])

        # set parameters
        self.set_default_params()
        if 'satellite_alignment_strength' in kwargs.keys():
            mu_sat = kwargs['satellite_alignment_strength']
            self.param_dict = ({'satellite_alignment_strength': mu_sat})

    def set_default_params(self):
        r"""
        set default parameters
        """
        d = {'satellite_alignment_strength': 0.8}
        self.param_dict = d

    def inherit_halocat_properties(self, seed=None, **kwargs):
        r"""
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_satellite_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Returns
        -------
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axies
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox

        # calculate the radial vector between satellites and centrals
        major_input_vectors, r = self.get_radial_vector(Lbox=Lbox, **kwargs)

        # check for length 0 radial vectors
        mask = (r<=0.0) | (~np.isfinite(r))
        N_bad_axes = np.sum(mask)
        if N_bad_axes>0:
            major_input_vectors[mask,:] = random_unit_vectors_3d(N_bad_axes)
            msg = ('{0} galaxies have a radial distance equal to zero (or infinity) from their host. '
                   'These galaxies will be re-assigned random alignment vectors.'.format(int(N_bad_axes)))
            warn(msg)

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` key not detected in `table`.'
                       'The value set in self.param_dict of this class will be used instead.')
                warn(msg)
                p = np.ones(len(table))*self.param_dict['satellite_alignment_strength']
        else:
            N = len(self.param_dict['x'])
            p = np.ones(N)*self.param_dict['satellite_alignment_strength']

        # set prim_gal_axis orientation
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # check for nan vectors
        mask = (~np.isfinite(np.prod(A_v, axis=-1)))
        N_bad_axes = np.sum(mask)
        if N_bad_axes>0:
            A_v[mask,:] = random_unit_vectors_3d(N_bad_axes)
            msg = ('{0} correlated alignment axis(axes) were found to be not finite. '
                   'These will be re-assigned random vectors.'.format(int(N_bad_axes)))
            warn(msg)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("`gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


    def get_radial_vector(self, Lbox=None, **kwargs):
        """
        caclulate the radial vector for satellite galaxies

        Parameters
        ==========
        x, y, z : array_like
            galaxy positions

        halo_x, halo_y, halo_z : array_like
            host halo positions

        halo_r : array_like
            halo size

        Lbox : array_like
            array len(3) giving the simulation box size along each dimension

        Returns
        =======
        r_vec : numpy.array
            array of radial vectors of shape (Ngal, 3) between host halos and satellites

        r : numpy.array
            radial distance
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            x = table['x']
            y = table['y']
            z = table['z']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
        else:
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']

        if Lbox is None:
            Lbox = self._Lbox

        # define halo-center - satellite vector
        # accounting for PBCs
        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        r_vec = np.vstack((dx, dy, dz)).T
        r = np.sqrt(np.sum(r_vec*r_vec, axis=-1))

        return r_vec, r


class SubhaloAlignment(object):
    r"""
    alignment model for satellite galaxies in sub-halos aligning with their respective subhalos
    most of the functionality here is copied from SatelltieAlignment by Duncan Campbell.

    Notes
    -----
    When using this module, you must use SubhaloPhaseSpace as the phase space model and you must use the alignment_inherited_subhalo_props_dict as the
    inherited_subhalo_props_dict. This will ensure the proper subhalo value columns are included in the galaxy table
    """
    def __init__(self, halocat=None, satellite_alignment_strength=1.0, prim_gal_axis='major', rotate_relative=True, **kwargs):
        r"""
        Parameters
        ----------
        satellite_alignment_strength : float
            [-1,1] bounded number indicating alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`.
            
        rotate_relative : bool, optional
            bool indicating whether any subhalos with real_subhalo = False should be rotated to 
            preserve the relative orientation they had to their true host halo with the current host halo.
            Default is True, which will rotate the false subhalos to keep the same relative orientation
            with respect to the new host halo. A value of False will keep their original values.

        alignment_keys : list
            A list of strings indicating the keywords for the x,y- and z components of the
            halo alignment vector. Deafult is ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z'].

        Notes
        -----
        If the kwargs or table contains a key "alignment_strength", when populating a mock,
        this will be used instead of the `satellite_alignment_stregth` parameter passed during intialization.
        This is how varying the alignment strength as a function of galaxy/halo properties is handeled.
        
        Unlike SatelliteAlignment, this alignment model needs a table to exist prior to being called. This is
        because the 
        """
        
        if halocat is None:
            msg = "halocat must be passed as an argument to preserve subhalo information"
            raise Exception(msg)
        self._halocat = halocat
        
        self._rotate_relative = rotate_relative

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        # specify the halo alignment vector
        if 'alignment_keys' in kwargs.keys():
            assert len(kwargs['alignment_keys'])==3
            self.list_of_haloprops_needed = kwargs['alignment_keys']
        else:
            self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_satellite_orientation'])
        self.param_dict = ({
            'satellite_alignment_strength': satellite_alignment_strength})
       
    def _orient_false_subhalo(self, table):
        r"""
        For subhalos taken from other host halos (marked by real_subhalo == False), this function rotates the false subhalos so 
        that their relative orientation with respect to their new host halo is the same as their original orientation was with
        respect to their original host halo. Rotates the position, orientation, and velocities.

        Parameters
        ==========
        table: astropy table holding the halo and galaxy information for the model

        Returns
        =======
        None, the table is modified in place
        """
        
        # Values of interest
        host_axis_A = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']          # Use this axis to rotate
        axis_A = ['subhalo_axisA_x', 'subhalo_axisA_y', 'subhalo_axisA_z']
        velocity = ['vx', 'vy', 'vz']
        position = ['x', 'y', 'z']
        
        mask = ( table['gal_type'] == 'satellites' ) & ( table['real_subhalo'] == False )
        
        halo_ids = table[mask]['halo_id']
        inds1, inds2 = crossmatch(halo_ids, self._halocat.halo_table["halo_id"])
        
        original_host_ids = self._halocat.halo_table[inds2]["halo_hostid"]
        new_host_ids = table[mask]['halo_hostid']

        # crossmatch does not preserve the order
        # this will allow the crossmatched original_host_ids to match the original order (the same order as in mask)
        og_order = np.arange(len(original_host_ids))
        reorder = np.argsort(og_order[inds1])
        original_host_ids = original_host_ids[reorder]
        
        # inds1[i] is the index in halo_ids of the halo id at position i in model_instance.mock.galaxy_table['halo_ids']
        # Similarly, inds2[i] is the index in model_instance.mock.galaxy_table['halo_ids'] of the halo id in position i in halo_ids
        
        halo_axesA = np.array( [ table[mask][axis_A[0]], table[mask][axis_A[1]], table[mask][axis_A[2]] ] ).T
        halo_velocities = np.array( [ table[mask][velocity[0]], table[mask][velocity[1]], table[mask][velocity[2]] ] ).T
        halo_positions = np.array( [ table[mask][position[0]], table[mask][position[1]], table[mask][position[2]] ] ).T
        
        # Get original host axes
        inds1, inds2 = crossmatch( original_host_ids, self._halocat.halo_table["halo_id"] )
        original_host_axesA = np.array( [ self._halocat.halo_table[inds2][host_axis_A[0]], self._halocat.halo_table[inds2][host_axis_A[1]], self._halocat.halo_table[inds2][host_axis_A[2]] ] ).T
        
        # re-order to match original_host_ids
        og_order = np.arange(len(original_host_ids))
        reorder = np.argsort( og_order[inds1] )
        original_host_axesA = original_host_axesA[reorder]
        
        # Get new host axes
        inds1, inds2 = crossmatch(new_host_ids, self._halocat.halo_table["halo_id"])
        new_host_axesA = np.array( [ self._halocat.halo_table[inds2][host_axis_A[0]], self._halocat.halo_table[inds2][host_axis_A[1]], self._halocat.halo_table[inds2][host_axis_A[2]] ] ).T
        
        # re-order to match new_host_ids
        og_order = np.arange(len(new_host_ids))
        reorder = np.argsort( og_order[inds1] )
        new_host_axesA = new_host_axesA[reorder]
    
        # Get the rotation axis to rotate original_host_axisA into new_host_axisA
        # Then use that rotation matrix to rotate halo_axisA and halo_velocities
        rot = np.array( [ self._get_rotation_matrix(original_host_axesA[i], new_host_axesA[i]) for i in range(len(original_host_axesA)) ] )
        rotated_axes = rotate_vector_collection(rot, halo_axesA )
        rotated_velocities = rotate_vector_collection(rot, halo_velocities )
        rotated_positions = rotate_vector_collection(rot, halo_positions)
        
        for i in range(len(axis_A)):
            table[ axis_A[i] ][ mask ] = rotated_axes[:,i]
            table[ velocity[i] ][ mask ] = rotated_velocities[:,i]
            table[ position[i] ][ mask ] = rotated_positions[:,i]
    
    def _get_rotation_matrix(self, a, b):
        r"""
        Returns the rotation matrix (only 3D) needed to rotate a into b.
        Used for _orient_false_subhalo
        
        Parameters
        ==========
        a : 3 element numpy array representing a vector in 3D
        b : 3 element numpy array representing a vector in 3D
        
        Returns
        =======
        mat : 3x3 numpy array representing the rotation matrix needed to rotate a in the direction of b
        """
        unit_a = normalized_vectors(a)
        unit_b = normalized_vectors(b)
        v = np.cross(unit_a,unit_b)[0]
        s = np.linalg.norm(v)                           # Sin of the angle
        c = np.dot(unit_a,unit_b.T)                  # Cos of the angle
        
        vx = np.zeros((3,3))
        vx[0,1] = -v[2]
        vx[1,0] = v[2]
        vx[0,2] = v[1]
        vx[2,0] = -v[1]
        vx[1,2] = -v[0]
        vx[2,1] = v[0]
        
        mat = np.eye(3) + vx + np.dot(vx,vx)*((1-c)/(s*s))
        return mat

    def assign_satellite_orientation(self, **kwargs):
        r"""
        Assign a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ==========
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo alignment axis

        Returns
        =======
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axes
        """
        
        # Assume the proper columns exist (i.e. the user has used alignment_inherited_subhalo_props_dict in SubhaloPhaseSpace)
        
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table["sub"+self.list_of_haloprops_needed[0]]
            Ay = table["sub"+self.list_of_haloprops_needed[1]]
            Az = table["sub"+self.list_of_haloprops_needed[2]]
            if self._rotate_relative:
                self._orient_false_subhalo(table=table)
        else:
            Ax = kwargs['subhalo_axisA_x']
            Ay = kwargs['subhalo_axisA_y']
            Az = kwargs['subhalo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']
        else:
            p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v
        else:
            msg = ('primary galaxy axis {0} is not recognized.'.format(self.prim_gal_axis))
            raise ValueError(msg)

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


def axes_correlated_with_z(p, seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the z-axis (0, 0, 1).
    Parameters
    ----------
    p : ndarray
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)

    Notes
    -----
    The `axes_correlated_with_z` function works by modifying the standard method
    for generating random points on the unit sphere. In the standard calculation,
    the z-coordinate :math:`z = \cos(\theta)`, where :math:`\cos(\theta)` is just a
    uniform random variable. In this calculation, :math:`\cos(\theta)` is not
    uniform random, but is instead implemented as a clipped power law
    implemented with `scipy.stats.powerlaw`.
    """

    p = np.atleast_1d(p)
    npts = p.shape[0]

    with NumpyRNGContext(seed):
        phi = np.random.uniform(0, 2*np.pi, npts)
        # sample cosine theta nonuniformily to correlate with in z-axis
        if np.all(p == 0):
            uran = np.random.uniform(0, 1, npts)
            cos_t = uran*2.0 - 1.0
        else:
            k = alignment_strength(p)
            d = DimrothWatson()
            cos_t = d.rvs(k)

    sin_t = np.sqrt((1.-cos_t*cos_t))

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    return np.vstack((x, y, z)).T


def axes_correlated_with_input_vector(input_vectors, p=0., seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the orientation of `input_vectors`.

    Parameters
    ----------
    input_vectors : ndarray
        Numpy array of shape (npts, 3) storing a list of 3d vectors defining the
        preferred orientation with which the returned vectors will be correlated.
        Note that the normalization of `input_vectors` will be ignored.

    p : ndarray, optional
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Default is zero, for no correlation.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)
    """

    input_unit_vectors = normalized_vectors(input_vectors)
    assert input_unit_vectors.shape[1] == 3
    npts = input_unit_vectors.shape[0]

    z_mask = is_z(input_unit_vectors)
    
    # For some reason, this function is very sensitive, and the difference between float32 and float64 drastically changes results
    # With only a single alignment strength, the np.ones(length)*alignmnet_strength gives float64 numbers
    # but pulling the satellite_slignment_strength column from the table gives float32
    # At values of exactl 1 and -1, the float64 numbers do fine, but float32 don'table
    # Not sure why. But they do. So I put in this next line
    p = np.array(p).astype("float64")

    z_correlated_axes = axes_correlated_with_z(p, seed)

    z_axes = np.tile((0, 0, 1), npts).reshape((npts, 3))

    angles = angles_between_list_of_vectors(z_axes, input_unit_vectors)
    rotation_axes = vectors_normal_to_planes(z_axes, input_unit_vectors)
    matrices = rotation_matrices_from_angles(angles, rotation_axes)

    unit_vectors = rotate_vector_collection(matrices, z_correlated_axes)

    # Any vectors directly along z will be nan as you cannot rotate z into z
    unit_vectors[z_mask] = z_correlated_axes[z_mask]
    return unit_vectors

def is_z(axes, normalize=False):
    r"""
    Returns array of indices where the rows of axes are equal to the z axis.

    Parameters
    -----------
    axes : ndarray
        ndarray of shape (npts,3) where each row represents a 3d orientation axis.

    normalize : boolean, optionsl
        Boolean specifying whether or not to normalize the axes. Default is False.

    Returns
    -------
    z_mask : array of indices marking the rows in axes that point exactly along the z-axis
    """
    if normalize:
        axes = normalized_vectors(axes)
    
    return np.where( np.logical_and( axes[:,0] == 0, axes[:,1] == 0, axes[:,2] != 0 ) )
