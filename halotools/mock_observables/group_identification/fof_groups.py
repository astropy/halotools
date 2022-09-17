r"""
Module containing the `~halotools.mock_observables.FoFGroups` class used to
identify friends-of-friends groups of points.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse import csgraph, csr_matrix

from ..pair_counters.pairwise_distance_xy_z import pairwise_distance_xy_z

from ...custom_exceptions import HalotoolsError

igraph_available = True
try:
    import igraph
except ImportError:
    igraph_available = False
if (
    igraph_available is True
):  # there is another package called igraph--need to distinguish.
    if not hasattr(igraph, "Graph"):
        igraph_available is False
no_igraph_msg = (
    "igraph package not installed.  Some functions will not be available. \n"
    "See http://igraph.org/ and note that there are two packages called 'igraph'."
)

__all__ = ["FoFGroups"]
__author__ = ["Duncan Campbell"]


class FoFGroups(object):
    r"""
    Friends-of-friends (FoF) groups class.
    """

    def __init__(
        self, positions, b_perp, b_para, period=None, Lbox=None, num_threads=1
    ):
        r"""
        Build FoF groups in redshift space assuming the distant observer approximation.

        The first two dimensions (x, y) define the plane for perpendicular distances.
        The third dimension (z) is used for line-of-sight distances.

        See the :ref:`mock_obs_pos_formatting` documentation page for
        instructions on how to transform your coordinate position arrays into the
        format accepted by the ``positions`` argument.

        See also :ref:`galaxy_catalog_analysis_tutorial5`.

        Parameters
        ----------
        positions : array_like
            Npts x 3 numpy array containing 3-D positions of galaxies.
            Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        b_perp : float
            Maximum linking length in the perpendicular direction,
            normalized to the mean separation between galaxies.

        b_para : float
            Maximum linking length in the parallel direction,
            normalized to the mean separation between galaxies.

        period : array_like, optional
            Length-3 sequence defining the periodic boundary conditions
            in each dimension. If you instead provide a single scalar, Lbox,
            period is assumed to be the same in all Cartesian directions.
            Length units are comoving and assumed to be in Mpc/h, here and throughout Halotools.

        Lbox : array_like, optional
            length 3 array defining boundaries of the simulation box.

        num_threads : int, optional
            Number of threads to use in calculation, where parallelization is performed
            using the python ``multiprocessing`` module. Default is 1 for a purely serial
            calculation, in which case a multiprocessing Pool object will
            never be instantiated. A string 'max' may be used to indicate that
            the pair counters should use all available cores on the machine.

        Examples
        --------
        In this example we will populate the `~halotools.sim_manager.FakeSim`
        with an HOD-style model to demonstrate how to use the group-finder.

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()

        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model = PrebuiltHodModelFactory('zheng07', threshold = -22)
        >>> model.populate_mock(halocat)

        Now that we have a mock galaxy catalog, we will extract the 3d coordinates
        of the galaxy positions and place this information into the shape of the
        multi-d array expected by the ``positions`` argument of `FoFGroups`
        using the `~halotools.mock_observables.return_xyz_formatted_array` function.

        >>> from halotools.mock_observables import return_xyz_formatted_array

        Note that `FoFGroups` is based on 2+1 dimensional positions, with the z-dimension
        having a separate linking length from the xy-plane. To make our example
        more realistic, we will apply the redshift-space distortions to the z-coordinate
        when constructing the ``positions`` array.

        >>> x = model.mock.galaxy_table['x']
        >>> y = model.mock.galaxy_table['y']
        >>> z = model.mock.galaxy_table['z']
        >>> vz = model.mock.galaxy_table['vz']

        >>> positions = return_xyz_formatted_array(x, y, z, velocity = vz, velocity_distortion_dimension = 'z', period=halocat.Lbox)

        The ``b_perp`` and ``b_para`` arguments of `FoFGroups` control the linking lengths
        for the group-finding. The values passed for these variables are assumed to be in
        units of the mean number density of the input points, so that if you want your FoF
        linking length to be, say, 0.15 times the mean number density, then you should set
        ``b_perp`` to 0.15. Here we adopt the convention given in Berlind et al. (2006)
        and set ``b_perp`` to 0.14 and ``b_para`` to 0.75.

        >>> b_perp, b_para = (0.14,0.75)
        >>> groups = FoFGroups(positions, b_perp, b_para, period=halocat.Lbox)

        Now that groups have been identified, we can create a new column of our
        ``galaxy_table`` storing the group ID that each galaxy belongs to.

        >>> model.mock.galaxy_table['fof_group_ID'] = groups.group_ids

        At this point, we are now in a position to calculate a large variety of
        *group aggregation* statistics with the ``fof_group_ID`` as our grouping key.
        The `~halotools.utils.group_member_generator` is designed for exactly such
        calculations. See :ref:`galaxy_catalog_analysis_tutorial5` for a
        tutorial showing how to use this generator to analyze galaxy groups.

        See also
        --------
        :ref:`galaxy_catalog_analysis_tutorial5`
        """

        self.b_perp = float(b_perp)  # perpendicular linking length
        self.b_para = float(b_para)  # parallel linking length
        self.positions = np.asarray(
            positions, dtype=np.float64
        )  # coordinates of galaxies

        # process Lbox parameter
        if (Lbox is None) & (period is None):
            raise ValueError("Lbox and Period cannot be both be None.")
        elif (Lbox is None) & (period is not None):
            period = np.atleast_1d(period)
            if len(period) == 1:
                period = np.array((period[0], period[0], period[0]))
            Lbox = period
        elif (Lbox is not None) & (period is None):
            Lbox = np.atleast_1d(Lbox)
            if len(Lbox) == 1:
                Lbox = np.array([Lbox[0], Lbox[0], Lbox[0]])
            period = Lbox

        if np.shape(Lbox) != (3,):
            raise ValueError(
                "Lbox must be an array of length 3, or number indicating the "
                "length of one side of a cube"
            )
        if (period is not None) and (not np.all(Lbox == period)):
            raise ValueError("If both Lbox and Period are defined, they must be equal.")

        self.period = period  # simulation box periodic boundary conditions
        self.Lbox = np.asarray(
            Lbox, dtype="float64"
        )  # simulation box periodic boundary conditions
        # calculate the physical linking lengths
        self.volume = np.prod(self.Lbox)
        self.n_gal = len(positions) / self.volume
        self.d_perp = self.b_perp / (self.n_gal ** (1.0 / 3.0))
        self.d_para = self.b_para / (self.n_gal ** (1.0 / 3.0))
        self.m_perp, self.m_para = pairwise_distance_xy_z(
            self.positions,
            self.positions,
            self.d_perp,
            self.d_para,
            period=self.period,
            num_threads=num_threads,
        )

        self.m = self.m_perp.multiply(self.m_perp) + self.m_para.multiply(self.m_para)
        self.m = self.m.sqrt()

    @property
    def group_ids(self):
        r"""
        Determine integer IDs for groups.

        Each member of a group is assigned a unique integer ID that it shares with all
        connected group members.

        Returns
        -------
        group_ids : np.array
            array of group IDs for each galaxy

        """
        if getattr(self, "_group_ids", None) is None:
            self._n_groups, self._group_ids = csgraph.connected_components(
                self.m_perp, directed=False, return_labels=True
            )
        return self._group_ids

    @property
    def n_groups(self):
        r"""
        Calculate the total number of groups, including 1-member groups

        Returns
        -------
        N_groups: int
            number of distinct groups

        """
        if getattr(self, "_n_groups", None) is None:
            self._n_groups = csgraph.connected_components(
                self.m_perp, directed=False, return_labels=False
            )
        return self._n_groups

    def create_graph(self):
        """
        Create graph from FoF sparse matrix (requires igraph package).
        """
        if igraph_available is True:
            self.g = _scipy_to_igraph(self.m, self.positions, directed=False)
        else:
            raise HalotoolsError(no_igraph_msg)

    def get_degree(self):
        """
        Calculate the 'degree' of each galaxy vertex (requires igraph package).

        Returns
        -------
        degree : np.array
            the 'degree' of galaxies in groups

        """
        if igraph_available is True:
            try:
                self.degree = self.g.degree()
            except AttributeError:
                self.create_graph()
                self.degree = self.g.degree()
            return self.degree
        else:
            raise HalotoolsError(no_igraph_msg)

    def get_betweenness(self):
        """
        Calculate the 'betweenness' of each galaxy vertex (requires igraph package).

        Returns
        -------
        betweeness : np.array
            the 'betweenness' of galaxies in groups
        """
        if igraph_available is True:
            try:
                self.betweenness = self.g.betweenness()
            except AttributeError:
                self.create_graph()
                self.betweenness = self.g.betweenness()
            return self.betweenness
        else:
            raise HalotoolsError(no_igraph_msg)

    def get_multiplicity(self):
        """
        Return the multiplicity of galaxies' group (requires igraph package).
        """
        if igraph_available is True:
            try:
                clusters = self.g.clusters()
            except AttributeError:
                self.create_graph()
                clusters = self.g.clusters()
            mltp = np.array(clusters.sizes())
            self.multiplicity = mltp[self.group_ids]
            return self.multiplicity
        else:
            raise HalotoolsError(no_igraph_msg)

    def get_edges(self):
        r"""
        Return all edges of the graph (requires igraph package).

        Returns
        -------
        edges: np.ndarray
            N_edges x 2 array of vertices that are connected by an edge.  The vertices are
            indicated by their index.

        """
        if igraph_available is True:
            try:
                self.edges = np.asarray(self.g.get_edgelist())
            except AttributeError:
                self.create_graph()
                self.edges = np.asarray(self.g.get_edgelist())
            return self.edges
        else:
            raise HalotoolsError(no_igraph_msg)

    def get_edge_lengths(self):
        r"""
        Return the length of all edges (requires igraph package).

        Returns
        -------
        lengths: np.array
            The length of an 'edge' econnnecting galaxies, i.e. distance between galaxies.

        Notes
        ------
        The length is caclulated as:

        .. math::
            L_{\rm edge} = \sqrt{r_{\perp}^2 + r_{\parallel}^2},

        where :math:`r_{\perp}` and :math:`r_{\parallel}` are the perendicular and
        parallel distance between galaixes.

        """
        if igraph_available is True:
            try:
                edges = self.g.es()
            except AttributeError:
                self.create_graph()
                edges = self.g.es()
            lens = edges.get_attribute_values("weight")
            self.edge_lengths = np.array(lens)
            return self.edge_lengths
        else:
            raise HalotoolsError(no_igraph_msg)


def _scipy_to_igraph(matrix, coords, directed=False):
    r"""
    Convert a scipy sparse matrix to an igraph graph object (requires igraph package).

    Paramaters
    ----------
    matrix : object
        scipy.sparse pairwise distance matrix

    coords : np.array
        N by 3 array of coordinates of points

    Returns
    -------
    graph : object
        igraph graph object

    """

    matrix = csr_matrix(matrix)
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets].tolist()[0]

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if igraph_available:
        g = igraph.Graph(
            list(zip(sources, targets)),
            n=matrix.shape[0],
            directed=directed,
            edge_attrs={"weight": weights},
            vertex_attrs={"x": x, "y": y, "z": z},
        )
        return g
    else:
        raise HalotoolsError(no_igraph_msg)
