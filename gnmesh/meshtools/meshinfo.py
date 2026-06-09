"""
Mesh bookkeeping utilities.

This module provides helpers for building and storing neighbour
information on unstructured meshes. It includes lightweight containers
such as `CellNeighbourInfo` and `MeshInfo`, plus utility functions for
triangle area computation and neighbourhood selection used by gradient
and Hessian routines.

Functions
---------
cell_area_triangle
    Compute the geometric area of a triangle cell.
distance_to_neighbour_list_for_cell
    Return indices of cells within a distance threshold from a cell.
distance_to_neighbour_list_for_mesh
    Return neighbour lists for all cells in a mesh.
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import logging
from pygimli.viewer.mpl import drawModel, drawMeshBoundaries

logger = logging.getLogger(__name__)

# Calculate area from a pyGIMLi triangle cell.
def cell_area_triangle(cell):
    """
    Calculate the area of a triangle cell.

    Parameters:
    cell (object): The cell object representing a triangle.

    Returns:
    float: The area of the triangle.

    Raises:
    ValueError: If the cell does not have exactly 3 nodes.

    """
    # Get the nodes of the cell
    nodes = cell.nodes()
    # Get the number of nodes
    n_nodes = len(nodes)
    # Initialize the area
    area = 0
    # Abort if the cell has less/more than 3 nodes
    if n_nodes < 3 or n_nodes > 3:
        raise ValueError("Only triangles are supported")
    # Get the coordinates of the nodes
    x1, y1 = nodes[0].x(), nodes[0].y()
    x2, y2 = nodes[1].x(), nodes[1].y()
    x3, y3 = nodes[2].x(), nodes[2].y()
    # Cross-product area of the triangle.
    return 0.5 * (x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3)

# Functions to calculate cell neighbourhoods.
def distance_to_neighbour_list_for_cell(cell, mesh, dist = 1.0, dimension=2):
    """
    Function to calculate the neighbours of a cell.

    Parameters:
    cell (object): The cell object.
    mesh (object): The mesh object.
    dist (float): The distance to consider as a neighbour.
    dimension (int): The dimension of the mesh.

    Returns:
    
    function: The function to calculate the neighbours of a cell.

    """
    # Get the centers of the cells
    centers = np.zeros((len(mesh.cells()),dimension))
    for num, cell_temp in enumerate(mesh.cells()):
        centers[num] = np.array(cell_temp.center())[0:dimension]

    # Calculate the distances
    centers = centers - np.array(cell.center())[0:dimension]

    # Collect all cell indices within the distance threshold.
    neigh_list = list(np.where(np.linalg.norm(centers, axis=1)<=dist)[0])
    return neigh_list

def distance_to_neighbour_list_for_mesh(mesh, dist = 1.0, dimension=2):
    """
    Function to calculate the neighbours of a cell for the whole mesh.

    Parameters:
    mesh (object): The mesh object.
    dist (float): The distance to consider as a neighbour.
    dimension (int): The dimension of the mesh.

    Returns:
    
    function: The function to calculate the neighbours of a cell.

    """
    # Get the centers of the cells
    centers = np.zeros((len(mesh.cells()),dimension))
    for num, cell in enumerate(mesh.cells()):
        centers[num] = np.array(cell.center())[0:dimension]

    # Calculate the distance matrix
    dist_mat = sp.spatial.distance_matrix(centers, centers)
    dist_mat_bool = dist_mat<=dist

    # Collect all cell indices within the distance threshold.
    neigh_list_list = []
    for j in range(dist_mat_bool.shape[0]):
        temp_list = list(np.where(dist_mat_bool[j])[0])
        temp_list.remove(j)
        neigh_list_list.append(temp_list)
    return neigh_list_list

def get_n_closest_neighbours(cell, mesh, n=3):
    """
    Function to get the n closest neighbours of a cell.

    Parameters:
    cell (object): The cell object.
    mesh (object): The mesh object.
    n (int): The number of neighbours to get.

    Returns:

    np.array: The list of the cell numbers of the n closest neighbours.
    """
    assert n > 0, "n must be greater than 0"
    cell_centers = np.array([np.array(ce.center()) for ce in mesh.cells()])
    distances = sp.spatial.distance_matrix(cell_centers, [cell_centers[cell.id()]])
    closest_neighbours = np.argsort(distances, axis=0)[1:n+1]
    return closest_neighbours[:,0]

def get_n_closest_neighbours_function_for_mesh(mesh, n=3):
    """
    Function to get the n closest neighbours of a cell for the whole mesh.

    Parameters:
    mesh (object): The mesh object.
    n (int): The number of neighbours to get.

    Returns:

    function: The function to get the n closest neighbours of a cell.
    """
    cell_centers = np.array([np.array(ce.center()) for ce in mesh.cells()])
    distance_matrix = sp.spatial.distance_matrix(cell_centers, cell_centers)
    def get_closest_neighbour_preset_function(cell):
        distances = distance_matrix[:, cell.id()]
        closest_neighbours = np.argsort(distances)[1:n+1]
        return closest_neighbours
    return get_closest_neighbour_preset_function

# Store geometry and Taylor-system matrices for one cell.
class CellNeighbourInfo:
    """
    Class to store the information about the neighborhood of a cell.

    Attributes:
    cell_number (int): The number of the cell.
    cell_area (float): The area of the cell.
    cell_center (np.array): The center of the cell.
    dimension (int): The dimension of the mesh.
    neighbour_cells (list): The list of the cell numbers of the neighbours.
    distance_matrix (np.array): The distance matrix of the cell.
    distance_matrix_gn_taylor1 (np.array): The distance matrix of the cell used for the 
                                           Gauss-Newton algorithm.
    distance_matrix_gn_taylor2 (np.array): The distance matrix of the cell used for the 
                                           Gauss-Newton algorithm from a second degree 
                                           taylor polynomial approximation.

    """

    def __init__(
            self,
            cell,
            dimension=2,
            cell_area_function=cell_area_triangle,
            neighbour_function=None,
            verbose=False):
        """
        Initialize the NeighbourAttribute object.

        Parameters:
        cell (object): The cell object.
        dimension (int): The dimension of the mesh.
        cell_area_function (function): The function to calculate the area of the cell.
        neighbour_function (function): The function to calculate the neighbours of the cell.
        verbose (bool): If True, output the progress.
        """
        # Get the cell number
        self._cell_number = cell.id()

        # Get the area of the cell
        self._cell_area = cell_area_function(cell)

        # Get the dimension of the cell
        assert isinstance(dimension, int) and dimension > 0, "Dimension must be a positive integer"
        self._dimension = dimension

        # Get the center of the cell
        self._cell_center = np.array(cell.center())[0:self.dimension]

        # Get the neighbours of the cell.
        if neighbour_function is not None:
            self._neighbour_cells = neighbour_function(cell)
        else:
            self._neighbour_cells = []
            for j in range(cell.neighborCellCount()):
                try:
                    self._neighbour_cells.append(cell.neighborCell(j).id())
                except AttributeError:
                    if verbose:
                        logger.info("Cell %s has no neighbour at position %s", cell.id(), j)

        # Lazily populated matrices used by the Taylor-based solvers.
        self._distance_matrix = None
        self._distance_matrix_gn_taylor1 = None
        self._distance_matrix_gn_taylor2 = None

    @property
    def cell_number(self):
        """ The number of the cell. """
        return self._cell_number

    @property
    def cell_area(self):
        """ The area of the cell. """
        return self._cell_area

    @property
    def cell_center(self):
        """ The center of the cell. """
        return self._cell_center

    @property
    def dimension(self):
        """ The dimension of the mesh. """
        return self._dimension

    @property
    def neighbour_cells(self):
        """ The list of the cell numbers of the neighbours. """
        return self._neighbour_cells

    @property
    def distance_matrix(self):
        """ The distance matrix of the cell. """
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, mesh):
        distance_matrix = np.zeros((len(self.neighbour_cells), self.dimension))

        for i, cell_id in enumerate(self.neighbour_cells):
            cell = mesh.cell(cell_id)
            distance_matrix[i] = np.array(cell.center())[0:self.dimension] - self.cell_center
        self._distance_matrix = distance_matrix.copy()

    @property
    def distance_matrix_gn_taylor1(self):
        """ The distance matrix of the cell used for the Gauss-Newton algorithm. 
        With the distance matrix as D, this is given as (D.T@D)^-1@D. """
        return self._distance_matrix_gn_taylor1

    @distance_matrix_gn_taylor1.setter
    def distance_matrix_gn_taylor1(self, mesh):
        if not isinstance(self._distance_matrix, np.ndarray):
            self.distance_matrix = mesh
        else:
            try:
                # Solve the normal equations once the local distance matrix is available.
                self._distance_matrix_gn_taylor1 = np.linalg.inv(
                    self._distance_matrix.T @ self._distance_matrix
                ) @ self._distance_matrix.T
            except np.linalg.LinAlgError:
                self._distance_matrix_gn_taylor1 = None
                logger.warning("Singular matrix for cell %s", self.cell_number)
                logger.info("GN Taylor 1 matrix not calculated - try increasing the number of neighbours")

    @property
    def distance_matrix_gn_taylor2(self):
        """ The distance matrix of the cell used for the Gauss-Newton algorithm from a
        second degree taylor polynomial approximation. The second degree taylor polynomial
        approximation is given as f(x) - f(x0) = (x-x0).T nabla{f(x0)} + 0.5*(x-x0).T H(x0)(x-x0).
        Writing the gradient as [a b] and the Hessian as [[c d] [e f]], the equation reads
        f(x) - f(x0) = (x-x0).T [a b] + 0.5*(x-x0).T [[c d] [e f]] (x-x0). Combining the vector of unknowns
        as [a b c d e f] and the vector of model differences as [f(x) - f(x0)] and the difference matrix as D,
        the equation reads [f(x) - f(x0)] = [D (x-x0)_0 * D (x-x0)_1 * D ...] [a b c d e f ...], with the 
        Hessian matrix "stacked" columnwise. The components of the gradient and Hessian are then given as
        the solutiong of the linear system.
        """
        return self._distance_matrix_gn_taylor2

    @distance_matrix_gn_taylor2.setter
    def distance_matrix_gn_taylor2(self, mesh):
        if not isinstance(self._distance_matrix, np.ndarray):
            self.distance_matrix = mesh
        else:
            try:
                # First block corresponds to the gradient terms, followed by Hessian terms.
                dist_mat_sq = np.tile(self._distance_matrix, reps=(1,1+self.dimension)).copy()
                # Multiply the Hessian columns by the corresponding coordinate offsets.
                for dim in range(self.dimension):
                    dist_mat_sq[:,(dim+1)*self.dimension:(dim+2)*self.dimension] *= np.tile(dist_mat_sq[:,dim], reps=(self.dimension,1)).T
                # Remove duplicated symmetric Hessian entries.
                triu_indices = []
                tril_indices = []
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        running_index = self.dimension + i * self.dimension + j
                        if i<j:
                            triu_indices.append(running_index)
                        if j<i:
                            tril_indices.append(running_index)
                dist_mat_sq[:, triu_indices] = dist_mat_sq[:, tril_indices]
                dist_mat_sq = np.delete(dist_mat_sq, tril_indices, axis=1)
                self._distance_matrix_gn_taylor2 = np.dot(
                    np.linalg.inv(np.dot(dist_mat_sq.T, dist_mat_sq)), dist_mat_sq.T
                    )
            except np.linalg.LinAlgError:
                self._distance_matrix_gn_taylor2 = None
                logger.warning("Singular matrix for cell %s", self.cell_number)
                logger.info("GN Taylor 2 matrix not calculated - try increasing the number of neighbours")

    def get_gradient_mesh_sensitivities(self, order=1):
        """
        Function to return the sensitivities of the gradient with respect to the mesh. That means,
        that the gradient is given as gradient = output @ model. This functions returns an column index vector
        as well as the entries of the columns.
        
        Parameters:
        order (int): The order of the taylor polynomial approximation.

        Returns:
        np.array: The column index vector.
        np.array: The entries of the columns.
        """
        if order == 1:
            assert self._distance_matrix_gn_taylor1 is not None, "GN Taylor 1 matrix not calculated"
            original_matrix = self._distance_matrix_gn_taylor1
        elif order == 2:
            assert self._distance_matrix_gn_taylor2 is not None, "GN Taylor 2 matrix not calculated"
            original_matrix = self._distance_matrix_gn_taylor2[:self.dimension, :]
        else:
            raise ValueError("Only orders 1 and 2 are supported")
        # Append the center cell so the returned matrix matches the full stencil.
        column_index_vector = np.array([*self.neighbour_cells, self.cell_number])

        # The center-cell coefficient is the negative sum of the neighbour coefficients.
        center_cell_it_sensitivity = -np.sum(original_matrix, axis=1)
        original_matrix = np.hstack((original_matrix, center_cell_it_sensitivity[:,np.newaxis]))
        return column_index_vector, original_matrix
    
    def get_hessian_mesh_sensitivities(self):
        """
        Function to return the sensitivities of the Hessian with respect to the mesh. That means,
        that the Hessian is given as Hessian = output @ model. This functions returns an column index vector.

        Parameters:
            None

        Returns:
            np.array: The column index vector.
            np.array: The entries of the columns.
        """
        assert self._distance_matrix_gn_taylor2 is not None, "GN Taylor 2 matrix not calculated"
        original_matrix = self._distance_matrix_gn_taylor2[self.dimension:, :]
        # Append the center cell so the returned matrix matches the full stencil.
        column_index_vector = np.array([*self.neighbour_cells, self.cell_number])

        # The center-cell coefficient is the negative sum of the neighbour coefficients.
        center_cell_it_sensitivity = -np.sum(original_matrix, axis=1)
        original_matrix = np.hstack((original_matrix, center_cell_it_sensitivity[:,np.newaxis]))
        return column_index_vector, original_matrix

class MeshInfo:
    """
    Class to store the information about the mesh.

    Attributes:
    mesh (object): The mesh object.
    dimension (int): The dimension of the mesh.
    cell_area_function (function): The function to calculate the area of the cell.
    neighbour_function (function): The function to calculate the neighbours of the cell.
    cell_neighbour_info (list): The list of the CellNeighbourInfo objects.

    """

    def __init__(
            self,
            mesh,
            dimension=2,
            cell_area_function=cell_area_triangle,
            neighbour_function=None,
            initialise_gn1=True,
            initialise_gn2=False):
        """
        Initialize the MeshInfo object.

        Parameters:
        mesh (object): The mesh object.
        dimension (int): The dimension of the mesh.
        cell_area_function (function): The function to calculate the area of the cell.
        neighbour_function (function): The function to calculate the neighbours of the cell.
        initialise_matrices (bool): If True, initialise the matrices for gradient calculation.

        """
        # Set the mesh
        self._mesh = mesh

        # Set the dimension of the mesh
        self._dimension = dimension

        # Set the function to calculate the area of the cell
        self._cell_area_function = cell_area_function

        # Set the function to calculate the neighbours of the cell
        self._neighbour_function = neighbour_function

        # Region of interest defaults to the non-minimum cell markers.
        cell_markers = np.array(mesh.cellMarkers())
        smallest_cell_marker = np.min(cell_markers)
        if np.all(cell_markers == smallest_cell_marker):
            logger.info("All cells have the same marker - region of interest is the whole mesh")
            default_region_of_interest = np.array([True] * len(mesh.cells()))
        else:
            logger.info("Cells have different markers - region of interest are cells with non minimum marker")
            default_region_of_interest = np.array(cell_markers != smallest_cell_marker)
        self.region_of_interest = default_region_of_interest

        # Build the per-cell neighbourhood objects.
        cell_neighbour_info_list = []
        no_of_cells = len(self._mesh.cells())
        five_percent_cells = max(1, no_of_cells // 20)
        counter = 0

        for num, cell in enumerate(self._mesh.cells()):
            if num % five_percent_cells == 0:
                logger.info(f"Progress at {5*counter}% - Calculating cell {num}/{no_of_cells}")
                counter += 1
            cell_neighbour_info = CellNeighbourInfo(
                cell,
                self._dimension,
                self._cell_area_function,
                self._neighbour_function)

            # Precompute the local matrices used by the Taylor-based solvers.
            cell_neighbour_info.distance_matrix = self._mesh

            if initialise_gn1:
                cell_neighbour_info.distance_matrix_gn_taylor1 = self._mesh
            if initialise_gn2:
                cell_neighbour_info.distance_matrix_gn_taylor2 = self._mesh

            # Store the completed cell info.
            cell_neighbour_info_list.append(cell_neighbour_info)
        self._cell_neighbour_info = cell_neighbour_info_list
        self._gn_taylor_1_set_successfully = np.all([cni.distance_matrix_gn_taylor1 is not None for cni in self._cell_neighbour_info])
        self._gn_taylor_2_set_successfully = np.all([cni.distance_matrix_gn_taylor2 is not None for cni in self._cell_neighbour_info])
        
    @property
    def mesh(self):
        """The mesh object."""
        return self._mesh

    @property
    def dimension(self):
        """The dimension of the mesh."""
        return self._dimension

    @property
    def cell_area_function(self):
        """The function used to calculate cell areas."""
        return self._cell_area_function
    
    @property
    def region_of_interest(self):
        """Boolean mask selecting the region of interest."""
        return self._region_of_interest

    @region_of_interest.setter
    def region_of_interest(self, region_of_interest):
        """Set the region-of-interest mask after validating shape and length.

        Parameters:
        region_of_interest (array-like): Boolean-like vector with one entry per cell.

        Raises:
        ValueError: If the input is not one-dimensional or does not match the
            number of mesh cells.
        """
        region_of_interest = np.asarray(region_of_interest)
        if region_of_interest.ndim != 1:
            raise ValueError("Region of interest must be a one-dimensional vector")
        if len(region_of_interest) != len(self.mesh.cells()):
            raise ValueError(
                "Region of interest must have the same length as the number of mesh cells"
            )
        self._region_of_interest = region_of_interest.astype(bool)

    @property
    def neighbour_function(self):
        """The function used to calculate neighbours of a cell."""
        return self._neighbour_function

    @property
    def cell_neighbour_info(self):
        """The list of CellNeighbourInfo objects."""
        return self._cell_neighbour_info
    
    def show_region_of_interest(self, markersize=1, marker="o", mode="triang", ax=None):
        """Show the region of interest."""
        mesh = self.mesh
        cell_centers = mesh.cellCenters()
        region_of_interest = self.region_of_interest

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        else:
            fig = ax.get_figure()
        ax.set_title("Region of Interest")

        if mode == "scatter":
            x, y = cell_centers[:, 0], cell_centers[:, 1]
            ax.scatter(x[region_of_interest], y[region_of_interest], marker=marker, color="red", s=markersize)
            ax.scatter(x[~region_of_interest], y[~region_of_interest], marker=marker, color="blue", s=markersize)
            # Legend entries are created with dummy points.
            ax.plot(x[0], y[0], marker=marker, markersize=markersize, label="Region of Interest")
            ax.plot(x[0], y[0], marker=marker, markersize=markersize, label="Background")
            ax.legend(loc="upper right")
        elif mode == "triang":
            # drawModel keeps the plotting consistent with the rest of the package.
            im = drawModel(ax, mesh, data=region_of_interest, cMap="coolwarm", cMin=0, cMax=1, showMesh=False)
            if im is not None:
                fig.colorbar(im, ax=ax, label="Region of Interest", ticks=[0, 1], location="bottom")
            drawMeshBoundaries(ax, mesh, hideMesh=False, lw=0.2, color="black", fitView=False)
        return fig, ax
