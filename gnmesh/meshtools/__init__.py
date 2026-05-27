##################################################
# This is the init-file for the package 'gnmesh.meshtools'.
# Author: Hagen Söding
# Address: hagen.soeding@eaps.ethz.ch
# Date: 2026/05/26
##################################################

from . import meshinfo
from . import modelinfo
from . import spatialgradient
from . import transformation

from .meshinfo import (
    CellNeighbourInfo,
    MeshInfo,
    cell_area_triangle,
    distance_to_neighbour_list_for_cell,
    distance_to_neighbour_list_for_mesh,
    get_n_closest_neighbours,
    get_n_closest_neighbours_function_for_mesh,
)
from .modelinfo import ModelInfo, ModelInfoMixedGeoPetro
from .spatialgradient import (
    calculate_hessian_matrix,
    calculate_laplacian_from_hessian_matrix_model,
    calculate_spatial_gradient,
    plot_absolute_value_of_gradient_field,
    plot_absolute_value_of_gradient_field_from_vectors,
    plot_gradient_field,
    plot_hessian_matrix_overview,
    quadratic_indices_to_vector_indices,
    vector_indices_to_quadratic_indices,
)
from .transformation import (
    InverseTransformation,
    LogarithmicBarrierTransformationGreaterThan,
    LogarithmicBarrierTransformationLessThan,
    LogarithmicBarrierTransformationTwoSided,
    MultiplicativeTransformation,
    PowerTransformation,
)

__all__ = [
    'meshinfo',
    'modelinfo',
    'spatialgradient',
    'transformation',
    'CellNeighbourInfo',
    'MeshInfo',
    'cell_area_triangle',
    'distance_to_neighbour_list_for_cell',
    'distance_to_neighbour_list_for_mesh',
    'get_n_closest_neighbours',
    'get_n_closest_neighbours_function_for_mesh',
    'ModelInfo',
    'ModelInfoMixedGeoPetro',
    'calculate_hessian_matrix',
    'calculate_laplacian_from_hessian_matrix_model',
    'calculate_spatial_gradient',
    'plot_absolute_value_of_gradient_field',
    'plot_absolute_value_of_gradient_field_from_vectors',
    'plot_gradient_field',
    'plot_hessian_matrix_overview',
    'quadratic_indices_to_vector_indices',
    'vector_indices_to_quadratic_indices',
    'InverseTransformation',
    'LogarithmicBarrierTransformationGreaterThan',
    'LogarithmicBarrierTransformationLessThan',
    'LogarithmicBarrierTransformationTwoSided',
    'MultiplicativeTransformation',
    'PowerTransformation',
]