"""Smoothing regularisation utilities (spatial).

Note
----
Some functions in this module are not exhaustively tested. Verify results
and add unit tests for workflows that rely on these operators.
"""

import numpy as np
from functools import partial
from .regularisationcore import Regularisation
from gnmesh.meshtools.spatialgradient import calculate_spatial_gradient, calculate_hessian_matrix, calculate_laplacian_from_hessian_matrix_model

def first_order_smoothing_squared_gradient_jacobian(physics_and_data, model_info, order=1):
    """
    Returns the partial Jacobian of the first order squared smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the smoothing operator with respect to the model.
    """
    assert model_info.taylor_order == order, "The taylor order of the model_info object must match the order parameter."

    mesh_info = model_info.mesh_info
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )
    
    jacobian_matrix = np.zeros((no_of_model_parameters, no_of_model_parameters))

    spatial_gradient = model_info.spatial_gradient

    for i in range(no_of_model_parameters):
        # Get the sensitivities of the gradient of the model with respect to the mesh
        cell_neighbours, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)

        jacobian_matrix[i, cell_neighbours] = np.dot(spatial_gradient[i], gradient_matrix)

    jacobian_matrix = jacobian_matrix * np.sqrt(area_of_cells)
    return jacobian_matrix

def first_order_smoothing_squared_gradient_phi(physics_and_data, model_info, model_transformation_regularisation, order=1):
    """
    Returns the partial right-hand side of the first order squared smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the smoothing operator with respect to the model.
    """
    # jacobian = first_order_smoothing_squared_gradient_jacobian(physics_and_data, model_info, order=order)
    # if model_transformation_regularisation is not None:
    #     model_vector = model_transformation_regularisation.forward(model_info.model)
    # else:
    #     model_vector = model_info.model
    # return jacobian @ model_vector
    
    if model_transformation_regularisation is not None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model

    spatial_gradient = calculate_spatial_gradient(
        model=model_vector,
        mesh_info=model_info.mesh_info,
        taylor_order=order
    )
    return np.linalg.norm(spatial_gradient, axis=1)**4

class FirstOrderSmoothingSquaredGradient(Regularisation):
    """
    Smoothing with a first order operator.
    """
    def __init__(self, order=1):
        assert order in [1, 2], "The order parameter must be 1 or 2."
        self._order = order

        first_order_jacobian_object = partial(first_order_smoothing_squared_gradient_jacobian, order=order)
        first_order_phi_object = partial(first_order_smoothing_squared_gradient_phi, order=order)

        super().__init__(
            calculate_jacobian=first_order_jacobian_object,
            calculate_phi=first_order_phi_object,
            static_jacobian=False,
        )

def first_order_smoothing_gradient_jacobian(physics_and_data, model_info, order=1):
    """
    Returns the partial Jacobian of the first order smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the smoothing operator with respect to the model.
    """
    assert model_info.taylor_order == order, "The taylor order of the model_info object must match the order parameter."

    mesh_info = model_info.mesh_info
    dimension = mesh_info.dimension

    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )
    
    jacobian_matrix = np.zeros((dimension * no_of_model_parameters, no_of_model_parameters))

    spatial_gradient = model_info.spatial_gradient

    for i in range(no_of_model_parameters):
        # Get the sensitivities of the gradient of the model with respect to the mesh
        cell_neighbours, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)

        jacobian_matrix[2*i:2*(i+1), cell_neighbours] = gradient_matrix

    jacobian_matrix = jacobian_matrix * np.sqrt(area_of_cells)
    return jacobian_matrix

def first_order_smoothing_gradient_phi(physics_and_data, model_info, model_transformation_regularisation, order=1):
    """
    Returns the partial right-hand side of the first order squared smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the smoothing operator with respect to the model.
    """
    # jacobian = first_order_smoothing_squared_gradient_jacobian(physics_and_data, model_info, order=order)
    # if model_transformation_regularisation is not None:
    #     model_vector = model_transformation_regularisation.forward(model_info.model)
    # else:
    #     model_vector = model_info.model
    # return jacobian @ model_vector

    area_of_cells = np.array(
        [cni.cell_area for cni in model_info.mesh_info.cell_neighbour_info]
    )

    if not model_transformation_regularisation is None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model

    spatial_gradient = calculate_spatial_gradient(
        model=model_vector,
        mesh_info=model_info.mesh_info,
        taylor_order=order
    )
    spatial_gradient = spatial_gradient * np.sqrt(area_of_cells)[:, None]
    return spatial_gradient.flatten()

class FirstOrderSmoothingGradient(Regularisation):
    """
    Smoothing with a first order operator.
    """
    def __init__(self, order=1):
        assert order in [1, 2], "The order parameter must be 1 or 2."
        self._order = order

        first_order_jacobian_object = partial(first_order_smoothing_gradient_jacobian, order=order)
        first_order_phi_object = partial(first_order_smoothing_gradient_phi, order=order)

        super().__init__(
            calculate_jacobian=first_order_jacobian_object,
            calculate_phi=first_order_phi_object,
            static_jacobian=True,
        )

def laplacian_smoothing_jacobian(physics_and_data, model_info, order=2):
    """
    Returns the partial Jacobian of the Laplacian smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the smoothing operator with respect to the model.
    """
    assert model_info.taylor_order == order, "The taylor order of the model_info object must match the order parameter."
    assert order == 2, "The order parameter must be 2."

    mesh_info = model_info.mesh_info
    dimension = mesh_info.dimension

    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    jacobian_matrix = np.zeros((no_of_model_parameters, no_of_model_parameters))


    for i in range(no_of_model_parameters):
        # Get the sensitivities of the gradient of the model with respect to the mesh
        cell_neighbours, hessian_matrix_sensitivities = mesh_info.cell_neighbour_info[i].get_hessian_mesh_sensitivities()

        jacobian_matrix[i, cell_neighbours] = np.sum(hessian_matrix_sensitivities, axis=0)

    jacobian_matrix = jacobian_matrix * np.sqrt(area_of_cells)
    return jacobian_matrix

def laplacian_smoothing_phi(physics_and_data, model_info, model_transformation_regularisation, order=2):
    """
    Returns the partial right-hand side of the Laplacian smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    order : int, optional
        The order of the taylor expansion. The default is 1.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the smoothing operator with respect to the model.
    """
    assert order >= 2, "The order parameter must be at least 2."

    if not model_transformation_regularisation is None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model

    area_of_cells = np.array(
        [cni.cell_area for cni in model_info.mesh_info.cell_neighbour_info]
    )

    hessian_matrix = calculate_hessian_matrix(
        model=model_vector,
        mesh_info=model_info.mesh_info,
        taylor_order=order
    )

    laplacian = calculate_laplacian_from_hessian_matrix_model(
        hessian_matrix_model=hessian_matrix,
    )

    return np.sqrt(area_of_cells) * laplacian

class LaplacianSmoothing(Regularisation):
    """
    Smoothing with a Laplacian operator.
    """
    def __init__(self, order=2):
        assert order == 2, "The order parameter must be 2."
        self._order = order

        laplacian_jacobian_object = partial(laplacian_smoothing_jacobian, order=order)
        laplacian_phi_object = partial(laplacian_smoothing_phi, order=order)

        super().__init__(
            calculate_jacobian=laplacian_jacobian_object,
            calculate_phi=laplacian_phi_object,
            static_jacobian=False,
        )