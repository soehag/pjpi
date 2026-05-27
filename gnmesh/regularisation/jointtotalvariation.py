"""
This module contains the joint total variation regularisation term. This dual regularisation term is used to enforce the similarity between the two images by
punishing variations at different points in the images. The joint total variation is defined as follows:

JTV(m,n) = \int_{\Omega} \sqrt{|\nabla m(x)|_2^2 + |\nabla n(x)|_2^2} dx

where m and n are the two images, and \nabla m(x) and \nabla n(x) are the gradients of the images at point x. The joint total variation can be discretised by the Riemann sum:

JTV(m,n)    ~ \sum_{i=1}^{N} \sqrt{|\nabla m(x_i)|_2^2 + |\nabla n(x_i)|_2^2} A_i
            = \sum_{i=1}^{N} ( {|\nabla m(x_i)|_2^2 + |\nabla n(x_i)|_2^2}**0.25 * A_i**0.5 )**2

where A_i is the area of the pixel i.

The gradient of the joint total variation with respect to the images m and n can be computed as product of the following three terms:

1. The area term --- A_i**0.5 --- | --- A_i**0.5 ---
2. The outer derivative of the norm term 
--- 0.25 * (|\nabla m(x_i)|_2^2 + |\nabla n(x_i)|_2^2)**-0.75 --- | --- 0.25 * (|\nabla m(x_i)|_2^2 + |\nabla n(x_i)|_2^2)**-0.75 ---
3. The derivative of the norm term \| O @ m \|_2^2 is given by 2 * O.T @ O @ m, where O @ m is the numerical gradient of the image m.
    O.T has originally the shape m x (1/3) but here is smaller since a lot of the values are zero.

Note
----
This implementation has not been exhaustively tested. Use with caution and add unit tests for critical use cases.
"""

import numpy as np
from functools import partial
from .regularisationcore import RegularisationMultiModel
from gnmesh.meshtools.spatialgradient import calculate_spatial_gradient

def joint_total_variation_jacobian(physics_and_data, model_info_list, beta=1e-4, order=1):
    """
    Returns the Jacobian of the joint total variation regularisation term with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info_list : ModelInfo
        The list of model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the joint total variation regularisation term with respect to the model.
    """

    assert len(model_info_list) == 2, "The joint total variation regularisation term requires two models."

    mesh_info = model_info_list[0].mesh_info
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    area_of_cells_vector = np.tile(area_of_cells, 2)

    gradient_model_0 = model_info_list[0].spatial_gradient
    gradient_model_1 = model_info_list[1].spatial_gradient

    if gradient_model_0.ndim == 1:
        gradient_model_0_norm = np.abs(gradient_model_0)
        gradient_model_1_norm = np.abs(gradient_model_1)
    else:
        gradient_model_0_norm = np.linalg.norm(gradient_model_0, axis=1)
        gradient_model_1_norm = np.linalg.norm(gradient_model_1, axis=1)
        
    polynomial_derivative_vector = 0.25 * (gradient_model_0_norm**2 + gradient_model_1_norm**2+beta)**(-0.75)
    polynomial_derivative = np.tile(polynomial_derivative_vector, 2)

    joint_total_variation_jacobian_matrix = np.zeros((no_of_model_parameters, 2*no_of_model_parameters))

    for i in range(no_of_model_parameters):
        # Get cell neighbours of the i-th cell
        cell_indices, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)
        # Model 1
        matrix_model_0 = 2 * gradient_matrix.T @ gradient_model_0[i]
        joint_total_variation_jacobian_matrix[i, cell_indices] = matrix_model_0

        # Model 2
        matrix_model_1 = 2 * gradient_matrix.T @ gradient_model_1[i]
        joint_total_variation_jacobian_matrix[i, no_of_model_parameters + cell_indices] = matrix_model_1

    joint_total_variation_jacobian_matrix = joint_total_variation_jacobian_matrix * np.sqrt(area_of_cells_vector) * polynomial_derivative
    return joint_total_variation_jacobian_matrix

def joint_total_variation_phi(physics_and_data, model_info_list, model_transformation_regularisation_list=None, beta=1e-4, order=1):
    """
    Returns the joint total variation regularisation term.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info_list : ModelInfo
        The list of model information object.

    Returns
    -------
    float
        The joint total variation regularisation term.
    """

    assert len(model_info_list) == 2, "The joint total variation regularisation term requires two models."

    mesh_info = model_info_list[0].mesh_info
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    taylor_order = [model.taylor_order for model in model_info_list]
    assert np.all([to==order for to in taylor_order]), "Model and regularisation order must be the same."
    assert model_transformation_regularisation_list is None or len(model_transformation_regularisation_list) == 2,\
    "The joint total variation regularisation term requires two model transformation regularisation terms."

    if model_transformation_regularisation_list is None:
        model_transformation_regularisation_list = [None, None]

    # Get the first model spatial gradient
    if model_transformation_regularisation_list[0] is not None:
        gradient_model_0 = calculate_spatial_gradient(
            model=model_transformation_regularisation_list[0].forward(model_info_list[0].model),
            mesh_info=model_info_list[0].mesh_info,
            taylor_order=order
        )
    else:
        gradient_model_0 = model_info_list[0].spatial_gradient

    # Get the second model spatial gradient
    if model_transformation_regularisation_list[1] is not None:
        gradient_model_1 = calculate_spatial_gradient(
            model=model_transformation_regularisation_list[1].forward(model_info_list[1].model),
            mesh_info=model_info_list[1].mesh_info,
            taylor_order=order
        )
    else:
        gradient_model_1 = model_info_list[1].spatial_gradient

    if gradient_model_0.ndim == 1:
        gradient_model_0_norm = np.abs(gradient_model_0)
        gradient_model_1_norm = np.abs(gradient_model_1)
    else:
        gradient_model_0_norm = np.linalg.norm(gradient_model_0, axis=1)
        gradient_model_1_norm = np.linalg.norm(gradient_model_1, axis=1)

    phi = (gradient_model_0_norm**2 + gradient_model_1_norm**2 + beta)**0.25 * np.sqrt(area_of_cells)

    return phi

class JointTotalVariation(RegularisationMultiModel):
    """
    The joint total variation regularisation term.
    """

    def __init__(self, regularisation_beta=1e-4, order =1):
        self._regularisation_beta = regularisation_beta
        self._order = order
        calculate_jacobian = partial(joint_total_variation_jacobian, beta = self._regularisation_beta, order=order)
        calculate_phi = partial(joint_total_variation_phi, beta = self._regularisation_beta, order=order)
        super().__init__(
            calculate_jacobian=calculate_jacobian,
            calculate_phi=calculate_phi,
            static_jacobian=False,
        )
