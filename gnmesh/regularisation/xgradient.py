"""
This module contains the implementation of the XGradient regularisation class, which is a
regularisation technique that aims for structural similarity by aligning the gradients of the
input and output images.

The XGradient regularisation class is implemented as a subclass of the Regularisation class.

For two models m and d, the continous XGradient regularisation term is defined as:
    \int ||\nabla m \cross \nabla n||^2 dV

The discrete XGradient regularisation term is given by discretising the integral by a quadrature rule/sum:
    \sum_{i} (\sqrt(A(i)*||\nabla m_i \cross \nabla n_i||_2^2)**2  where A(i) is the area of the i-th cell.

    where \\nabla m_i is the gradient of the model at the i-th cell and \\nabla n_i is the gradient of the data at the i-th cell.

The gradient of the XGradient regularisation term is given by three terms:
    1. The area term \sqrt(A(i)) | \sqrt(A(i))
    2. The derivative of the norm term 2 * \\nabla m_i \cross \\nabla n_i | 2 * \\nabla m_i \cross \\nabla n_i
    3. The derivative of the cross product term \nabla m_i \cross \nabla n_i. With the gradients given as 
    \nabla m_i = O @ m and \nabla n_i = O @ n, the derivative of the cross product term is given by:
    \nabla (\nabla m_i) \cross \nabla n_i + \nabla m_i \cross \nabla (\nabla n_i)
    = O \cross \nabla n_i | \nabla m_i \cross O

Since the cross product is bilinear, the derivative is given by the 
"""

import numpy as np
from functools import partial
from .regularisationcore import RegularisationMultiModel, Regularisation
from gnmesh.meshtools.spatialgradient import calculate_spatial_gradient

def xgradient_jacobian(physics_and_data, model_info_list, order=1):
    """
    Returns the Jacobian of the XGradient regularisation term with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info_list : ModelInfo
        The list of model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the XGradient regularisation term with respect to the model.
    """

    assert len(model_info_list) == 2, "The XGradient regularisation term requires two models."

    mesh_info = model_info_list[0].mesh_info
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    area_of_cells_vector = np.tile(area_of_cells, 2)

    #! Is a transformation missing here? - For sure but not used yet
    gradient_model_0 = model_info_list[0].spatial_gradient
    gradient_model_1 = model_info_list[1].spatial_gradient

    xgradient = np.cross(gradient_model_0, gradient_model_1)

    xgradient_jacobian_matrix = np.zeros((no_of_model_parameters, 2*no_of_model_parameters))

    for i in range(no_of_model_parameters):
        # Get the sensitivities of the gradient of the model with respect to the mesh
        cell_neighbours, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)
        gradient_matrix_model_0 = np.cross(gradient_matrix.T, gradient_model_1[i])
        gradient_matrix_model_1 = np.cross(gradient_model_0[i], gradient_matrix.T)

        # Model 1
        xgradient[i] = 1.0
        if gradient_matrix_model_0.ndim == 1:
            xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] * gradient_matrix_model_0
        else:
            xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] @ gradient_matrix_model_0

        # Model 2
        if gradient_matrix_model_1.ndim == 1:
            xgradient_jacobian_matrix[i, no_of_model_parameters + cell_neighbours] = xgradient[i] * gradient_matrix_model_1
        else:
            xgradient_jacobian_matrix[i, no_of_model_parameters + cell_neighbours] = xgradient[i] @ gradient_matrix_model_1

    xgradient_jacobian_matrix = np.sqrt(area_of_cells_vector) * xgradient_jacobian_matrix
    return xgradient_jacobian_matrix

def xgradient_phi(physics_and_data, model_info_list, model_transformation_regularisation_list=None, order=1):
    assert len(model_info_list) == 2, "The XGradient regularisation term requires two models."

    mesh_info = model_info_list[0].mesh_info

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    taylor_orders = [model_info.taylor_order for model_info in model_info_list]
    assert all([taylor_order == order for taylor_order in taylor_orders]), "The model info objects must have the same taylor order as the regularisation term."
    assert model_transformation_regularisation_list is None or len(model_transformation_regularisation_list) == 2, "The XGradient regularisation term requires two model transformation regularisation objects."

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

    xgradient = np.cross(gradient_model_0, gradient_model_1)

    if xgradient.ndim == 2:
        norm_of_xgradient = np.linalg.norm(xgradient, axis=1)
    else:
        norm_of_xgradient = xgradient
    phi = np.sqrt(area_of_cells) * norm_of_xgradient
    return phi

class XGradient(RegularisationMultiModel):
    """
    The XGradient regularisation class."""

    def __init__(self, weight=1.0, order=1):
        assert order in [1, 2], "The XGradient regularisation term only supports first and second order derivatives."
        self._order = order

        xgradient_jacobian_obj = partial(xgradient_jacobian, order=order)
        xgradient_phi_obj = partial(xgradient_phi, order=order)
        super().__init__(
            calculate_jacobian=xgradient_jacobian_obj,
            calculate_phi=xgradient_phi_obj,
            static_jacobian=False,
            weight=weight,
        )

# This class implements the cross gradient regularisation term for two models with respect to two reference models
class XGradientReferenceModel(RegularisationMultiModel):
    """
    The XGradient regularisation class."""

    def __init__(self, reference_model_list, weight=1.0, order=1):
        assert order in [1, 2], "The XGradient regularisation term only supports first and second order derivatives."
        assert reference_model_list is None or len(reference_model_list) == 2, "The XGradient regularisation term requires two reference models."
        self._order = order
        if reference_model_list is None:
            reference_model_list = [None, None]
        self._reference_model_list = reference_model_list

        def xgradient_reference_model_jacobian(physics_and_data, model_info_list, order=1):
            """
            Returns the Jacobian of the XGradient regularisation term (difference to reference model) with respect to the model.

            Parameters
            ----------
            physics_and_data : PhysicsAndData
                The physics and data object.
            model_info_list : ModelInfo
                The list of model information object.

            Returns
            -------
            numpy.ndarray
                The Jacobian of the XGradient regularisation term with respect to the model.
            """

            assert len(model_info_list) == 2, "The XGradient regularisation term requires two models."

            mesh_info = model_info_list[0].mesh_info
            no_of_model_parameters = mesh_info.mesh.cellCount()

            area_of_cells = np.array(
                [cni.cell_area for cni in mesh_info.cell_neighbour_info]
            )

            area_of_cells_vector = np.tile(area_of_cells, 2)

            #! Is a transformation missing here? - For sure but not used yet
            gradient_model_0 = model_info_list[0].spatial_gradient
            gradient_model_1 = model_info_list[1].spatial_gradient

            gradient_model_0_reference = calculate_spatial_gradient(
                model=self._reference_model_list[0].model,
                mesh_info=model_info_list[0].mesh_info,
                taylor_order=order
            )
            gradient_model_1_reference = calculate_spatial_gradient(
                model=self._reference_model_list[1].model,
                mesh_info=model_info_list[1].mesh_info,
                taylor_order=order
            )

            xgradient = np.cross(gradient_model_0, gradient_model_1)

            xgradient_jacobian_matrix = np.zeros((no_of_model_parameters, 2*no_of_model_parameters))

            for i in range(no_of_model_parameters):
                # Get the sensitivities of the gradient of the model with respect to the mesh
                cell_neighbours, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)
                gradient_matrix_model_0 = np.cross(gradient_matrix.T, gradient_model_1[i] - gradient_model_1_reference[i])
                gradient_matrix_model_1 = np.cross(gradient_model_0[i] - gradient_model_0_reference[i], gradient_matrix.T)

                # Model 1
                xgradient[i] = 1.0
                if gradient_matrix_model_0.ndim == 1:
                    xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] * gradient_matrix_model_0
                else:
                    xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] @ gradient_matrix_model_0

                # Model 2
                if gradient_matrix_model_1.ndim == 1:
                    xgradient_jacobian_matrix[i, no_of_model_parameters + cell_neighbours] = xgradient[i] * gradient_matrix_model_1
                else:
                    xgradient_jacobian_matrix[i, no_of_model_parameters + cell_neighbours] = xgradient[i] @ gradient_matrix_model_1

            xgradient_jacobian_matrix = np.sqrt(area_of_cells_vector) * xgradient_jacobian_matrix
            return xgradient_jacobian_matrix


        def xgradient_reference_model_phi(physics_and_data, model_info_list, model_transformation_regularisation_list=None, order=1):
            assert len(model_info_list) == 2, "The XGradient regularisation term requires two models."

            mesh_info = model_info_list[0].mesh_info

            area_of_cells = np.array(
                [cni.cell_area for cni in mesh_info.cell_neighbour_info]
            )

            taylor_orders = [model_info.taylor_order for model_info in model_info_list]
            assert all([taylor_order == order for taylor_order in taylor_orders]), "The model info objects must have the same taylor order as the regularisation term."
            assert model_transformation_regularisation_list is None or len(model_transformation_regularisation_list) == 2, "The XGradient regularisation term requires two model transformation regularisation objects."

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

            # Get the first reference model spatial gradient
            if self._reference_model_list[0] is not None:
                if model_transformation_regularisation_list[0] is not None:
                    gradient_reference_model_0 = calculate_spatial_gradient(
                        model=model_transformation_regularisation_list[0].forward(self._reference_model_list[0].model),
                        mesh_info=model_info_list[0].mesh_info,
                        taylor_order=order
                    )
                else:
                    gradient_reference_model_0 = calculate_spatial_gradient(
                        model=self._reference_model_list[0].model,
                        mesh_info=model_info_list[0].mesh_info,
                        taylor_order=order
                    )
            else:
                gradient_reference_model_0 = np.zeros_like(model_info_list[0].spatial_gradient)

            # Get the second reference model spatial gradient
            if self._reference_model_list[1] is not None:
                if model_transformation_regularisation_list[1] is not None:
                    gradient_reference_model_1 = calculate_spatial_gradient(
                        model=model_transformation_regularisation_list[1].forward(self._reference_model_list[1].model),
                        mesh_info=model_info_list[1].mesh_info,
                        taylor_order=order
                    )
                else:
                    gradient_reference_model_1 = calculate_spatial_gradient(
                        model=self._reference_model_list[1].model,
                        mesh_info=model_info_list[1].mesh_info,
                        taylor_order=order
                    )
            else:
                gradient_reference_model_1 = np.zeros_like(model_info_list[1].spatial_gradient)

            #* Get the cross product of the gradients
            xgradient_to_reference = np.cross(gradient_model_0-gradient_reference_model_0, gradient_model_1- gradient_reference_model_1)
            #* Get the norm of the cross product
            if xgradient_to_reference.ndim == 2:
                norm_of_xgradient = np.linalg.norm(xgradient_to_reference, axis=1)**2
            else:
                norm_of_xgradient = np.abs(xgradient_to_reference)**2
            phi = np.sqrt(area_of_cells) * norm_of_xgradient
            return phi

        xgradient_jacobian_obj = partial(xgradient_reference_model_jacobian, order=order)
        xgradient_phi_obj = partial(xgradient_reference_model_phi, order=order)
        super().__init__(
            calculate_jacobian=xgradient_jacobian_obj,
            calculate_phi=xgradient_phi_obj,
            static_jacobian=False,
            weight=weight,
        )

# This class implements the cross gradient regularisation term for a single model with respect to a fixed second model

def xgradient_single_model_jacobian(physics_and_data, model_info, fixed_model_info, order=1):
    """
    Returns the Jacobian of the XGradient regularisation term with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.
    fixed_model_info : ModelInfo
        The fixed model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the XGradient regularisation term with respect to the model.
    """
    taylor_orders = [model_info.taylor_order, fixed_model_info.taylor_order]
    assert all([taylor_order == order for taylor_order in taylor_orders]), "The model info objects must have the same taylor order as the regularisation term."

    mesh_info = model_info.mesh_info
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    gradient_model = model_info.spatial_gradient
    gradient_fixed_model = fixed_model_info.spatial_gradient

    xgradient = np.cross(gradient_model, gradient_fixed_model)

    xgradient_jacobian_matrix = np.zeros((no_of_model_parameters, no_of_model_parameters))

    for i in range(no_of_model_parameters):
        # Get the sensitivities of the gradient of the model with respect to the mesh
        cell_neighbours, gradient_matrix = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities(order=order)

        gradient_matrix_model = np.cross(gradient_matrix.T, gradient_fixed_model[i])

        if gradient_matrix_model.ndim == 1:
            xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] * gradient_matrix_model
        else:
            xgradient_jacobian_matrix[i, cell_neighbours] =  xgradient[i] @ gradient_matrix_model

    xgradient_jacobian_matrix = np.sqrt(area_of_cells) * xgradient_jacobian_matrix
    return xgradient_jacobian_matrix

def xgradient_single_model_phi(physics_and_data, model_info, fixed_model_info, model_transformation_regularisation=None, order=1):
    taylor_orders = [model_info.taylor_order, fixed_model_info.taylor_order]
    assert all([taylor_order == order for taylor_order in taylor_orders]), "The model info objects must have the same taylor order as the regularisation term."

    mesh_info = model_info.mesh_info

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    # Get the first model spatial gradient
    if not model_transformation_regularisation is None:
        gradient_model = calculate_spatial_gradient(
            model=model_transformation_regularisation.forward(model_info.model),
            mesh_info=model_info.mesh_info,
            taylor_order=order
        )
    else:
        gradient_model = model_info.spatial_gradient

    # Get the second model spatial gradient
    gradient_fixed_model = fixed_model_info.spatial_gradient

    xgradient = np.cross(gradient_model, gradient_fixed_model)

    if xgradient.ndim == 2:
        norm_of_xgradient = np.linalg.norm(xgradient, axis=1)**2
    else:
        norm_of_xgradient = np.abs(xgradient)**2
    phi = np.sqrt(area_of_cells) * norm_of_xgradient
    return phi

class XGradientSingleModel(Regularisation):
    """
    The XGradientSingleModel regularisation class."""

    def __init__(self, fixed_model_info, order=1):
        assert order in [1, 2], "The XGradient regularisation term only supports first and second order derivatives."
        self._order = order

        xgradient_jacobian_obj = partial(xgradient_single_model_jacobian, fixed_model_info=fixed_model_info, order=order)
        xgradient_phi_obj = partial(xgradient_single_model_phi, fixed_model_info=fixed_model_info, order=order)
        super().__init__(
            calculate_jacobian=xgradient_jacobian_obj,
            calculate_phi=xgradient_phi_obj,
            static_jacobian=False,
        )