"""
This module contains the smoothing regularisation terms derived from pygimli routines.
The smoothing regularisation terms are used to smooth the model.

The smoothing term is given by:
    ||L @ m||^2
where m is the model and L is the smoothing operator.

This module contains the following smoothing regularisation terms:

1. FirstOrderSmoothing: Smoothing with a first order operator.
2. SecondOrderSmoothing: Smoothing with a second order operator.
"""

import numpy as np
import pygimli as pg
from .regularisationcore import Regularisation

# Smoothing with a first order operator
def first_order_smoothing_jacobian(physics_and_data, model_info):
    """
    Returns the Jacobian of the first order smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the smoothing operator with respect to the model.
    """
    forward_operator = physics_and_data.method_manager.fop
    mesh_copy = model_info.mesh_info.mesh.copy()
    for cell in mesh_copy.cells():
        cell.setMarker(1)
    forward_operator.setMesh(mesh_copy)
    region_manager = forward_operator.regionManager()
    region_manager.setConstraintType(1)
    jacobian = forward_operator.createConstraints()
    if isinstance(jacobian, pg.SparseMapMatrix):
        jacobian = pg.utils.sparseMatrix2Dense(jacobian)
    return jacobian

def first_order_smoothing_phi(physics_and_data, model_info, model_transformation_regularisation):
    """
    Returns the right-hand side of the first order smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the smoothing operator with respect to the model.
    """
    jacobian = first_order_smoothing_jacobian(physics_and_data, model_info)
    if model_transformation_regularisation is not None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model
    return jacobian @ model_vector

class FirstOrderSmoothing(Regularisation):
    """
    Smoothing with a first order operator.
    """
    def __init__(self):
        super().__init__(
            calculate_jacobian=first_order_smoothing_jacobian,
            calculate_phi=first_order_smoothing_phi,
            static_jacobian=True,
        )

# Smoothing with a second order operator
def second_order_smoothing_jacobian(physics_and_data, model_info):
    """
    Returns the Jacobian of the second order smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the smoothing operator with respect to the model.
    """
    forward_operator = physics_and_data.method_manager.fop
    mesh_copy = model_info.mesh_info.mesh.copy()
    for cell in mesh_copy.cells():
        cell.setMarker(1)
    forward_operator.setMesh(mesh_copy)
    region_manager = forward_operator.regionManager()
    region_manager.setConstraintType(2)
    jacobian = forward_operator.createConstraints()
    if isinstance(jacobian, pg.SparseMapMatrix):
        jacobian = pg.utils.sparseMatrix2Dense(jacobian)
    return jacobian

def second_order_smoothing_phi(physics_and_data, model_info, model_transformation_regularisation):
    """
    Returns the right-hand side of the second order smoothing operator with respect to the model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the smoothing operator with respect to the model.
    """
    jacobian = second_order_smoothing_jacobian(physics_and_data, model_info)
    if model_transformation_regularisation is not None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model
    return jacobian @ model_vector

class SecondOrderSmoothing(Regularisation):
    """
    Smoothing with a second order operator.
    """
    def __init__(self):
        super().__init__(
            calculate_jacobian=second_order_smoothing_jacobian,
            calculate_phi=second_order_smoothing_phi,
            static_jacobian=True,
        )