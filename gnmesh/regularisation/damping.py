"""
This module contains the damping regularisation terms.
The damping regularisation terms are used to damp the model towards a reference model or
the old model.

The damping term is given by:
    ||m - m_ref||^2
where m is the model and m_ref is the reference model.

This module contains the following damping regularisation terms:

1. DampingReferenceModel: Damping towards a fixed reference model.
2. DampingStepWidth: Damping towards the old model.
"""

import numpy as np
from .regularisationcore import Regularisation

# Damping to a fixed reference model
def damping_reference_model_jacobian(physics_and_data, model_info):
    """
    Returns the Jacobian of the damping term with respect to a reference model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the damping term with respect to a reference model.
    """
    mesh_size = model_info.mesh_info.mesh.cellCount()
    return np.eye(mesh_size)

def damping_reference_model_phi(physics_and_data, model_info, model_transformation_regularisation):
    """
    Returns the right-hand side of the damping term with respect to a reference model.
    The reference model is stored in the physics_and_data object and given in transformed form.
    I.e. if the model is given as slowness, the model_tranformation_regularisation is given as inverse to 
    compare velocities, the reference model is treated as velocities.q

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the damping term with respect to a reference model.
    """
    assert  hasattr(physics_and_data, "reference_model"), \
            "The physics_and_data object does not have a reference_model attribute."
    
    if model_transformation_regularisation is not None:
        model_vector = model_transformation_regularisation.forward(model_info.model)
    else:
        model_vector = model_info.model
    return model_vector - physics_and_data.reference_model

class DampingReferenceModel(Regularisation):
    """
    Damping towards a fixed reference model.
    """
    def __init__(self):
        super().__init__(
            calculate_jacobian=damping_reference_model_jacobian,
            calculate_phi=damping_reference_model_phi,
            static_jacobian=True,
        )

# Damping of the step_width i.e. damping towards the old model
def damping_step_width_jacobian(physics_and_data, model_info):
    """
    Returns the Jacobian of the damping term with respect to the "old" model.

    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The Jacobian of the damping term with respect to the "old" model.
    """
    mesh_size = model_info.mesh_info.mesh.cellCount()
    return np.eye(mesh_size)

def damping_step_width_phi(physics_and_data, model_info, model_transformation_regularisation):
    """
    Returns the right-hand side of the damping term with respect to the "old" model.
    
    Parameters
    ----------
    physics_and_data : PhysicsAndData
        The physics and data object.
    model_info : ModelInfo
        The model information object.

    Returns
    -------
    numpy.ndarray
        The right-hand side of the damping term with respect to the "old" model.
    """
    return np.zeros_like(model_info.model)

class DampingStepWidth(Regularisation):
    """
    Damping towards the old model.
    """
    def __init__(self):
        super().__init__(
            calculate_jacobian=damping_step_width_jacobian,
            calculate_phi=damping_step_width_phi,
            static_jacobian=True,
        )