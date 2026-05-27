"""
This module contains the LinearOperator regularisation term. The linear operator regularisation term is given by minimising
the following expression:
    ||L @ m - b||^2

where L is a linear operator and b is a vector. The linear operator / matrix as well as the RHS vector are fix and stored
in the regularisation object.

This module contains the following linear operator regularisation terms:

1. LinearOperator: Linear operator regularisation term.
"""

import numpy as np
from .regularisationcore import Regularisation

class LinearOperator(Regularisation):
    """
    Linear operator regularisation term.
    """
    def __init__(self, linear_operator, rhs_vector):
        """
        Constructor.

        Parameters
        ----------
        linear_operator : numpy.ndarray
            The linear operator / matrix.
        rhs_vector : numpy.ndarray
            The right-hand side vector.
        """
        self._linear_operator = linear_operator.copy()
        self._rhs_vector = rhs_vector.copy()

        def calculate_jacobian(physics_and_data, model_info):
            return self._linear_operator
        
        def calculate_phi(physics_and_data, model_info, model_transformation_regularisation):
            if model_transformation_regularisation is not None:
                model_vector = model_transformation_regularisation.forward(model_info.model)
            else:
                model_vector = model_info.model
            return self._linear_operator @ model_vector - self._rhs_vector
        
        super().__init__(
            calculate_jacobian=calculate_jacobian,
            calculate_phi=calculate_phi,
            static_jacobian=True,
        )
