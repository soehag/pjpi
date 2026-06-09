"""
This module contains the core functionality for regularisation terms.

The regularisation term is discretised as Phi(m) = weight * sum_{i=1}^{N} (phi_i(m))^2=0
where phi_i is the i-th regularisation functional and N is the number of regularisation functionals.

This module supports the following regularisation schemes:
- Creeping scheme: m_new - m_old = (J^T @ J)^{-1} @ J^T @ (0-Phi(m_old))
- Jumping scheme: m_new = (J^T @ J)^{-1} @ (J^T @ ( 0 - Phi(m_old) + J @ m_old))

These reguarisation schemes can be expressed by the equivalent equations in a least-squares sense:
- Creeping scheme: J @ (m_new - m_old) = (0 - Phi(m_old))
- Jumping scheme: J @ m_new = (0 - Phi(m_old)) + J @ m_old

The regularisation term is weighted by the weight attribute of the regularisation object.

This file contains the following classes:
- Regularisation: A class representing a regularisation term of the form Phi(m)=0.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def remove_entries_below_thresh_from_coo(coo_matrix, relativ_threshold):
    """
    This functions removes entries from a COO matrix that are below a certain threshold.
    The threshold is relative to the maximum entry of the matrix.

    Parameters:
    - coo_matrix: The COO matrix.
    - relativ_threshold: The relative threshold.

    Returns:
    - coo_matrix: The COO matrix with removed entries.
    """
    coo_matrix_new = coo_matrix.copy()
    if coo_matrix_new.nnz == 0:
        return coo_matrix_new
    max_entry = np.max(np.abs(coo_matrix_new.data))
    mask = np.abs(coo_matrix_new.data) > relativ_threshold * max_entry
    coo_matrix_new.data = coo_matrix_new.data[mask]
    coo_matrix_new.row = coo_matrix_new.row[mask]
    coo_matrix_new.col = coo_matrix_new.col[mask]
    return coo_matrix_new

def remove_columns_from_csr(csr_matrix, columns_to_remove):
    """
    Removes columns from a csr matrix.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The csr matrix to remove columns from.
    columns_to_remove : list, np.array

    Returns
    -------
    scipy.sparse.csr_matrix
        The csr matrix with the columns removed.
    """
    if all(isinstance(ent, bool) for ent in columns_to_remove):
        columns_to_remove = np.where(columns_to_remove)[0]
    
    columns_to_keep = np.setdiff1d(np.arange(csr_matrix.shape[1]), columns_to_remove)
    # Define coo matrix for multiplication
    data_vector = np.ones(len(columns_to_keep))
    row_vector = columns_to_keep
    col_vector = np.arange(len(columns_to_keep))

    coo_matrix = sp.sparse.coo_matrix(
        (data_vector, (row_vector, col_vector)),
        shape=(csr_matrix.shape[1], len(columns_to_keep))
    )
    return csr_matrix.dot(coo_matrix)

class Regularisation:
    """
    This class represents a regularisation term of the form Phi(m)=0.
    The regularisation term is discretised as Phi(m) = weight * sum_{i=1}^{N} (phi_i(m))^2=0
    where phi_i is the i-th regularisation functional and N is the number of regularisation
    functionals.

    The Gauss-Newton formulation is based on the 1st order Taylor expansion of the
    regularisation functional.
    
    Attributes:
    - weight: The weight of the regularisation term.
    - model_transformation_regularisation: The transformation of the model that is applied before the application
        of the regularisation. Setting the transformation to x^2 mean applying the regularisation
        to the square of the model.
    - _calculate_jacobian: The function to calculate the Jacobian.
    - _calculate_phi: The function to calculate the regularisation term.
    - _static_jacobian: A flag to indicate if the Jacobian is static.
    - _static_jacobian_saved: The saved static Jacobian.

    The function _calculate_jacobian must have the following signature:
    - _calculate_jacobian(physics_and_data, model_info)

    The function _calculate_phi must have the following signature:
    - _calculate_phi(physics_and_data, model_info, model_transformation_regularisation)
    """
    #*calculate_jacobian / calculate residual has signatur (physics_and_data , model_info)
    def __init__(
            self,
            calculate_jacobian,
            calculate_phi,
            static_jacobian,
            weight=1.0,
            model_transformation_regularisation=None,
            ):
        # Initialise functions to calculate jacobian and rhs
        self._calculate_jacobian  = calculate_jacobian
        self._calculate_phi = calculate_phi

        # Initialise flags for static jacobian or rhs
        self._static_jacobian = static_jacobian
        self._static_jacobian_saved = None

        # Intialise weight and transformation
        self.weight = weight
        self.model_transformation_regularisation = model_transformation_regularisation

    def get_jacobian(self, physics_and_data, model_info, domain="default"):
        """
        Calculate the Jacobian matrix for the given physics and data.

        Parameters:
            physics_and_data (object): The physics and data object.
            model_info (object): The model information object.
            domain (str): The domain of the Jacobian. The domain can be either "default" or "inversion".
                The default domain is the default domain of the model. The inversion domain is the domain
                of the transformed model.

        Returns:
            numpy.ndarray: The Jacobian matrix. The matrix is already weighted and transformed.
        """
        assert domain in ["default", "inversion"] , "domain must be either 'default' or 'inversion'"
        mesh_size = model_info.mesh_info.mesh.cellCount()
        transformation_vec = np.ones(mesh_size)
        if not self.model_transformation_regularisation is None:
            # transformation_vec = self.model_transformation_regularisation.derivative_backward(model_info.model)
            transformation_vec = self.model_transformation_regularisation.derivative_forward(model_info.model)

        region_of_interest_ids = np.where(model_info.mesh_info.region_of_interest)[0]
        non_region_of_interest_ids = np.where(~model_info.mesh_info.region_of_interest)[0]

        if self._static_jacobian is True:
            #* Calculate jacobian once or return saved jacobian
            if self._static_jacobian_saved is None:
                jacobian = self._calculate_jacobian(physics_and_data, model_info)
                jacobian = sp.sparse.coo_matrix(jacobian)
                jacobian = remove_entries_below_thresh_from_coo(jacobian, 1e-10)
                jacobian = remove_columns_from_csr(jacobian.tocsr(), non_region_of_interest_ids)
                self._static_jacobian_saved = jacobian.tocsr()
            jacobian_default = self.weight * self._static_jacobian_saved.multiply(transformation_vec[region_of_interest_ids])
        else: 
            #* Recalculate jacobian
            jacobian = self._calculate_jacobian(physics_and_data, model_info)
            jacobian = sp.sparse.coo_matrix(jacobian)
            jacobian = remove_entries_below_thresh_from_coo(jacobian, 1e-10)
            jacobian = remove_columns_from_csr(jacobian.tocsr(), region_of_interest_ids)
            jacobian = jacobian.tocsr()

            jacobian_default = self.weight * jacobian.multiply(transformation_vec[region_of_interest_ids])

        if domain == "default":
            return jacobian_default
        else:
            return jacobian_default.multiply(model_info.transformed_model_gradient)

    def get_rhs_jumping(self, physics_and_data, model_info, domain="default"):
        """ 
        This function returns the right hand side for the jumping scheme.
        
        Parameters:
        - physics_and_data: The physics and data.
        - model_info: The model information.
        - domain: The domain of the right hand side. The domain can be either "default" or "inversion".
        
        Returns:
        - The right hand side for the jumping scheme. The right hand side is already weighted.
        """
        assert domain in ["default", "inversion"] , "domain must be either 'default' or 'inversion'"

        rhs_creeping = self.get_rhs_creeping(physics_and_data, model_info)

        region_of_interest_ids = np.where(model_info.mesh_info.region_of_interest)[0]    
        non_region_of_interest_ids = ~region_of_interest_ids

        if domain == "default":
            jacobian_used = self.get_jacobian(
                physics_and_data=physics_and_data,
                model_info=model_info,
                domain="default",
                )
            model_used = model_info.model[region_of_interest_ids]
        else:
            jacobian_used = self.get_jacobian(
                physics_and_data=physics_and_data,
                model_info=model_info,
                domain="inversion",
                )
            model_used = model_info.transformed_model[region_of_interest_ids]
        rhs_jumping = rhs_creeping + jacobian_used @ model_used
        return rhs_jumping

    def get_rhs_creeping(self, physics_and_data, model_info):
        """
        Calculate the right-hand side (RHS) for the creeping scheme.

        Parameters:
        - physics_and_data: The physics and data used for the calculation.
        - model_info: Information about the model.

        Returns:
        - rhs_creeping: The right hand side for the creeping scheme. The right hand side
        is already weighted.
        """
        phi = self._calculate_phi(
            physics_and_data=physics_and_data,
            model_info=model_info,
            model_transformation_regularisation=self.model_transformation_regularisation,
        )
        return (0-phi) * self.weight

    def plot_jacobian_spy(self, physics_and_data, model_info, markersize=2):
        """
        Plots the Jacobian matrix using the `spy` function from `matplotlib.pyplot`.

        Parameters:
        - physics_and_data: The physics and data used to calculate the Jacobian matrix.
        - model_info: Information about the model.
        - markersize: The size of the markers in the plot (default is 2).

        Returns:
        - fig: The figure object of the plot.
        - ax: The axes object of the plot.
        """
        jacobian = self.get_jacobian(physics_and_data, model_info)
        fig, ax = plt.subplots(1, 1, layout="constrained")
        plt.spy(jacobian, markersize=markersize)
        plt.show()
        return fig, ax

    def get_phi(self, physics_and_data, model_info, weighted=False):
        """
        Calculate the regularisation term for the given physics and data.

        Parameters:
        - physics_and_data: The physics and data object.
        - model_info: The model information object.
        - weighted: A flag to indicate if the regularisation term should be weighted.

        Returns:
        - phi: The regularisation term.
        """
        phi = self._calculate_phi(
            physics_and_data=physics_and_data,
            model_info=model_info,
            model_transformation_regularisation=self.model_transformation_regularisation,
        )
        if weighted:
            return self.weight * phi
        else:
            return phi
    
class RegularisationMultiModel:
    """
    This class represents a regularisation term of the form Phi(m)=0.
    The regularisation term is discretised as Phi(m) = weight * sum_{i=1}^{N} (phi_i(m))^2=0
    where phi_i is the i-th regularisation functional and N is the number of regularisation
    functionals.

    The Gauss-Newton formulation is based on the 1st order Taylor expansion of the
    regularisation functional.

    Comparing to Regularisation, this class supports/needs multiple models.

    Attributes:
    - weight: The weight of the regularisation term.
    - model_transformation_regularisation_list: The list of transformations of the model that is applied before the application
        of the regularisation. Setting the transformation to x^2 mean applying the regularisation
        to the square of the model.
    - _calculate_jacobian: The function to calculate the Jacobian.
    - _calculate_phi: The function to calculate the regularisation term.
    - _static_jacobian: A flag to indicate if the Jacobian is static.
    - _static_jacobian_saved: The saved static Jacobian.

    The function _calculate_jacobian must have the following signature:
    - _calculate_jacobian(physics_and_data, model_info_list)

    The function _calculate_phi must have the following signature:
    - _calculate_phi(physics_and_data, model_info_list, model_transformation_regularisation_list)
    """
    #*calculate_jacobian / calculate residual has signatur (physics_and_data , model_info_list)
    def __init__(
            self,
            calculate_jacobian,
            calculate_phi,
            static_jacobian,
            weight=1.0,
            model_transformation_regularisation_list=None,
            ):
        # Initialise functions to calculate jacobian and rhs
        self._calculate_jacobian  = calculate_jacobian
        self._calculate_phi = calculate_phi

        # Initialise flags for static jacobian or rhs
        self._static_jacobian = static_jacobian
        self._static_jacobian_saved = None

        # Intialise weight and transformation
        self.weight = weight
        self.model_transformation_regularisation_list = model_transformation_regularisation_list

        if model_transformation_regularisation_list is not None:
            assert isinstance(model_transformation_regularisation_list, list), "model_transformation_regularisation_list must be a list."

    def get_jacobian(self, physics_and_data, model_info_list, domain="default"):
        """
        Calculate the Jacobian matrix for the given physics and data.

        Parameters:
            physics_and_data (object): The physics and data object.
            model_info_list (list): The list of model information objects.
            domain (str): The domain of the Jacobian. The domain can be either "default" or "inversion".
                The default domain is the default domain of the model. The inversion domain is the domain
                of the transformed model.

        Returns:
            numpy.ndarray: The Jacobian matrix. The matrix is already weighted and transformed.
        """
        assert domain in ["default", "inversion"] , "domain must be either 'default' or 'inversion'"
        assert len(model_info_list) > 1, "The regularisation term requires at least two models."
        if not self.model_transformation_regularisation_list is None:
            assert len(model_info_list) == len(self.model_transformation_regularisation_list), "The number of transformations must be equal to the number of models."

        mesh_size = model_info_list[0].mesh_info.mesh.cellCount()
        no_of_models = len(model_info_list)

        if self.model_transformation_regularisation_list is None:
            transformation_vec = np.ones(mesh_size*no_of_models)
        else:
            transformation_vec = [transformation.derivative_backward(model_info.model) for (transformation, model_info) in zip(self.model_transformation_regularisation_list, model_info_list)]
            transformation_vec = np.array(transformation_vec).flatten()

        region_of_interest_ids = np.where(model_info_list[0].mesh_info.region_of_interest)[0]
        non_region_of_interest_ids = np.where(~model_info_list[0].mesh_info.region_of_interest)[0]

        region_of_interest_ids_repeated = np.tile(region_of_interest_ids, no_of_models) + np.repeat(np.arange(no_of_models), len(region_of_interest_ids))*mesh_size
        non_region_of_interest_ids_repeated = np.tile(non_region_of_interest_ids, no_of_models) + np.repeat(np.arange(no_of_models), len(non_region_of_interest_ids))*mesh_size

        transformation_vec = transformation_vec[region_of_interest_ids_repeated]

        if self._static_jacobian is True:
            #* Calculate jacobian once or return saved jacobian
            if self._static_jacobian_saved is None:
                jacobian = self._calculate_jacobian(physics_and_data, model_info_list)
                jacobian = sp.sparse.coo_matrix(jacobian)
                jacobian = remove_entries_below_thresh_from_coo(jacobian, 1e-10)
                jacobian = remove_columns_from_csr(jacobian.tocsr(), non_region_of_interest_ids_repeated)
                self._static_jacobian_saved = jacobian.tocsr()
            jacobian = self.weight * self._static_jacobian_saved.multiply(transformation_vec)
        else:
            #* Recalculate jacobian
            jacobian = self._calculate_jacobian(physics_and_data, model_info_list)
            jacobian = sp.sparse.coo_matrix(jacobian)
            jacobian = remove_entries_below_thresh_from_coo(jacobian, 1e-10)
            jacobian = remove_columns_from_csr(jacobian.tocsr(), non_region_of_interest_ids_repeated)
            jacobian = jacobian.tocsr()
            jacobian = self.weight * jacobian.multiply(transformation_vec)
        if domain == "default":
            return jacobian
        else:
            model_inversion_transformation = [model_info.transformed_model_gradient[region_of_interest_ids] for model_info in model_info_list]
            model_inversion_transformation = np.array(model_inversion_transformation).flatten()
            return jacobian.multiply(model_inversion_transformation)

    def get_rhs_jumping(self, physics_and_data, model_info_list, domain="default"):
        """ 
        This function returns the right hand side for the jumping scheme.
        
        Parameters:
        - physics_and_data: The physics and data.
        - model_info_list: The list of model information.
        - domain: The domain of the right hand side. The domain can be either "default" or "inversion".
        
        Returns:
        - The right hand side for the jumping scheme. The right hand side is already weighted.
        """
        assert domain in ["default", "inversion"] , "domain must be either 'default' or 'inversion'"

        rhs_creeping = self.get_rhs_creeping(physics_and_data, model_info_list)

        mesh_size = model_info_list[0].mesh_info.mesh.cellCount()
        no_of_models = len(model_info_list)

        region_of_interest_ids = np.where(model_info_list[0].mesh_info.region_of_interest)[0]
        non_region_of_interest_ids = ~region_of_interest_ids

        if domain == "default":
            jacobian = self.get_jacobian(
                physics_and_data=physics_and_data,
                model_info_list=model_info_list,
                domain="default",
                )
            model = np.array([model_info.model[region_of_interest_ids] for model_info in model_info_list]).flatten()
            return rhs_creeping + jacobian @ model
        else:
            jacobian = self.get_jacobian(
                physics_and_data=physics_and_data,
                model_info_list=model_info_list,
                domain="inversion",
                )
            transformed_model = np.array([model_info.transformed_model[region_of_interest_ids] for model_info in model_info_list]).flatten()
            return rhs_creeping + (jacobian @ transformed_model)

    def get_rhs_creeping(self, physics_and_data, model_info_list):
        """
        Calculate the right-hand side (RHS) for the creeping scheme.

        Parameters:
        - physics_and_data: The physics and data used for the calculation.
        - model_info_list: Information about the model.

        Returns:
        - rhs_creeping: The right hand side for the creeping scheme. The right hand side
        is already weighted.
        """
        phi = self._calculate_phi(
            physics_and_data=physics_and_data,
            model_info_list=model_info_list,
            model_transformation_regularisation_list=self.model_transformation_regularisation_list,
        )
        return (0-phi) * self.weight

    def plot_jacobian_spy(self, physics_and_data, model_info_list, markersize=2):
        """
        Plots the Jacobian matrix using the `spy` function from `matplotlib.pyplot`.

        Parameters:
        - physics_and_data: The physics and data used to calculate the Jacobian matrix.
        - model_info_list: Information about the model.
        - markersize: The size of the markers in the plot (default is 2).

        Returns:
        - fig: The figure object of the plot.
        - ax: The axes object of the plot.
        """
        jacobian = self.get_jacobian(physics_and_data, model_info_list)
        fig, ax = plt.subplots(1, 1, layout="constrained")
        plt.spy(jacobian, markersize=markersize)
        plt.show()
        return fig, ax
    
    def get_phi(self, physics_and_data, model_info_list, weighted=False):
        """
        Calculate the regularisation term for the given physics and data.

        Parameters:
        - physics_and_data: The physics and data object.
        - model_info_list: The list of model information objects.
        - weighted: A flag to indicate if the regularisation term should be weighted.

        Returns:
        - phi: The regularisation term.
        """
        phi = self._calculate_phi(
            physics_and_data=physics_and_data,
            model_info_list=model_info_list,
            model_transformation_regularisation_list=self.model_transformation_regularisation_list,
        )
        if weighted:
            return self.weight * phi
        else:
            return phi
