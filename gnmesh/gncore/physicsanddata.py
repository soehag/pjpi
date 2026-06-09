"""Physics and data utilities for geophysical and petrophysical inversion.

This module provides helper classes that wrap method managers and data
containers to compute forward responses and Jacobians for geophysical and
petrophysical inversions. It exposes utilities to filter sparse matrices and
to obtain weighted Jacobians and responses suitable for stacking multiple
methods.

Classes
-------
pyhsics_and_data_geophysical
    Wrapper for a single geophysical method providing jacobian/response
physics_and_data_petrophysical
    Combines multiple geophysical methods for petrophysical inversion
"""

import logging
import numpy as np
import scipy as sP
import pygimli as pg

logger = logging.getLogger(__name__)

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
    max_entry = np.max(np.abs(coo_matrix_new.data))
    mask = np.abs(coo_matrix_new.data) > relativ_threshold * max_entry
    logger.debug("Number of non-zero values in Jacobian: %s - %s", np.sum(mask), coo_matrix_new.size)
    coo_matrix_new.data = coo_matrix_new.data[mask]
    coo_matrix_new.row = coo_matrix_new.row[mask]
    coo_matrix_new.col = coo_matrix_new.col[mask]
    return coo_matrix_new

class pyhsics_and_data_geophysical:
    def __init__(self, manager, data_container, data_observed_field_name, weight = 1.0):        
        self._method_manager = manager
        assert isinstance(data_container, pg.DataContainer), "Data container must be a pygimli DataContainer."
        assert isinstance(data_observed_field_name, str), "Data observed must be a numpy array."
        self._data_container = data_container
        self._data_observed = np.array(data_container[data_observed_field_name])
        self._weight = weight
        if data_container.haveData("err"):
            err = np.array(data_container["err"])
        else:
            err = np.ones_like(self._data_observed) * 1e-6
        err = np.clip(err, a_min=1e-6, a_max=None)
        self._err = err
        self._data_transformation = None

    @property
    def data_observed(self):
        return self._data_observed
    
    @property
    def err(self):
        return self._err

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        assert isinstance(value, (int, float, np.number)), "Weight must be a number."
        self._weight = float(value)

    @property
    def data_weight(self):
        return self._weight

    @data_weight.setter
    def data_weight(self, value):
        self.weight = value
    
    @property
    def method_manager(self):
        return self._method_manager

    @property
    def data_transformation(self):
        return self._data_transformation

    @data_transformation.setter
    def data_transformation(self, value):
        assert value in [None, "log"]
        self._data_transformation = value

    def get_jacobian_and_response(self, model, domain="default"):
        """ Returns the Jacobian of the geophysical data."""
        assert domain in ["default", "inversion"], "Domain must be either 'default' or 'inversion'."
        # Initiate manager
        # Initiate mesh by removing all cell markers - otherwise pg does some weird fwd simulations
        mesh_for_manager = model.mesh_info.mesh.copy()
        for cell in mesh_for_manager.cells():
            cell.setMarker(0)

        # self._method_manager.fop.setMesh(mesh_for_manager)
        self._method_manager.setMesh(mesh_for_manager)
        self._method_manager.setData(self._data_container)

        response = self._method_manager.fop.response(model.model)
        self._method_manager.fop.createJacobian(model.model)
        jacobian = self._method_manager.fop.jacobian()
        if isinstance(jacobian, pg.SparseMapMatrix):
            # Convert pygimli sparse map matrix to CSR
            jacobian_dense = pg.utils.sparseMatrix2Dense(jacobian)
            jacobian_full = sP.sparse.csr_matrix(jacobian_dense)
        else:
            jacobian_full=pg.utils.toCSR(jacobian)
        # Slice out non-ROI columns
        jacobian_full = jacobian_full.tocsc()
        jacobian_full = jacobian_full[:, model.mesh_info.region_of_interest]
        jacobian_full = jacobian_full.tocsr()
        # Note: an earlier implementation created an explicit COO matrix and
        # filtered small entries. That approach was removed in favour of
        # slicing the CSR/CSC matrix directly. See VCS history if needed.

        #* Apply data transformation if set
        if self.data_transformation == "log":
            response = np.log(response)
            jacobian_full = sP.sparse.diags(1.0 / self._data_observed) @ jacobian_full

        weighted_response = self.weight * response
        if domain == "default":
            weighted_jacobian = self.weight * jacobian_full
        else:
            weighted_jacobian = self.weight * jacobian_full.multiply(model.transformed_model_gradient)
        return weighted_jacobian, weighted_response

    def get_err(self):
        """ Returns the error of the geophysical data."""
        return self._err

class physics_and_data_petrophysical:
    """Physics-and-data wrapper for petrophysical inversion.

    This class aggregates multiple geophysical method managers and their
    associated data containers. It provides helpers to compute forward
    responses and Jacobians of each method with respect to a shared
    petrophysical model.
    """
    def __init__(
            self,
            manager_and_transformation_list,
            data_container_list,
            data_observed_field_name_list
            ):
        """Initialises the physics and data object for petrophysical data."""
        number_of_methods = len(manager_and_transformation_list)
        assert number_of_methods == len(data_container_list),\
        "Number of methods and data containers must match."
        assert number_of_methods == len(data_observed_field_name_list),\
            "Number of methods and data observed fields must match."

        self._method_manager_list = [
            manager_and_transformation[0]
            for manager_and_transformation in manager_and_transformation_list
        ]
        # This is for PG regularisation formalities
        self.method_manager = self._method_manager_list[0]
        self._petro_transformation_list = [
            manager_and_transformation[1]
            for manager_and_transformation in manager_and_transformation_list
        ]
        self._data_container_list = data_container_list
        self._data_observed_list = [
            np.array(data_container[data_observed_field_name])
            for data_container, data_observed_field_name in zip(
                data_container_list,
                data_observed_field_name_list
            )
        ]
        
        err_list = []
        for data_container in data_container_list:
            if data_container.haveData("err"):
                err = np.array(data_container["err"])
            else:
                err = np.ones_like(self._data_observed) * 1e-6
            err = np.clip(err, a_min=1e-6, a_max=None)
            err_list.append(err)
        self._err_list = err_list

    def get_data_observed(self, number_of_method):
        """ Returns the observed data for the petrophysical data."""
        return self._data_observed_list[number_of_method]
    
    def get_err(self, number_of_method):
        """ Returns the error for the petrophysical data."""
        return self._err_list[number_of_method]

    def get_method_manager(self, number_of_method):
        """ Returns the method manager for the petrophysical data."""
        return self._method_manager_list[number_of_method]

    def get_geophysical_model(self, petrophysical_model, number_of_method):
        """ Returns the geophysical model for the petrophysical data."""
        petrophysical_transformation = self._petro_transformation_list[number_of_method]
        geophysical_model = petrophysical_transformation.fwd(petrophysical_model.model)
        return geophysical_model

    def get_jacobian_and_response_for_single_method(
            self,
            petrophysical_model,
            number_of_method,
            weight=1.0,
            domain="default",
            ):
        """ Returns the jacobian for the petrophysical data. This corresponds to the stacked 
        Jacobian of all methods with respect to the petrophysical model."""
        assert domain in ["default", "petro", "inversion"],\
            "Domain must be either 'default', 'petro' or 'inversion'."
        # Initiate manager
        mesh_for_manager = petrophysical_model.mesh_info.mesh.copy()
        for cell in mesh_for_manager.cells():
            cell.setMarker(0)

        method_manager = self._method_manager_list[number_of_method]
        method_manager.setMesh(mesh_for_manager)
        method_manager.setData(self._data_container_list[number_of_method])

        # Get response and geophysical Jacobian
        geophysical_model = self.get_geophysical_model(
            petrophysical_model=petrophysical_model,
            number_of_method=number_of_method
            )
        response = method_manager.fop.response(geophysical_model)
        method_manager.fop.createJacobian(geophysical_model)
        jacobian = method_manager.fop.jacobian()
        if isinstance(jacobian, pg.SparseMapMatrix):
            # jacobian_dense = pg.utils.sparseMatrix2Dense(jacobian)
            # jacobian_full = pg.utils.sparseMatrix2csr(jacobian_dense)
            jacobian_full = pg.utils.sparseMatrix2csr(jacobian)
        else:
            jacobian_full = pg.utils.toCSR(jacobian)

        # Create indices for ROI columns
        region_of_interest_ids = np.where(petrophysical_model.mesh_info.region_of_interest)[0]
        # Slice out non-ROI columns
        jacobian_full = jacobian_full.tocsc()
        jacobian_full = jacobian_full[:, petrophysical_model.mesh_info.region_of_interest]
        jacobian_full = jacobian_full.tocsr()

        # Note: explicit COO construction and thresholding was removed in
        # favour of operating directly on the sparse matrix slices. See
        # VCS history for previous approach if needed.

        # Get the gradient for the petrophysical transformation
        petro_transformation = self._petro_transformation_list[number_of_method]
        petro_transformation_gradient = petro_transformation.derivative_forward(
            petrophysical_model.model[region_of_interest_ids]
            )
        model_transformation_gradient = petrophysical_model.transformed_model_gradient[region_of_interest_ids]

        # Finalise the Jacobian and response
        if domain == "default":
            jacobian = jacobian_full
        elif domain == "petro":
            jacobian = jacobian_full.multiply(petro_transformation_gradient)
        elif domain == "inversion":
            jacobian = jacobian_full.multiply(petro_transformation_gradient)
            jacobian = jacobian.multiply(model_transformation_gradient)
        else:
            raise ValueError("Domain must be either 'default' or 'inversion'.")
        return weight*jacobian, weight*response

    def get_jacobian_and_response(self, petrophysical_model, weight_list=None, domain="default"):
        """ Returns the jacobian for the petrophysical data. This corresponds to the stacke
        Jacobian of all methods with respect to the petrophysical model."""

        if weight_list is None:
            weight_list = [1.0] * len(self._method_manager_list)
        assert isinstance(weight_list, list), "Weight list must be a list."
        assert len(weight_list) == len(self._method_manager_list),\
            "Weight list must have the same length as the number of methods."
        weighted_jacobian_list = []
        weighted_response_list = []
        for i in range(len(self._method_manager_list)):
            weight=weight_list[i]
            jacobian, response = self.get_jacobian_and_response_for_single_method(
                petrophysical_model=petrophysical_model,
                number_of_method=i,
                domain=domain,
                )
            weighted_jacobian_list.append(weight*jacobian)
            weighted_response_list.append(weight*response)

        weighted_jacobian_full = sP.sparse.vstack(weighted_jacobian_list)
        weighted_response_full = np.concatenate(weighted_response_list)
        return weighted_jacobian_full, weighted_response_full
