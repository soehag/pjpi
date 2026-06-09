"""
Geophysical Gauss-Newton manager and helpers.

This module contains `GaussNewtonGeophysical`, a specialization of
`GaussNewtonCore` that configures and runs geophysical inversions using
one or multiple data sets. It handles data/regularisation setup and the
main inversion loop.
"""

import logging
import numpy as np
import scipy as sP
import matplotlib.pyplot as plt
import time
from .gaussnewtoncore import GaussNewtonCore

logger = logging.getLogger(__name__)

class GaussNewtonGeophysical(GaussNewtonCore):
    """Initialises the Gauss-Newton manager for geophysical inversion."""

    _maximum_update_per_step_mode = "per_model"
    _allowed_num_solvers = ("scipy_sparse", "scipy_dense", "cupy_sparse", "cupy_dense")

    def __init__(
            self,
            mesh_info,
            geophysical_data,
            initial_models=None,
            single_model_regularisation=[],
            dual_model_regularisation=[],
			decouple_regularisation=None,
            maximum_iterations=100,
            save_model_history=True,
            scheme="creeping",
            verbose=True
            ):
        """ Initialises the Gauss-Newton manager for geophysical inversion."""
        # Set the mesh
        self._mesh_info = mesh_info

        # Set the region of interest
        mesh = self.mesh_info.mesh
        background_marker = np.min(mesh.cellMarkers())
        if not np.all(mesh.cellMarkers() == background_marker):
            logger.warning("Some cells are not part of the inversion region.")
        self._region_of_interest = mesh_info.region_of_interest

        # Check if geophysical_data is a list or tuple
        if isinstance(geophysical_data, (list, tuple)):
            self._data = geophysical_data
        else:
            self._data = [geophysical_data]

        self._number_of_data_sets = len(self._data)
        self.verbose = verbose

        if self.verbose:
            logger.info("No. of data sets: %s", self._number_of_data_sets)
            if self._number_of_data_sets == 0:
                logger.info("No data provided. Performing model fusion.")

        #* Check if single_model_regularisation is a list or tuple of lists
        if single_model_regularisation == []:
            self._single_model_regularisation = []
            logger.info("No single model regularisation provided.")
        elif isinstance(single_model_regularisation, (list, tuple)):
            if np.all([isinstance(reg, (list, tuple)) for reg in single_model_regularisation]):
                self._single_model_regularisation = single_model_regularisation
                logger.info("Single model regularisation provided")
            elif len(self._data) == 1:
                self._single_model_regularisation = [single_model_regularisation]
            else:
                logger.error("Please provide a list of lists for single model regularisation.")
                raise ValueError
        else:
            logger.error("Please provide a list of lists for single model regularisation.")
            raise ValueError
        
        # Check if dual_model_regularisation is a list
        if dual_model_regularisation == []:
            self._dual_model_regularisation = []
            logger.info("No dual model regularisation provided.")
        elif isinstance(dual_model_regularisation, (list, tuple)):
            self._dual_model_regularisation = dual_model_regularisation
            logger.info("Dual model regularisation provided.")
        else:
            logger.error("Please provide a list of dual model regularisation.")
            raise ValueError
        
        number_of_single_model_regularisation = 0
        for i, reg in enumerate(self._single_model_regularisation):
            number_of_single_model_regularisation += len(reg)
            logger.info("Number of single model regularisation for data set %s: %s", i+1, len(reg))
        self._number_of_single_model_regularisation = number_of_single_model_regularisation

        number_of_dual_model_regularisation = len(self._dual_model_regularisation)
        logger.info("Number of dual model regularisation: %s", number_of_dual_model_regularisation)
        self._number_of_dual_model_regularisation = number_of_dual_model_regularisation

        # Set maximum number of iterations
        self._maximum_iterations = maximum_iterations
        self._current_iteration = 0
        if self.verbose:
            logger.info("Maximum iterations: %s", self._maximum_iterations)

        #* Initialis tracking dictionary
        self._tracking_dict = {
            "general": {},
        }
        # Set initial models or generate them
        if initial_models is not None:
            if isinstance(initial_models, (list, tuple)):
                assert len(initial_models) == len(self._data) or len(self._data)==0, "Number of initial models must match number of data sets."
                initial_models = [mod.copy() for mod in initial_models]
            else:
                initial_models = [initial_models.copy()]
            logger.info("No. of initial models: %s", len(initial_models))
        else:
            initial_models = None
            logger.info("No initial models provided. Initial models will be initialised by the manager.")
            # TODO: Implement initialisation of models by the manager
            # TODO: Implement the creation of model info instances for each model

        self._tracking_dict["general"]["initial_models"] = initial_models
        self._current_models = [mod.copy() for mod in self.initial_models]

        self._number_of_models = len(self._current_models)
        logger.info("GaussNewtonGeophysical initialized.")

        # Set numerical scheme
        self.scheme = scheme

        # Set scaling
        self.scaling = "column_sum_l1"

        # Intialise misfit history
        self.save_model_history = save_model_history

        # Initialise model update trigger
        self._model_update_bool = [True for i in range(self._number_of_models)]

        # Set maximum update per step
        self._maximum_update_per_step = [(-np.inf, np.inf)]*self._number_of_models

        # Initialise the numerical solver by default
        self.num_solver = "cupy_sparse"

        self._model_history = []
        self._data_misfit_history = []
        self._single_model_regularisation_misfit_history = []
        self._dual_model_regularisation_misfit_history = []

        #* Set up decoupling
        if decouple_regularisation is not None:
            assert isinstance(decouple_regularisation, (list, tuple)), "Decouple regularisation must be a list or tuple."
            assert len(decouple_regularisation) == 2, "Decouple regularisation must have two elements."
            assert isinstance(decouple_regularisation[0], np.ndarray), "Decouple regularisation must be a numpy array."
            assert decouple_regularisation[0].shape[0] == self._mesh_info.mesh.cellCount(), "Decouple regularisation must have the same length as the number of cells."
            assert isinstance(decouple_regularisation[1], list), "Decouple regularisation must be a numpy array."
            assert all(isinstance(reg, (list, tuple)) for reg in decouple_regularisation[1]), "Decouple regularisation must be a list of lists."
            if self.verbose:
                logger.info("Decoupling regularisation provided.")
        else:
            if self.verbose:
                logger.info("No decoupling regularisation provided.")
        self.decouple_regularisation = decouple_regularisation
        if self.verbose: logger.info("Misfit history initialised.")

        #* Set termination criterion
        self._terminate_on_chi2_decrease = 0.0

    @property
    def single_model_regularisation(self):
        """Returns the single-model regularisation configuration."""
        return self._single_model_regularisation

    @single_model_regularisation.setter
    def single_model_regularisation(self, value):
        """Sets the single-model regularisation configuration."""
        assert isinstance(value, (list, tuple)), "Single model regularisation must be a list or tuple."
        self._single_model_regularisation = value

    @property
    def dual_model_regularisation(self):
        """Returns the dual-model regularisation configuration."""
        return self._dual_model_regularisation

    @dual_model_regularisation.setter
    def dual_model_regularisation(self, value):
        """Sets the dual-model regularisation configuration."""
        assert isinstance(value, (list, tuple)), "Dual model regularisation must be a list or tuple."
        self._dual_model_regularisation = value

    @property
    def number_of_data_sets(self):
        """Returns the number of data sets."""
        return self._number_of_data_sets

    @property
    def number_of_models(self):
        """Returns the number of models."""
        return self._number_of_models

    @property
    def number_of_single_model_regularisation(self):
        """Returns the number of single-model regularisation terms."""
        return self._number_of_single_model_regularisation

    @property
    def number_of_dual_model_regularisation(self):
        """Returns the number of dual-model regularisation terms."""
        return self._number_of_dual_model_regularisation

    @property
    def single_model_regularisation_misfit_history(self):
        """Returns the single-model regularisation misfit history."""
        return self._single_model_regularisation_misfit_history

    @property
    def dual_model_regularisation_misfit_history(self):
        """Returns the dual-model regularisation misfit history."""
        return self._dual_model_regularisation_misfit_history

    @property
    def model_regularisation_misfit_history(self):
        """Returns the single-model regularisation misfit history."""
        return self._single_model_regularisation_misfit_history

    @property
    def current_models(self):
        """Returns the current models."""
        return self._current_models

    @property
    def initial_models(self):
        """Returns the initial models."""
        return self._tracking_dict["general"]["initial_models"]

    @property
    def model_update_bool(self):
        """Returns the model update mask."""
        return self._model_update_bool

    @property
    def model_regularisation(self):
        """Returns the single-model regularisation configuration."""
        return self._single_model_regularisation

    @model_regularisation.setter
    def model_regularisation(self, value):
        """Sets the single-model regularisation configuration."""
        self.single_model_regularisation = value

    def run(self):
        """ Runs the Gauss-Newton inversion."""
        logger.info("----------Running Gauss-Newton inversion.----------")
        if self._current_iteration == self._maximum_iterations:
            logger.info("----------Maximum number of iterations reached. Returning.----------")
            return
        
        if np.any(self._model_update_bool is False):
            logger.warning("Some models will not be updated.")

        while self._current_iteration < self._maximum_iterations:
            logger.info("----------Processing iteration %s----------", self._current_iteration+1)
            start_time_iteration = time.time()
            # Create iteration dictionary for tracking if necessary
            if self._current_iteration not in self._tracking_dict:
                self._tracking_dict[self._current_iteration] = {}

            #* Save current model if required
            if self._save_model_history:
                # Save the models
                self._tracking_dict[self._current_iteration]["models"] = [mod.copy() for mod in self._current_models]

            if self.verbose:
                logger.info("----------Start: Calculating geophysical jacobians and rhs.----------")
            start_time = time.time()

            #* Set up the geophysical inversion
            if len(self._data) > 0:
                jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list, chi_squared_list = self.get_final_geophysical_jacobian_and_rhs()
            else:
                jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list = None, None, None, None

            if self.verbose:
                logger.info("Time taken to calculate geophysical jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating geophysical jacobians and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model regularisation jacobians and rhs.----------")
            start_time = time.time()

            #* Set up the single model regularisation
            if self._number_of_single_model_regularisation > 0:
                single_model_regularisation_jacobian, single_model_regularisation_rhs, single_model_regularisation_misfit = self.get_single_model_regularisation_jacobian_and_rhs()
            else:
                single_model_regularisation_jacobian, single_model_regularisation_rhs, single_model_regularisation_misfit = None, None, None

            #* Set up the dual model regularisation
            if self._number_of_dual_model_regularisation > 0:
                dual_model_regularisation_jacobian, dual_model_regularisation_rhs, dual_model_regularisation_misfit = self.get_dual_model_regularisation_jacobian_and_rhs()
            else:
                dual_model_regularisation_jacobian, dual_model_regularisation_rhs, dual_model_regularisation_misfit = None, None, None

            assert self._number_of_data_sets>0 or self._number_of_single_model_regularisation>0 or self._number_of_dual_model_regularisation, "No data or regularisation provided."

            if self.verbose:
                logger.info("Time taken to calculate model regularisation jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating model regularisation jacobians and rhs.----------")

            #* Save misfit values -  these are technically from the iteration before
            self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
            self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list
            self._tracking_dict[self._current_iteration]["single_model_regularisation_misfit"] = single_model_regularisation_misfit
            self._tracking_dict[self._current_iteration]["dual_model_regularisation_misfit"] = dual_model_regularisation_misfit

            #* Check if inversion is finished (by chi2 decrease)
            if self._current_iteration > 0:
                curr_dict = self._tracking_dict.get(self._current_iteration, {})
                prev_dict = self._tracking_dict.get(self._current_iteration - 1, {})
                chi_curr = curr_dict.get("chi_squared")
                chi_prev = prev_dict.get("chi_squared")
                if chi_curr is not None and chi_prev is not None and len(chi_curr) > 0:
                    chi2_percentage_decrease_list = self.percent_decrease_in_chi2(iteration=self._current_iteration)
                    if self.verbose:
                        logger.info("Chi2 percentage decrease: %s", chi2_percentage_decrease_list)
                    if all([chi2_percentage_decrease < self.terminate_on_chi2_decrease for chi2_percentage_decrease in chi2_percentage_decrease_list]):
                        logger.info("Chi2 decrease criterion reached. Returning.")
                        break

            if self.verbose:
                logger.info("----------Start: Calculating full jacobian and rhs.----------")
            start_time = time.time()

            #* Setup the full Jacobian and response
            full_jacobian_list = []
            full_rhs_list = []

            if len(self._data) > 0:
                full_jacobian_list.append(jacobian_inversion_data)
                full_rhs_list.append(rhs_data)
            if self._number_of_single_model_regularisation > 0:
                full_jacobian_list.append(single_model_regularisation_jacobian)
                full_rhs_list.append(single_model_regularisation_rhs)
            if self._number_of_dual_model_regularisation > 0:
                full_jacobian_list.append(dual_model_regularisation_jacobian)
                full_rhs_list.append(dual_model_regularisation_rhs)
            
            full_jacobian = sP.sparse.vstack(blocks=full_jacobian_list, format="csr")
            full_rhs = np.concatenate(full_rhs_list, axis=0).flatten()

            if self.verbose:
                logger.info("Time taken to calculate full jacobian and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating full jacobian and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Solving linear system.----------")

            if self.verbose:
                logger.info("Sparsity of Jacobian: %.2e", full_jacobian.nnz/(np.prod(full_jacobian.shape)))

            #* Solve the linear system
            model_update_small = self.solve_linear_system(
                A=full_jacobian,
                b=full_rhs,
                enable_scaling=True,
            )

            if self.verbose:
                logger.info("----------End: Solving linear system.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model updates.----------")
            start_time = time.time()

            #* Update the models
            geophysical_model_update_list = self.get_model_updates_from_inversion_result(
                model_updates_vector=model_update_small,
                force_disable_clipping=False,
            )

            self.apply_model_updates(
                geophysical_model_update_list=geophysical_model_update_list,
            )

            if self.verbose:
                logger.info("Time taken to calculate model updates: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating model updates.----------")

            #* Save updates and sizes
            if self._save_model_history:
                self._tracking_dict[self._current_iteration]["model_updates"] = geophysical_model_update_list.copy()
                model_update_size_list = []
                for model_update in geophysical_model_update_list:
                    if model_update is not None:
                        model_update_size_list.append(np.linalg.norm(model_update))
                    else:
                        model_update_size_list.append(None)
                self._tracking_dict[self._current_iteration]["model_update_sizes"] = model_update_size_list.copy()
            self._current_iteration += 1

            if self.verbose:
                logger.info("Time taken for iteration %s: %.2f seconds.", self._current_iteration, time.time()-start_time_iteration)
                logger.info("----------End: Iteration.----------")

        if self.verbose:
            logger.info("----------Start: Finalising inversion.----------")
        if self._current_iteration not in self._tracking_dict:
            self._tracking_dict[self._current_iteration] = {}
        # Save the final models
        if self._save_model_history:
            # Save the models
            self._tracking_dict[self._current_iteration]["models"] = [mod.copy() for mod in self._current_models]
            self._model_history.append([mod.copy() for mod in self._current_models])

        # Calculate the final misfit
        if self._number_of_data_sets > 0:
            jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list, chi_squared_list = self.get_final_geophysical_jacobian_and_rhs()
            self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
            self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list

        # Calculate final single model regularisation misfit
        if self._number_of_single_model_regularisation > 0:
            single_model_regularisation_jacobian, single_model_regularisation_rhs, single_model_regularisation_misfit = self.get_single_model_regularisation_jacobian_and_rhs()
            self._tracking_dict[self._current_iteration]["single_model_regularisation_misfit"] = single_model_regularisation_misfit

        # Calculate final dual model regularisation misfit
        if self._number_of_dual_model_regularisation > 0:
            dual_model_regularisation_jacobian, dual_model_regularisation_rhs, dual_model_regularisation_misfit = self.get_dual_model_regularisation_jacobian_and_rhs()
            self._tracking_dict[self._current_iteration]["dual_model_regularisation_misfit"] = dual_model_regularisation_misfit

        #* Adjust maximum iterations
        if self._current_iteration == self._maximum_iterations:
            logger.info("Maximum number of iterations reached.")
        else:
            logger.info("Finished Gauss-Newton inversion after %s iterations. Overwriting maximum iterations.", self._current_iteration)
            self._maximum_iterations = self._current_iteration
        logger.info("----------End: Finalising inversion.----------")
        logger.info("----------Gauss-Newton inversion finished.----------")

# Jacobian and response functions
    def get_geophysical_jacobian_and_response(
            self,
            respect_model_update_bool=True,
            ):
        """ Returns the Jacobian of the geophysical data."""
        # Collect all the jacobians and rhs
        weighted_jacobian_inversion_list = []
        weighted_response_list = []
        for num, (pyhsics_and_data, model) in enumerate(zip(self._data, self._current_models)):
            if (respect_model_update_bool is True) and (self._model_update_bool[num] is False):
                continue
            weighted_jacobian_inversion, weighted_response = pyhsics_and_data.get_jacobian_and_response(model, domain="inversion")
            weighted_jacobian_inversion_list.append(weighted_jacobian_inversion)
            weighted_response_list.append(weighted_response)
        return weighted_jacobian_inversion_list, weighted_response_list

    def get_geophysical_jacobian_and_rhs(
            self,
            weighted_jacobian_inverse_list_data,
            weighted_response_list,
            respect_model_update_bool=True,
            ):
        """ Returns the right hand side of the geophysical data."""
        # Create the right hand side
        # Weighted response
        weight_list = [data.weight for data in self._data]
        # Remove models that are not to be updated
        if respect_model_update_bool:
            weight_list = [weight for weight, update_bool in zip(weight_list, self._model_update_bool) if update_bool]
        weighted_response_vector = np.concatenate(weighted_response_list, axis=0).flatten()

        # Weighted observed data
        observed_data_list = [data.data_observed for data in self._data]
        # Remove models that are not to be updated
        if respect_model_update_bool:
            observed_data_list = [data for data, update_bool in zip(observed_data_list, self._model_update_bool) if update_bool]

        weighted_observed_data_list = [weight * data for weight, data in zip(weight_list, observed_data_list)]
        weighted_observed_data_vector = np.concatenate(weighted_observed_data_list, axis=0).flatten()

        # Transformed model
        transformed_model_vector_list = [model.transformed_model for model in self._current_models]

        # Regions of interest
        region_of_interest_list = [model.mesh_info.region_of_interest for model in self._current_models]

        # Remove models that are not to be updated
        if respect_model_update_bool:
            transformed_model_vector_list = [model for model, update_bool in zip(transformed_model_vector_list, self._model_update_bool) if update_bool]
            region_of_interest_list = [region_of_interest for region_of_interest, update_bool in zip(region_of_interest_list, self._model_update_bool) if update_bool]
  
        # for num, single_model_vector in enumerate(transformed_model_vector_list):
        #     transformed_model_vector_list[num] = single_model_vector[region_of_interest_list[num]]
        #     # transformed_model_vector_list = [model[self.region_of_interest] for model in transformed_model_vector_list]
        transformed_model_vector = np.concatenate(transformed_model_vector_list, axis=0).flatten()

        weighted_residual = weighted_observed_data_vector - weighted_response_vector

        # data_misfit_list = [np.linalg.norm(weighted_residual) for weighted_residual in weighted_residual]
        data_misfit_list = [np.linalg.norm(w_obs - w_resp) for w_obs, w_resp in zip(weighted_observed_data_list, weighted_response_list)]
        data_misfit = np.linalg.norm(weighted_residual)

        #* Calculate chi squared list
        err_list = [data.err for data in self._data]
        chi_squared_list = [
            (w_obs.size)**-1 *np.sum(((w_obs - w_resp)/(err*w_obs))**2) for w_obs, w_resp, err, weight in zip(weighted_observed_data_list, weighted_response_list, err_list, weight_list)
        ]

        # Create the jacobian matrix
        # Remove the columns that are not to be updated
        # if respect_roi:
        #     for num, single_jacobian in enumerate(weighted_jacobian_inverse_list_data):
        #         single_jacobian = remove_columns_from_csr(single_jacobian, ~self.region_of_interest).copy()
        #         weighted_jacobian_inverse_list_data[num] = single_jacobian

        weighted_jacobian_inversion_data = sP.sparse.block_diag(
        mats=weighted_jacobian_inverse_list_data,
        format="csr"
        )

        if self.scheme == "creeping":
            rhs_data = weighted_residual
        elif self.scheme == "jumping":
            rhs_data = weighted_residual + weighted_jacobian_inversion_data @ transformed_model_vector
        else:
            raise ValueError("Invalid scheme provided.")
        return weighted_jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list, chi_squared_list

    def get_final_geophysical_jacobian_and_rhs(
            self,
            respect_model_update_bool=True,
            respect_roi=True
            ):
        """ Returns the final Jacobian of the geophysical data."""
        jacobian_inverse_list_data, response_list = self.get_geophysical_jacobian_and_response()
        jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list, chi_squared_list = self.get_geophysical_jacobian_and_rhs(
            weighted_jacobian_inverse_list_data=jacobian_inverse_list_data,
            weighted_response_list=response_list
            )
        return jacobian_inversion_data, rhs_data, data_misfit, data_misfit_list, chi_squared_list

    def get_single_model_regularisation_jacobian_and_rhs(self, respect_model_update_bool=True, respect_roi=True):
        """ Returns the Jacobian of the single model regularisation."""
        if self._single_model_regularisation == []:
            return None, None, None
        
        # Collect all the jacobians and rhs
        jacobian_list = []
        rhs_list = []
        single_model_regularisation_misfit = []
        data_weights = [data.weight for data in self._data]
        for (num, regularisation_list), weight in zip(enumerate(self._single_model_regularisation), data_weights):
            if (respect_model_update_bool is True) and (self._model_update_bool[num] is False):
                continue

            if self._number_of_data_sets == 0:
                physics_and_data = None
            else:
                physics_and_data = self._data[num]
            model = self._current_models[num]

            jacobian_list_temp = []
            rhs_list_temp = []
            single_model_regularisation_misfit_temp = []

            if not regularisation_list == []:
                for reg in regularisation_list:
                    jacobian = reg.get_jacobian(
                        physics_and_data=physics_and_data,
                        model_info=model,
                        domain="inversion") * weight
                    
                    # if respect_roi:
                    #     jacobian = remove_columns_from_csr(jacobian, ~self.region_of_interest)
                    
                    phi = reg.get_phi(
                        physics_and_data=physics_and_data,
                        model_info=model,
                        weighted=True,
                        )
                    
                    if self.scheme == "creeping":
                        rhs = reg.get_rhs_creeping(
                            physics_and_data=physics_and_data,
                            model_info=model
                            ) * weight
                    elif self.scheme == "jumping":
                        rhs = reg.get_rhs_jumping(
                            physics_and_data=physics_and_data,
                            model_info=model,
                            domain="inversion"
                            ) * weight
                    else:
                        raise ValueError("Invalid scheme provided.")
                    
                    jacobian_list_temp.append(jacobian)
                    rhs_list_temp.append(rhs)
                    single_model_regularisation_misfit_temp.append(phi)

                jacobian_list.append(jacobian_list_temp)
                rhs_list.append(rhs_list_temp)
                single_model_regularisation_misfit.append(single_model_regularisation_misfit_temp)

        # Concatenate the rhs
        rhs_concatenated_per_model = [np.concatenate(rhs_temp, axis=0).flatten() for rhs_temp in rhs_list]
        rhs = np.concatenate(rhs_concatenated_per_model, axis=0).flatten()

        single_model_regularisation_concatenated = [np.concatenate(misfit_temp, axis=0).flatten() for misfit_temp in single_model_regularisation_misfit]
        single_model_regularisation_misfit = np.concatenate(single_model_regularisation_concatenated, axis=0).flatten()
        single_model_misfit = np.linalg.norm(single_model_regularisation_misfit)

        # Concatenate the jacobians - vstack for single model - block_diag for multiple models
        list_of_vstacked_jacobians = []
        for num, jacobian_list_temp in enumerate(jacobian_list):
            list_of_vstacked_jacobians.append(sP.sparse.vstack(blocks=jacobian_list_temp, format="csr"))
        jacobian = sP.sparse.block_diag(
            mats=list_of_vstacked_jacobians,
            format="csr")
        
        #* Remove the rows that are coupled
        jacobian, rhs = self.remove_rows_coupling_trusted_untrusted(jacobian, rhs, self.mesh_info.region_of_interest)

        return jacobian, rhs, single_model_misfit

    def get_dual_model_regularisation_jacobian_and_rhs(self):
        """ Returns the Jacobian of the dual model regularisation."""
        if self._dual_model_regularisation == []:
            return None, None, None
        
        # Collect all the jacobians and rhs
        jacobian_list = []
        rhs_list = []
        dual_model_regularisation_misfit = []

        for reg in self._dual_model_regularisation:
            jacobian = reg.get_jacobian(
                physics_and_data=self._data,
                model_info_list=self._current_models,
                domain="inversion"
            )
            phi = reg.get_phi(
                physics_and_data = self._data,
                model_info_list=self._current_models,
                weighted=True,
            )
            if self.scheme == "creeping":
                rhs = reg.get_rhs_creeping(
                    physics_and_data=self._data,
                    model_info_list=self._current_models
                )
            elif self.scheme == "jumping":
                rhs = reg.get_rhs_jumping(
                    physics_and_data=self._data,
                    model_info_list=self._current_models,
                    domain="inversion"
                )
            else:
                raise ValueError("Invalid scheme provided.")

            jacobian_list.append(jacobian)
            rhs_list.append(rhs)
            dual_model_regularisation_misfit.append(phi)
        
        # Concatenate the rhs
        rhs = np.concatenate(rhs_list, axis=0).flatten()
        dual_model_regularisation_misfit = np.concatenate(dual_model_regularisation_misfit, axis=0).flatten()
        dual_model_misfit = np.linalg.norm(dual_model_regularisation_misfit)

        # Concatenate the jacobians
        jacobian = sP.sparse.vstack(blocks=jacobian_list, format="csr")
        # jacobian = np.vstack(jacobian_list)

        #* Remove the rows that are coupled
        jacobian, rhs = self.remove_rows_coupling_trusted_untrusted(jacobian, rhs, region_of_interest=self.mesh_info.region_of_interest)

        return jacobian, rhs, dual_model_misfit
 
    def get_model_updates_from_inversion_result(self, model_updates_vector, force_disable_clipping=False):
        model_update_list = []
        model_size = self._region_of_interest.size
        roi_size = np.sum(self._region_of_interest)
        counter = 0

        for i, model in enumerate(self._current_models):
            if self._model_update_bool[i]:
                if self.scheme == "creeping":
                    model_update_roi_inversion_domain = model_updates_vector[counter*roi_size:(counter+1)*roi_size]
                elif self.scheme == "jumping":
                    model_update_roi_inversion_domain = model_updates_vector[counter*roi_size:(counter+1)*roi_size] - model.transformed_model
                else:
                    raise ValueError("Invalid scheme provided.")

                #* Temporar update the model to get update in geophysical domain
                old_geophysical_model = model.model.copy()
                model.transformed_model = model.transformed_model + model_update_roi_inversion_domain
                new_geophysical_model = model.model.copy()
                geophysical_model_update = new_geophysical_model - old_geophysical_model
                model.model = old_geophysical_model

                if force_disable_clipping:
                    clipped_geophysical_model_update = geophysical_model_update
                else:
                    clipped_geophysical_model_update = self.clip_model_vector(
                        model_vector_update=geophysical_model_update,
                        model_no=i
                        )
                #*Add calculated update to the list
                model_update_list.append(clipped_geophysical_model_update.copy())
                counter +=1
            else:
                model_update_list.append(None)
        return model_update_list

    def apply_model_updates(self, geophysical_model_update_list):
        """ Applies the model updates to the current models."""
        for i, model in enumerate(self._current_models):
            if self._model_update_bool[i]:
                model.model = model.model + geophysical_model_update_list[i]
        return

    # Delegating implementations (solver/clipping/decoupling) are provided by GaussNewtonCore

# Plotting functions
    def plot_data_misfit_history(self, ax=None, figsize=(10, 6), ylim=(0,None), normalise=False):
        """ Plots the data misfit history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        no_models = len(self._current_models)
        iterations, data_misfit = self.assemble_iteration_vector_from_tracking_dict("data_misfit")
        if normalise:
            data_misfit = data_misfit/data_misfit[0]
        ax.plot(iterations, data_misfit, label="Data Misfit")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Data Misfit")
        ax.set_title("Data Misfit History")
        if ylim[1] is not None:
            ylim[1] = 1.1 * max(data_misfit)
        ax.set_ylim(ylim)
        ax.legend()
        return fig, ax
    
    def plot_single_model_regularisation_misfit_history(self, ax=None, figsize=(10, 6)):
        """ Plots the single model regularisation misfit history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        iterations, single_model_regularisation_misfit = self.assemble_iteration_vector_from_tracking_dict("single_model_regularisation_misfit")
        ax.plot(iterations, single_model_regularisation_misfit, label="Single Model Regularisation Misfit")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Single Model Regularisation Misfit")
        ax.set_title("Single Model Regularisation Misfit History")
        return fig, ax
    
    def plot_misfit_history(self, ax=None, figsize=(10,6), ylim=(0, None)):
        """ Plot all misfit histories."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        iterations_data, data_misfit = self.assemble_iteration_vector_from_tracking_dict("data_misfit")
        iterations_single_model_regularisation, single_model_regularisation_misfit = self.assemble_iteration_vector_from_tracking_dict("single_model_regularisation_misfit")
        iterations_dual_model_regularisation, dual_model_regularisation_misfit = self.assemble_iteration_vector_from_tracking_dict("dual_model_regularisation_misfit")

        if len(data_misfit) > 0:
            counter=0
            for no_dataset, model_updated in zip(range(self._number_of_data_sets), self._model_update_bool):
                if model_updated:
                    data_misfit_vector = [data_misfit_dict[counter] for data_misfit_dict in data_misfit]
                    ax.plot(iterations_data, data_misfit_vector, label=f"Data Misfit {no_dataset+1}")
                    counter += 1
        if len(single_model_regularisation_misfit) > 0:
            ax.plot(iterations_single_model_regularisation, single_model_regularisation_misfit, label="Single Model Regularisation Misfit")
        if len(dual_model_regularisation_misfit) > 0:
            ax.plot(iterations_dual_model_regularisation, dual_model_regularisation_misfit, label="Dual Model Regularisation Misfit")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Misfit")
        ax.set_title("Misfit History")
        if ylim[1] is not None:
            ylim[1] = 1.1 * max([np.max(data_misfit), np.max(single_model_regularisation_misfit), np.max(dual_model_regularisation_misfit)])
        ax.set_ylim(ylim)
        ax.legend()
        return fig, ax
    
    def plot_current_models(self, include_initial_models=False, figsize=None, **kwargs):
        """ Plot the current models."""
        current_modelinfo_list = [model for model in self._current_models]
        number_of_models = len(current_modelinfo_list)

        if include_initial_models:
            initial_modelinfo_list = [model for model in self.initial_models]
            figsize = (5 * number_of_models, 10) if figsize is None else figsize
        else:
            figsize = (5 * number_of_models, 5) if figsize is None else figsize

        if include_initial_models:
            fig, axs = plt.subplots(2, number_of_models, figsize=figsize)
        else:
            fig, axs = plt.subplots(1, number_of_models, figsize=figsize)

        # Plot the initial models
        if include_initial_models:
            ax_initial = axs[0]
            for i, model in enumerate(initial_modelinfo_list):
                if not number_of_models == 1:
                    ax = ax_initial[i]
                else:
                    ax = ax_initial
                model.plot_model(ax=ax, **kwargs)

        # Plot the current models
        for i, model in enumerate(current_modelinfo_list):
            axs_current = axs[1] if include_initial_models else axs
            if not number_of_models == 1:
                ax = axs_current[i]
            else:
                ax = axs_current
            model.plot_model(ax=ax, **kwargs)
        return fig, axs

    def show_region_of_interest(self, markersize=1, marker="o", mode="triang"):
        """ Shows the region of interest for all models."""
        fig, axs = plt.subplots(1, len(self._current_models), figsize=(5*len(self._current_models), 5))
        if len(self._current_models) == 1:
            axs = [axs]
        for i, model in enumerate(self._current_models):
            mesh_info = model.mesh_info
            mesh_info.show_region_of_interest(ax=axs[i], markersize=markersize, marker=marker, mode=mode)
        return fig, axs
    
    def show_regularisation_coverage(self, ax=None, cMap="turbo", cMin=0, cMax=None):
        """ Shows the coverage of the regularisation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        jacobian, _, _ = self.get_single_model_regularisation_jacobian_and_rhs()
        if jacobian is None:
            raise ValueError("No single-model regularisation Jacobian available to visualize.")
        jacobian_array = sP.sparse.csr_matrix(jacobian).toarray()

        jacobian_nnz = np.abs(jacobian_array) > 0
        jacobian_nnz = np.sum(jacobian_nnz, axis=0)

        if cMax is None:
            cMax = np.max(jacobian_nnz)

        #* Show the mesh
        import pygimli as pg
        _= pg.show(self._mesh_info.mesh, data=jacobian_nnz, ax=ax, cMap=cMap, cMin=cMin, cMax=cMax, showMesh=True)
        return fig, ax
