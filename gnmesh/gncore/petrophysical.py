"""
Petrophysical Gauss-Newton manager.

Contains `GaussNewtonPetrophysical` which runs petrophysical inversions
that couple petrophysical models with geophysical forward operators.
"""

import logging
import numpy as np
import scipy as sP
import matplotlib.pyplot as plt
import time
from .gaussnewtoncore import GaussNewtonCore

logger = logging.getLogger(__name__)

class GaussNewtonPetrophysical(GaussNewtonCore):
    """Class for Gauss-Newton inversion of petrophysical models."""

    _maximum_update_per_step_mode = "single"
    def __init__(
            self,
            mesh_info,
            petrophysical_data,
            initial_model=None,
            model_regularisation=None,
            decouple_regularisation=None,
            maximum_iterations=100,
            save_model_history=True,
            scheme="creeping",
            verbose=True,
            data_weight_list=None,
    ):
        if initial_model is None:
            initial_model = []
        if model_regularisation is None:
            model_regularisation = []

        # Set mesh info
        self._mesh_info = mesh_info

        # Set the region of interest
        self._region_of_interest = mesh_info.region_of_interest
        if not np.all(self._region_of_interest):
            logger.warning("Some cells are not part of the inversion region.")

        # Check if the petrophysical data is a list or tuple
        self._data = petrophysical_data
        self._number_of_datasets = len(petrophysical_data._data_container_list)
        self.verbose = verbose
        if self.verbose:
            logger.info("Number of datasets: %s", self._number_of_datasets)
            if self._number_of_datasets == 0:
                raise ValueError("No datasets provided.")

        # Check if the data weight is a list or tuple
        if data_weight_list is None:
            self._data_weight_list = [1.0] * self._number_of_datasets
            if self.verbose:
                logger.info("No data weights provided. Setting all to 1.0.")
        else:
            if isinstance(data_weight_list, (list, tuple)):
                assert len(data_weight_list) == self._number_of_datasets,\
                    "Number of data weights must match number of datasets."
                self._data_weight_list = data_weight_list
            elif isinstance(data_weight_list, (int, float)):
                self._data_weight_list = [data_weight_list]
            else:
                raise ValueError("Data weights must be a list, tuple, int or float.")

            if self.verbose:
                logger.info("Number of data weights: %s", len(self._data_weight_list))

        # Check if the model regularisation is a list or tuple
        if model_regularisation is None:
            self._model_regularisation = []
            if self.verbose:
                logger.info("No model regularisation provided.")
        else:
            if isinstance(model_regularisation, (list, tuple)):
                self._model_regularisation = model_regularisation
            else:
                self._model_regularisation = [model_regularisation]
            if self.verbose:
                logger.info("Number of model regularisations: %s", len(self._model_regularisation))
        self._number_of_single_model_regularisation = len(self._model_regularisation)

        # Set the maximum number of iterations
        self._maximum_iterations = maximum_iterations
        self._current_iteration = 0
        if self.verbose:
            logger.info("Maximum number of iterations: %s", self._maximum_iterations)

        # Initialise the tracking dictionary
        self._tracking_dict = {
            "general": {
            },
        }
        if initial_model is not None:
            self._tracking_dict["general"]["initial_model"] = initial_model.copy()
        else:
            self._tracking_dict["general"]["initial_model"] = None
        self._current_model = initial_model.copy()

        # Set numerical scheme
        self.scheme = scheme

        # Set scaling
        self.scaling = "column_sum_l1"

        # Set maximum update per step
        self._maximum_update_per_step = [-np.inf, np.inf]

        # Initialise the numerical solver by default
        self.num_solver = "cupy_sparse"

        # Initisalise the history of misfits
        self.save_model_history = save_model_history
        self._model_history = []
        self._data_misfit_history = []
        self._model_regularisation_misfit_history = []

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

        # Set the termination criterion
        self._terminate_on_chi2_decrease = 0.0

        if self.verbose:
            logger.info("Gauss-Newton inversion initialised.")

    @property
    def data_weight_list(self):
        """ Returns the data weights."""
        return self._data_weight_list

    @data_weight_list.setter
    def data_weight_list(self, value):
        """Sets the data weights."""
        assert isinstance(value, (list, tuple)), "Data weights must be a list or tuple."
        assert len(value) == self._number_of_datasets, "Number of data weights must match number of datasets."
        self._data_weight_list = list(value)

    @property
    def initial_model(self):
        """Returns the initial model."""
        return self._tracking_dict["general"]["initial_model"]

    @property
    def model_regularisation(self):
        """Returns the model regularisation configuration."""
        return self._model_regularisation

    @model_regularisation.setter
    def model_regularisation(self, value):
        """Sets the model regularisation configuration."""
        assert isinstance(value, (list, tuple)), "Model regularisation must be a list or tuple."
        self._model_regularisation = value

    @property
    def model_regularisation_misfit_history(self):
        """ Returns the history of misfits."""
        return self._model_regularisation_misfit_history

    @property
    def current_model(self):
        """ Returns the current models."""
        return self._current_model

    @property
    def current_models(self):
        """Returns the current model as a one-element list."""
        return [self._current_model]

    @property
    def initial_models(self):
        """Returns the initial model as a one-element list."""
        return [self._tracking_dict["general"]["initial_model"]] if self._tracking_dict["general"]["initial_model"] is not None else None

    @property
    def number_of_models(self):
        """Returns the number of models."""
        return 1

    @property
    def number_of_datasets(self):
        """Returns the number of datasets."""
        return self._number_of_datasets

    def run(self):
        """ Runs the Gauss-Newton inversion."""
        if self.verbose:
            logger.info("----------Running Gauss-Newton inversion.----------")

        if self._current_iteration == self._maximum_iterations:
            logger.info("----------Maximum number of iterations reached. Returning.----------")
            return

        while self._current_iteration < self._maximum_iterations:
            logger.info("----------Processing iteration %s----------", self._current_iteration+1)
            start_time_iteration = time.time()
            # Create iteration dictionary for tracking if necessary
            if self._current_iteration not in self._tracking_dict:
                self._tracking_dict[self._current_iteration] = {}

            #* Save current model if required
            if self._save_model_history:
                # Save the models
                self._tracking_dict[self._current_iteration]["models"] = self._current_model.copy()

            if self.verbose:
                logger.info("----------Start: Calculating geophysical jacobians and rhs.----------")
                start_time = time.time()

            # Set up the petrophysical jacobian and response and apply model transformation
            weighted_jacobian_data, rhs_data, _, data_misfit_list, chi_squared_list = \
                self.get_petrophysical_jacobian_and_rhs()
            
            if self.verbose:
                logger.info("Time taken to calculate geophysical jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating geophysical jacobians and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model regularisation jacobians and rhs.----------")
            start_time = time.time()

            # Set up the model regularisation
            jacobian_regularisation, rhs_regularisation, model_misfit = \
                self.get_model_regularisation_jacobian_and_rhs()
            
            if self.verbose:
                logger.info("Time taken to calculate model regularisation jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating model regularisation jacobians and rhs.----------")

            #* Save misfit values -  these are technically from the iteration before
            self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
            self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list
            self._tracking_dict[self._current_iteration][
                "single_model_regularisation_misfit"
            ] = model_misfit

            #* Check if the termination criterion is met
            if self._current_iteration > 0:
                curr_dict = self._tracking_dict.get(self._current_iteration, {})
                prev_dict = self._tracking_dict.get(self._current_iteration - 1, {})
                chi_curr = curr_dict.get("chi_squared")
                chi_prev = prev_dict.get("chi_squared")
                if chi_curr is not None and chi_prev is not None and len(chi_curr) > 0:
                    chi2_percentage_decrease_list = self.percent_decrease_in_chi2(self._current_iteration)
                    if self.verbose:
                        logger.info("Chi2 percentage decrease: %s", chi2_percentage_decrease_list)
                    if all([chi2_percentage_decrease < self._terminate_on_chi2_decrease for chi2_percentage_decrease in chi2_percentage_decrease_list]):
                        logger.info("Chi2 decrease criterion reached. Returning.")
                        break

            if self.verbose:
                logger.info("----------Start: Calculating the full jacobian and rhs.----------")
            start_time = time.time()

            #* Combine the jacobians and rhs
            if jacobian_regularisation is not None:
                full_jacobian = sP.sparse.vstack(
                    [weighted_jacobian_data, jacobian_regularisation],
                    format="csr"
                )
                full_rhs = np.concatenate([rhs_data, rhs_regularisation])
            else:
                full_jacobian = weighted_jacobian_data
                full_rhs = rhs_data

            if self.verbose:
                logger.info("Time taken to calculate the full jacobian and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating the full jacobian and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Solving linear system.----------")

            #* Calculate the model update
            model_update_small = self.solve_linear_system(
                A=full_jacobian,
                b=full_rhs,
                enable_scaling=True
            )

            if self.scheme == "creeping":
                update_vector = model_update_small
            elif self.scheme == "jumping":
                update_vector = model_update_small - self.current_model.transformed_model
            else:
                raise ValueError("Invalid scheme provided.")

            if self.verbose:
                logger.info("----------End: Solving linear system.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model update.----------")
            start_time = time.time()

            petrophysical_model_update = self.get_model_update_from_full_inversion_results(
                model_update_vector=update_vector,
                force_disable_clipping=False
            )
            self.apply_model_updates(
                model_update=petrophysical_model_update
            )

            if self.verbose:
                logger.info("Time taken to calculate model update: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating model update.----------")
            

            #* Save updates and models
            self._tracking_dict[self._current_iteration]["model_update"] = petrophysical_model_update
            self._tracking_dict[self._current_iteration]["model"] = self.current_model.copy()
            self._current_iteration += 1
            if self.verbose:
                logger.info("Time taken for iteration %s: %.2f seconds.", self._current_iteration, time.time()-start_time_iteration)
                logger.info("----------End: Iteration.----------")
        
        if self._current_iteration not in self._tracking_dict:
            self._tracking_dict[self._current_iteration] = {}
        #* Save the final models
        if self._save_model_history:
            self._tracking_dict[self._current_iteration]["models"] = self._current_model.copy()
            self._model_history.append(self._current_model.copy())

        #* Calculate the final data misfit
        _, _, _, data_misfit_list, chi_squared_list = self.get_petrophysical_jacobian_and_rhs()
        if self.verbose:
            logger.info("%s", data_misfit_list)
        self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
        self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list
        
        #* Calculate the final single model regularisation misfit
        if self._number_of_single_model_regularisation > 0:
            _, _, model_misfit = self.get_model_regularisation_jacobian_and_rhs()
            self._tracking_dict[self._current_iteration][
                "single_model_regularisation_misfit"
            ] = model_misfit

        #* Adjust maximum iteration
        if self._current_iteration == self._maximum_iterations:
            logger.info("Maximum number of iterations reached.")
        else:
            logger.info("Finished Gauss-Newton inversion after %s iterations. Overwriting maximum iterations.", self._current_iteration)
            self._maximum_iterations = self._current_iteration
        logger.info("Gauss-Newton inversion finished.")

    def get_petrophysical_jacobian_and_rhs(self):
        """ Returns the petrophysical jacobian and right hand side."""
        # Initiate weighted jacobian and response
        weighted_jacobian_inversion, weighted_response = self.data.get_jacobian_and_response(
            petrophysical_model=self.current_model,
            weight_list=self.data_weight_list,
            domain="inversion"
        )
        # Break the response into single data misfits
        weighted_response_list = []
        counter=0
        for i in range(self._number_of_datasets):
            length_of_dataset = self.data._data_observed_list[i].shape[0]
            weighted_response_list.append(
                weighted_response[counter:counter+length_of_dataset]
            )
            counter += length_of_dataset
        # Calculate the data misfit
        data_misfit_list = []
        for i in range(self._number_of_datasets):
            data_misfit_list.append(
                np.linalg.norm(weighted_response_list[i] - self.data._data_observed_list[i])
            )

        # Create a weighted observed data vector
        weighted_observed_data_list = [
            self.data_weight_list[i] * self.data._data_observed_list[i]
            for i in range(self._number_of_datasets)
        ]
        weighted_observed_data_vector = np.concatenate(weighted_observed_data_list)
        weighted_residual_vector = weighted_observed_data_vector - weighted_response

        data_misfit_list = [
            np.linalg.norm(wobs-wres)
            for wobs, wres in zip(weighted_observed_data_list, weighted_response_list)
        ]
        data_misfit = np.linalg.norm(weighted_residual_vector)

        #* Calculate chi squared list
        err_list = [
            self.data.get_err(i)
            for i in range(self._number_of_datasets)
        ]
        chi_squared_list = [
            (w_obs.size)**-1 * np.sum(((w_obs-w_resp)/(err*w_obs))**2) for w_obs, w_resp, err, weight in zip(weighted_observed_data_list, weighted_response_list, err_list, self.data_weight_list)
        ]

        # Create rhs data
        if self.scheme == "creeping":
            rhs_data = weighted_residual_vector
        elif self.scheme == "jumping":
            rhs_data = weighted_residual_vector + weighted_jacobian_inversion @ self.current_model.transformed_model

        return weighted_jacobian_inversion, rhs_data, data_misfit, data_misfit_list, chi_squared_list

    def get_model_regularisation_jacobian_and_rhs(self):
        """Returns the model regularisation jacobian and right hand side."""
        if self._model_regularisation == []:
            return None, None, None

        # Collect all the jacobians and rhs
        jacobian_list = []
        rhs_list = []
        model_regularisation_misfit = []
        model = self.current_model

        for reg in self._model_regularisation:
            jacobian = reg.get_jacobian(
                physics_and_data=self.data,
                model_info=model,
                domain="inversion")

            phi = reg.get_phi(
                physics_and_data=self.data,
                model_info=model,
                weighted=True,
                )

            if self.scheme == "creeping":
                rhs = reg.get_rhs_creeping(
                    physics_and_data=self.data,
                    model_info=model
                    )
            elif self.scheme == "jumping":
                rhs = reg.get_rhs_jumping(
                    physics_and_data=self.data,
                    model_info=model,
                    domain="inversion"
                    )
            else:
                raise ValueError("Invalid scheme provided.")

            jacobian_list.append(jacobian)
            rhs_list.append(rhs)
            model_regularisation_misfit.append(phi)

        # Concatenate the rhs
        rhs = np.concatenate(rhs_list, axis=0).flatten()

        model_regularisation_misfit = np.concatenate(
            model_regularisation_misfit, axis=0
            ).flatten()
        model_misfit = np.linalg.norm(model_regularisation_misfit)

        # Vertically stack the jacobian
        jacobian = sP.sparse.vstack(blocks=jacobian_list, format="csr")

        #* Remove the rows that are coupled
        jacobian, rhs = self.remove_rows_coupling_trusted_untrusted(jacobian, rhs, region_of_interest=self._region_of_interest)

        return jacobian, rhs, model_misfit

    
    def get_model_update_from_full_inversion_results(self, model_update_vector, force_disable_clipping=False):
        """Updates the model from the inversion results."""

        #* Temporary update the model to get update in petrophysical domain
        old_petrophysical_model = self.current_model.model.copy()
        self.current_model.transformed_model = self.current_model.transformed_model + model_update_vector
        new_petrophysical_model = self.current_model.model.copy()
        petrophysical_update = new_petrophysical_model - old_petrophysical_model
        self.current_model.model = old_petrophysical_model

        if force_disable_clipping:
            clipped_petrophysical_update = petrophysical_update
        else:
            clipped_petrophysical_update = self.clip_model_vector(petrophysical_update)
        return clipped_petrophysical_update

    def apply_model_updates(self, model_update):
        """ Applies the model updates to the current model."""
        #* Apply the model update
        self.current_model.model = self.current_model.model + model_update
        return

    # Delegating implementations (solver/clipping/decoupling) are provided by GaussNewtonCore
    
# Plotting functions
    def plot_data_misfit_history(self, ax=None, figsize=(10, 6), ylim=(0,None), normalise=False):
        """ Plots the data misfit history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        iterations, data_misfit = self.assemble_iteration_vector_from_tracking_dict(
            "data_misfit"
            )
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

    def plot_model_regularisation_misfit_history(self, ax=None, figsize=(10, 6)):
        """ Plots the single model regularisation misfit history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        iterations, model_regularisation_misfit =\
            self.assemble_iteration_vector_from_tracking_dict(
                "single_model_regularisation_misfit"
                )
        ax.plot(
            iterations,
            model_regularisation_misfit,
            label="Single Model Regularisation Misfit"
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Model Regularisation Misfit")
        ax.set_title("Model Regularisation Misfit History")
        return fig, ax

    def plot_misfit_history(self, ax=None, figsize=(10,6), ylim=(0, None)):
        """ Plot all misfit histories."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        iterations_data, data_misfit =\
            self.assemble_iteration_vector_from_tracking_dict(
            "data_misfit"
            )
        iterations_model_regularisation, model_regularisation_misfit =\
            self.assemble_iteration_vector_from_tracking_dict("single_model_regularisation_misfit")

        for no_dataset in range(self._number_of_datasets):
            data_misfit_vector = [misfit_list[no_dataset] for misfit_list in data_misfit]
            ax.plot(
                iterations_data,
                data_misfit_vector,
                label=f"Data Misfit {no_dataset+1}"
                )

        if len(model_regularisation_misfit) > 0:
            ax.plot(
                iterations_model_regularisation,
                model_regularisation_misfit,
                label="Model Regularisation Misfit"
                )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Misfit")
        ax.set_title("Misfit History")
        if ylim[1] is not None:
            ylim[1] = 1.1 * max([np.max(data_misfit), np.max(model_regularisation_misfit)])
        ax.set_ylim(ylim)
        ax.legend()
        return fig, ax

    def plot_current_models(
            self,
            include_initial_model=False,
            include_geophysical_models=False,
            figsize=None,
            **kwargs
            ):
        """ Plot the current model."""
        number_of_axes = 1
        if include_initial_model:
            initial_model = self.initial_model
            number_of_axes += 1
        if include_geophysical_models:
            number_of_axes += self._number_of_datasets
            geophysical_models = [
                self.data.get_geophysical_model(self.current_model, i)
                for i in range(self._number_of_datasets)
                ]

        figsize = (number_of_axes*5, 5) if figsize is None else figsize
        fig, axs = plt.subplots(1, number_of_axes, figsize=figsize)
        if number_of_axes == 1:
            axs = [axs]


        counter = 0
        # Plot the initial models
        if include_initial_model:
            ax = axs[counter]
            initial_model.plot_model(ax=ax, **kwargs)
            counter += 1
            ax.set_title("Initial Model")

        # Plot the current model
        ax = axs[counter]
        self.current_model.plot_model(ax=ax, **kwargs)
        ax.set_title("Current Petrophysical model")
        counter += 1

        # Plot the geophysical models
        data_set = 0
        if include_geophysical_models:
            for i in range(self._number_of_datasets):
                model_temp = self.current_model.copy()
                model_temp._transformation = None
                model_temp.model = geophysical_models[i]
                ax = axs[counter]
                model_temp.plot_model(ax=ax, **kwargs)
                ax.set_title(f"Geophysical Model {data_set+1}")
                data_set += 1
                counter += 1
        return fig, axs

    def show_region_of_interest(self, markersize=1, marker="o", mode="triang"):
        """ Shows the region of interest for all models."""
        fig, ax = plt.subplots(1, figsize=(5, 5))
        mesh_info = self.current_model.mesh_info
        mesh_info.show_region_of_interest(ax=ax, markersize=markersize, marker=marker, mode=mode)
        return fig, ax

    def show_regularisation_coverage(self, ax=None, cMap="turbo", cMin=0, cMax=None):
        """ Shows the coverage of the regularisation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        jacobian, _, _ = self.get_model_regularisation_jacobian_and_rhs()
        jacobian_nnz = np.abs(jacobian.toarray())>0
        jacobian_nnz = np.sum(jacobian_nnz, axis=0)

        if cMax is None:
            cMax = np.max(jacobian_nnz)

        #* Show the mesh
        import pygimli as pg
        _= pg.show(self._mesh_info.mesh, data=jacobian_nnz, ax=ax, cMap=cMap, cMin=cMin, cMax=cMax, showMesh=True)
        return fig, ax
    