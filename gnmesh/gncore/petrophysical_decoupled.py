"""
Decoupled petrophysical Gauss-Newton manager.

Provides `GaussNewtonPetrophysicalDecoupled` for petrophysical inversion
workflows that treat petrophysical and geophysical updates in a partially
decoupled fashion. This module focuses on configuration and orchestration
of the inversion loop; heavy numerical work is delegated to the shared core
and regularisation modules.
"""

import logging
import numpy as np
import scipy as sP
import matplotlib.pyplot as plt
from gnmesh import regularisation as reG
import time

from .gaussnewtoncore import GaussNewtonCore

logger = logging.getLogger(__name__)

class GaussNewtonPetrophysicalDecoupled(GaussNewtonCore):
    """Gauss-Newton manager for decoupled petrophysical inversion."""
    def __init__(
            self,
            mesh_info,
            geophysical_data_list,
            petrophysical_trust_region,
            initial_model=None,
            single_model_regularisation=None,
            decouple_regularisation_regions=None,
            maximum_iterations=100,
            save_model_history=True,
            scheme="creeping",
            verbose=True,
            inversion_settings=None,
    ):
        """
        Initialise decoupled petrophysical Gauss-Newton manager.

        Parameters
        ----------
        mesh_info : object
            Mesh and topology information (must provide `mesh.cellCount()` and `region_of_interest`).
        geophysical_data_list : list
            List of geophysical data manager instances used for the coupled inversion.
        petrophysical_trust_region : np.ndarray
            Boolean mask indicating trusted petrophysical cells.
        initial_model : object, optional
            Initial `ModelInfo`-like object (default: empty list).
        single_model_regularisation : list or None
            Regularisation objects to apply to petrophysical models.
        decouple_regularisation_regions : tuple or None
            Regions used to decouple regularisation (see usage in `remove_coupling_from_regions`).
        maximum_iterations : int
            Maximum number of Gauss-Newton iterations.
        save_model_history : bool
            Whether to keep model history.
        scheme : str
            'creeping' or 'jumping' update scheme.
        verbose : bool
            Verbosity flag for print statements.
        inversion_settings : dict or None
            Additional settings controlling the inversion process.
        """
        if initial_model is None:
            initial_model = []
        if single_model_regularisation is None:
            single_model_regularisation = []

        self.verbose = verbose

        # Set mesh info
        self._mesh_info = mesh_info

        # Check if the petrophysical data is a list or tuple
        if not isinstance(geophysical_data_list, (list, tuple)):
            geophysical_data_list = [geophysical_data_list]
        self._data = geophysical_data_list

        # Set weights
        data_weight_list = [data.weight for data in geophysical_data_list]
        logger.info("Data weights: %s", data_weight_list)

        self._number_of_datasets = len(geophysical_data_list)

        if self.verbose:
            logger.info("Number of datasets: %s", self._number_of_datasets)
            if self._number_of_datasets == 0:
                raise ValueError("No datasets provided.")

        # Check if single_model_regularisation is a list or tuple of lists
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
        
        # Check if the number of regularisation lists matches the number of datasets
        if len(self._single_model_regularisation) != self._number_of_datasets:
            logger.error("Number of single model regularisation lists must match the number of datasets.")
            raise ValueError

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
            self._tracking_dict["general"]["initial_models"] = [
                initial_model.copy(), initial_model.copy()
                ]
        else:
            self._tracking_dict["general"]["initial_models"] = None
        self._current_model = initial_model.copy()

        # Set the region of interest
        self._region_of_interest = mesh_info.region_of_interest
        if not np.all(self._region_of_interest):
            logger.warning("Some cells are not part of the inversion region.")

        # Set the trust region
        self.petrophysical_trust_region = petrophysical_trust_region

        # Set numerical scheme
        self.scheme = scheme

        # Set model history handling through the shared core API
        self.save_model_history = save_model_history

        # Set scaling
        self.scaling = "column_sum_l1"

        # Initialise the regularisation settings
        if inversion_settings is None:
            if self.verbose:
                logger.info("No regularisation settings provided. Using default settings.")
            inversion_settings = {
                "decouple_regularisation_trustregion": True,
                "domain": "petrophysical",
                "add_xg_for_untrusted_region": False,
                "xg_weight": 1.0,
                "update_petro_trust_region": False,
                "update_petro_trust_region_function": None,
                "update_after_iteration": 1,
                "fix_coupled_region": np.zeros(mesh_info.mesh.cellCount()),
                "individual_updates_with_xg": False,
                "minimum_petro_component_size": 0.0,
            }
            logger.info("No regularisation settings provided. Regularisation settings: %s", inversion_settings)
        else:
            assert isinstance(inversion_settings, dict), "Regularisation settings must be a dictionary."

            assert "decouple_regularisation_trustregion" in inversion_settings, "Decoupled key must be in regularisation settings."
            assert inversion_settings["decouple_regularisation_trustregion"] in [True, False], "Decoupled must be either True or False."

            assert "domain" in inversion_settings, "Domain key must be in regularisation settings."
            assert inversion_settings["domain"] in ["geophysical", "petrophysical"], "Domain must be either geophysical or petrophysical."
            
            assert "add_xg_for_untrusted_region" in inversion_settings, "add_xg_for_untrusted_region key must be in regularisation settings."
            assert isinstance(inversion_settings["add_xg_for_untrusted_region"], bool), "add_xg_for_untrusted_region must be a boolean."

            if inversion_settings["add_xg_for_untrusted_region"] or inversion_settings["individual_updates_with_xg"]:
                assert "xg_weight" in inversion_settings, "xg_weight key must be in regularisation settings."
                assert isinstance(inversion_settings["xg_weight"], (int, float)), "xg_weight must be an integer or float."
                assert inversion_settings["xg_weight"] > 0, "xg_weight must be greater than 0."

                self._xg_regularisation = reG.XGradient(
                    weight=inversion_settings["xg_weight"],
                )
            
            if inversion_settings["individual_updates_with_xg"]:
                self._xg_reference_regularisation = reG.XGradient(
                    weight=inversion_settings["xg_weight"],
                )

            assert "update_petro_trust_region" in inversion_settings, "update_petro_trust_region key must be in regularisation settings."
            assert isinstance(inversion_settings["update_petro_trust_region"], bool), "update_petro_trust_region must be a boolean."
            if inversion_settings["update_petro_trust_region"]:
                assert "update_petro_trust_region_function" in inversion_settings, "update_petro_trust_region_function key must be in regularisation settings."
                assert callable(inversion_settings["update_petro_trust_region_function"]), "update_petro_trust_region_function must be a callable."
                assert "update_after_iteration" in inversion_settings, "update_after_iteration key must be in regularisation settings."
                assert isinstance(inversion_settings["update_after_iteration"], int), "update_after_iteration must be an integer."
                assert inversion_settings["update_after_iteration"] >= 0, "update_after_iteration must be greater or equal than 0."
            
            assert "fix_coupled_region" in inversion_settings, "fix_coupled_region key must be in regularisation settings."
            assert isinstance(inversion_settings["fix_coupled_region"], np.ndarray), "fix_coupled_region must be a numpy array."
            assert len(inversion_settings["fix_coupled_region"]) == mesh_info.mesh.cellCount(), "fix_coupled_region must have the same length as the mesh cell count."
            assert all(isinstance(ent, (bool, np.bool_)) for ent in inversion_settings["fix_coupled_region"]), "fix_coupled_region must be a boolean array."

            assert "individual_updates_with_xg" in inversion_settings, "individual_updates_with_xg key must be in regularisation settings."
            assert isinstance(inversion_settings["individual_updates_with_xg"], bool), "individual_updates_with_xg must be a boolean."

            assert "minimum_petro_component_size" in inversion_settings, "minimum_petro_component_size key must be in regularisation settings."

            assert isinstance(inversion_settings["minimum_petro_component_size"], (int, float)), "minimum_petro_component_size must be an integer or float."

            assert inversion_settings["minimum_petro_component_size"] >= 0, "minimum_petro_component_size must be greater or equal than 0."

        if not "update_after_iteration" in inversion_settings:
            inversion_settings["update_after_iteration"] = np.inf
        
        if not "enable_petro_update_after_chi_decrease" in inversion_settings:
            inversion_settings["enable_petro_update_after_chi_decrease"] = -np.inf

        # Set maximum update per step
        self._maximum_update_per_step = [(-np.inf, np.inf)] * (1 + self._number_of_datasets)

        self._inversion_settings = inversion_settings.copy()
        logger.info("Regularisation settings: %s", self._inversion_settings)

        # Set enable petrophysical trustregion update
        self.update_petrophysical_trustregion_enabled = False

        # Initialise the numerical solver by default
        self.num_solver = "cupy_sparse"

        # Initisalise the history of misfits
        self._data_misfit_history = []
        self._model_regularisation_misfit_history = []
        self._xg_misfit_history = []

        #* Set up decoupling of regularisation
        if decouple_regularisation_regions is not None:
            assert isinstance(decouple_regularisation_regions, (list, tuple)), "Decouple regularisation must be a list or tuple."
            assert len(decouple_regularisation_regions) == 2, "Decouple regularisation must have two elements."
            assert isinstance(decouple_regularisation_regions[0], np.ndarray), "Decouple regularisation must be a numpy array."
            assert decouple_regularisation_regions[0].shape[0] == self._mesh_info.mesh.cellCount(), "Decouple regularisation must have the same length as the number of cells."
            assert isinstance(decouple_regularisation_regions[1], list), "Decouple regularisation must be a list."
            assert all(isinstance(reg, (list, tuple)) for reg in decouple_regularisation_regions[1]), "Decouple regularisation must be a list of lists or tuples."
            if self.verbose:
                logger.info("Decoupling regularisation provided.")
        else:
            if self.verbose:
                logger.info("No decoupling regularisation provided.")
        self._decouple_regularisation_regions = decouple_regularisation_regions

        # Mirror the shared core configuration as well.
        self.decouple_regularisation = decouple_regularisation_regions

        #* Set the termination criterion
        self.terminate_on_chi2_decrease = 0.0

        self._model_history = []

        if self.verbose:
            logger.info("Gauss-Newton inversion initialised.")

    @property
    def mesh_info(self):
        """ Returns the mesh."""
        return self._mesh_info

    @property
    def region_of_interest(self):
        """ Returns the region of interest."""
        return self._region_of_interest

    @property
    def petrophysical_trust_region(self):
        """ Returns the petrophysical trust region."""
        return self._petrophysical_trust_region

    @petrophysical_trust_region.setter
    def petrophysical_trust_region(self, value):
        """ Sets the petrophysical trust region."""
        assert isinstance(value, np.ndarray), "Petrophysical trust region must be a numpy array."
        assert value.shape[0] == self._mesh_info.mesh.cellCount(), "Petrophysical trust region must have the same length as the mesh cell count."
        assert all(isinstance(ent, (bool, np.bool_)) for ent in value), "Petrophysical trust region must be a boolean array."
        self._petrophysical_trust_region = value
        # keep model's internal attribute name consistent
        self._current_model.petrophysical_trust_region = value

    @property
    def data(self):
        """ Returns the geophysical data."""
        return self._data

    @property
    def maximum_iterations(self):
        """ Returns the maximum number of iterations."""
        return self._maximum_iterations

    @property
    def current_iteration(self):
        """ Returns the current iteration."""
        return self._current_iteration

    @property
    def data_misfit_history(self):
        """ Returns the history of misfits."""
        return self._data_misfit_history

    @property
    def model_regularisation_misfit_history(self):
        """ Returns the history of misfits."""
        return self._model_regularisation_misfit_history

    @property
    def current_model(self):
        """ Returns the current models."""
        return self._current_model

    @property
    def tracking_dict(self):
        """ Returns the tracking dictionary."""
        return self._tracking_dict

    @property
    def initial_models(self):
        """ Returns the initial models."""
        return self._tracking_dict["general"]["initial_models"]

    @property
    def maximum_update_per_step(self):
        """ Returns the maximum update per step."""
        return self._maximum_update_per_step

    @maximum_update_per_step.setter
    def maximum_update_per_step(self, value):
        """ Sets the maximum update per step."""
        assert isinstance(value, (tuple, list)), "Maximum update per step must be a tuple or list."
        assert all(isinstance(v, (tuple, list)) for v in value), "Maximum update per step must be a tuple/list of tuples/lists."
        assert len(value) == 1 + len(self.data), "Maximum update per step steps for petro model and each geophysical data."
        assert all(len(v) == 2 for v in value), "Maximum update per step must be a tuple of tuples with two elements."
        assert all(isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)) for v in value), "Maximum update per step must be a tuple of tuples with integers or floats."
        assert all(v[0] < v[1] for v in value), "Maximum update per step must be a tuple of tuples with the first element smaller than the second element."
        self._maximum_update_per_step = value

    @property
    def update_petrophysical_trustregion_enabled(self):
        """ Returns whether the petrophysical trust region update is enabled."""
        return self._update_petro_trustregion_enabled
    
    @update_petrophysical_trustregion_enabled.setter
    def update_petrophysical_trustregion_enabled(self, value):
        """ Sets whether the petrophysical trust region update is enabled."""
        assert isinstance(value, bool), "update_petrophysical_trustregion_enabled must be a boolean."
        self._update_petro_trustregion_enabled = value

    def run(self):
        """
        Execute the Gauss-Newton inversion loop.

        The method assembles per-iteration jacobians and RHS, solves the linearised
        system, computes model updates, applies clipping and trust-region updates,
        and records tracking information. The method modifies `self._current_model`
        and related tracking dictionaries in-place.
        """
        if self.verbose:
            logger.info("----------Running Gauss-Newton inversion.----------")

        if self._current_iteration == self._maximum_iterations:
            logger.info("----------Maximum number of iterations reached. Returning.----------")
            return

        while self._current_iteration < self._maximum_iterations:
            logger.info("----------Processing iteration %s----------", self._current_iteration+1)
            start_time_iteration = time.time()
            #* Create iteration dictionary for tracking if necessary
            if self._current_iteration not in self._tracking_dict:
                self._tracking_dict[self._current_iteration] = {}

            # Save current model
            self._tracking_dict[self._current_iteration]["initial_model"] = self._current_model.copy()

            if self.verbose:
                logger.info("----------Start: Calculating geophysical jacobians and rhs.----------")
            
            start_time = time.time()
            #* Set up the individual petrophysical jacobians and responses
            iwjacobian_data_list, iwrhs_data_list, data_misfit_list, iwresponse_data_list, chi_squared_list = \
                self.get_individual_inversion_jacobians_and_rhs()
            
            if self.verbose:
                logger.info("Time taken to calculate geophysical jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating geophysical jacobians and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model regularisation jacobians and rhs.----------")

            start_time = time.time()
            #* Set up the individual model regularisation
            iwjacobian_regularisation_list, iwrhs_regularisation_list, model_misfit_list = \
                self.get_individual_model_regularisation_jacobian_and_rhs()
            if self.verbose:
                logger.info("Time taken to calculate model regularisation jacobians and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating model regularisation jacobians and rhs.----------")

            #* Save misfit values -  these are technically from the iteration before
            self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
            self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list
            self._tracking_dict[self._current_iteration][
                "single_model_regularisation_misfit"
            ] = np.linalg.norm(model_misfit_list)
            if self._inversion_settings["add_xg_for_untrusted_region"] and self._current_iteration >= self._inversion_settings["update_after_iteration"]:
                self._tracking_dict[self._current_iteration]["dual_model_regularisation_misfit"] = np.linalg.norm(xg_jacobian_misfit)

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
                        if self._inversion_settings["update_petro_trust_region"] and not self.update_petrophysical_trustregion_enabled:
                            self.update_petrophysical_trustregion_enabled = True
                            if self.verbose:
                                logger.info("----------Petrophysical trust region update enabled by termination criterion.----------")
                        else:
                            logger.info("----------Termination criterion reached: Chi2 decrease below threshold. Returning.----------")
                            break

            #* Recalculate the petro trust region -> Calculate the individual model updates
            if self._inversion_settings["update_petro_trust_region"]:
                if self.current_iteration >= self._inversion_settings["update_after_iteration"]:
                    if self.verbose and not self.update_petrophysical_trustregion_enabled:
                        logger.info("----------Petrophysical trust region update enabled by iteration.----------")
                        self.update_petrophysical_trustregion_enabled = True

                if self._current_iteration >0 and all([chi2_percentage_decrease < self._inversion_settings["enable_petro_update_after_chi_decrease"] for chi2_percentage_decrease in chi2_percentage_decrease_list]):
                    if self.verbose and not self.update_petrophysical_trustregion_enabled:
                        logger.info("----------Petrophysical trust region update enabled by chi2 decrease.----------")
                        self.update_petrophysical_trustregion_enabled = True

            if self._inversion_settings["update_petro_trust_region"] and self.update_petrophysical_trustregion_enabled:
                if self.verbose:
                    logger.info("----------Start: Update petro trust region.----------")
                n_petro_in_roi = np.sum(self.petrophysical_trust_region[self._region_of_interest])
                start_time = time.time()
                #* Calculate the individual model updates
                if self._inversion_settings["individual_updates_with_xg"]:
                    logger.info("----------Using XG for individual model updates.----------")
                    individual_model_update_list = self.get_individual_model_updates_xg(
                        iwjacobian_data_list=iwjacobian_data_list,
                        iwrhs_data_list=iwrhs_data_list,
                        iwjacobian_regularisation_list=iwjacobian_regularisation_list,
                        iwrhs_regularisation_list=iwrhs_regularisation_list,
                        clip_update=True,
                    )
                else:
                    logger.info("----------Not using XG for individual model updates.----------")
                    individual_model_update_list = self.get_individual_model_updates_wo_xg(
                        iwjacobian_data_list=iwjacobian_data_list,
                        iwrhs_data_list=iwrhs_data_list,
                        iwjacobian_regularisation_list=iwjacobian_regularisation_list,
                        iwrhs_regularisation_list=iwrhs_regularisation_list,
                        clip_update=True,
                    )

                #* Calculate the petro trust region
                # (legacy) previously assembled long-format individual updates here — removed
                new_petrophysical_trust_region, diverging, significant = self._inversion_settings["update_petro_trust_region_function"](
                    list_models=[self.current_model.model_petro, self.current_model.model_petro],
                    list_model_updates=individual_model_update_list,
                    current_petrophysical_trust_region=self.petrophysical_trust_region,
                )

                #* Save diverging and significant and individual model updates
                self._tracking_dict[self._current_iteration]["diverging"] = diverging
                self._tracking_dict[self._current_iteration]["significant"] = significant
                self._tracking_dict[self._current_iteration]["individual_model_updates"] = individual_model_update_list.copy()
                self._tracking_dict[self._current_iteration]["individual_jacobians_data"] = iwjacobian_data_list.copy()
                self._tracking_dict[self._current_iteration]["individual_rhs_data"] = iwrhs_data_list.copy()
                self._tracking_dict[self._current_iteration]["individual_jacobians_regu"] = iwjacobian_regularisation_list.copy()
                self._tracking_dict[self._current_iteration]["individual_rhs_regu"] = iwrhs_regularisation_list.copy()

                #* Revert matrices to geophysical matrices
                for method_number, i_jacobian in enumerate(iwjacobian_data_list):
                    transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(
                        method_number=method_number,
                    )
                    i_jacobian = i_jacobian.multiply(1/ transformation_vector)
                    iwjacobian_data_list[method_number] = i_jacobian

                for method_number, regularisation_list in enumerate(iwjacobian_regularisation_list):
                    transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(
                        method_number=method_number,
                    )
                    for regu_number, regularisation_matrix in enumerate(regularisation_list):
                        regularisation_matrix = regularisation_matrix.multiply(1/ transformation_vector)
                        iwjacobian_regularisation_list[method_number][regu_number] = regularisation_matrix


                #* Update the petrophysical trust region
                new_petrophysical_trust_region = np.logical_or(new_petrophysical_trust_region, self._inversion_settings["fix_coupled_region"])

                #* Remove small components from the petrophysical trust region
                if self._inversion_settings["minimum_petro_component_size"] > 0:
                    self.remove_small_components_from_petrophysical_trust_region(
                        new_petrophysical_trust_region=new_petrophysical_trust_region,
                        min_size=self._inversion_settings["minimum_petro_component_size"]
                        )
                    
                #* Apply new model transformations to individual matrices
                for method_number, i_jacobian in enumerate(iwjacobian_data_list):
                    transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(
                        method_number=method_number,
                    )
                    i_jacobian = i_jacobian.multiply(transformation_vector)
                    iwjacobian_data_list[method_number] = i_jacobian.tocsr()

                for method_number, regularisation_list in enumerate(iwjacobian_regularisation_list):
                    transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(
                        method_number=method_number,
                    )
                    for regu_number, regularisation_matrix in enumerate(regularisation_list):
                        regularisation_matrix = regularisation_matrix.multiply(transformation_vector)
                        iwjacobian_regularisation_list[method_number][regu_number] = regularisation_matrix.tocsr()

                    if self.verbose:
                        n_petrophysical_in_roi_new = np.sum(self.petrophysical_trust_region[self._region_of_interest])
                        logger.info("Number of trusted cells in ROI changed from %s to %s.", n_petro_in_roi, n_petrophysical_in_roi_new)
                        logger.info("Time taken for petrophysical trust region update: %.2f seconds", time.time()-start_time)
                        logger.info("----------End: Petrophysical trust region update complete.----------")

            if self.verbose:
                logger.info("----------Start: Calculating the full jacobian and rhs.----------")
            start_time = time.time()

            #* Get the full petrophysical jacobian and calculate update
            weighted_jacobian_data, rhs_data, data_misfit, data_misfit_list = \
                self.get_full_petrophysical_jacobian_and_rhs(
                    weighted_jacobian_list=iwjacobian_data_list,
                    weighted_response_list=iwresponse_data_list
                )

            jacobian_regularisation, rhs_regularisation, model_misfit = \
                self.get_full_model_regularisation_jacobian_and_rhs(
                    weighted_jacobian_list=iwjacobian_regularisation_list,
                    weighted_rhs_list=iwrhs_regularisation_list,
                    model_regularisation_misfit_list=model_misfit_list
                )

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

            # Add XG for untrusted region
            if self._inversion_settings["add_xg_for_untrusted_region"]:
                self._xg_regularisation.weight = self._inversion_settings["xg_weight"]
                geo_model_list = self._current_model.get_individual_geo_model_instances()
                xg_jacobian = self._xg_regularisation.get_jacobian(
                    physics_and_data=self.data,
                    model_info_list=geo_model_list,
                ).tocsr()

                xg_jacobian_misfit = self._xg_regularisation.get_phi(
                    physics_and_data=self.data,
                    model_info_list=geo_model_list,
                    weighted=True,
                )
                if self.scheme == "creeping":
                    xg_rhs = self._xg_regularisation.get_rhs_creeping(
                        physics_and_data=self.data,
                        model_info_list=geo_model_list,
                    )
                elif self.scheme == "jumping":
                    xg_rhs = self._xg_regularisation.get_rhs_jumping(
                        physics_and_data=self.data,
                        model_info_list=geo_model_list,
                        domain="inversion",
                    )
                else:
                    raise ValueError("Invalid scheme provided.")

                #* Assemble the xg jacobian according to the petrophysical_trust_region
                trusted_indices_in_roi = self.petrophysical_trust_region[self._region_of_interest]
                trusted_indices_in_roi_double = np.concatenate([trusted_indices_in_roi, trusted_indices_in_roi])
                trusted_cols = np.where(trusted_indices_in_roi_double)[0]
                untrusted_indices_in_roi = ~self.petrophysical_trust_region[self._region_of_interest]
                untrusted_indices_in_roi_double = np.concatenate([untrusted_indices_in_roi, untrusted_indices_in_roi])
                untrusted_cols = np.where(untrusted_indices_in_roi_double)[0]

                # 1 - remove rows coupling the petrophysical trusted and untrusted regions as petrophysical and xg coupling is not consistent
                logger.info("Decoupling XG. Removing rows coupling trusted and untrusted regions. XG jacobian shape before: %s", xg_jacobian.shape)
                xg_rows_to_remove = []
                for row in range(xg_jacobian.shape[0]):
                    row_data = xg_jacobian.getrow(row)
                    has_trusted = row_data[:, trusted_cols].nnz > 0
                    has_untrusted = row_data[:, untrusted_cols].nnz > 0
                    if has_trusted and has_untrusted:
                        xg_rows_to_remove.append(row)

                xg_rows_to_keep = np.setdiff1d(np.arange(xg_jacobian.shape[0]), xg_rows_to_remove)

                xg_jacobian = xg_jacobian[xg_rows_to_keep,:]
                xg_rhs = xg_rhs[xg_rows_to_keep]

                logger.info("XG jacobian shape after removing rows coupling trusted and untrusted regions: %s", xg_jacobian.shape)

                # 2 - mask columns according to the petrophysical trust region (keep all columns but zero out trusted region columns to enforce xg only in untrusted region)

                # Prepare transformation vector
                transformation_vector_list = [
                    self.current_model.get_transformed_model_gradient_from_geo(
                        method_number=method_number,
                    )
                    for method_number in range(len(self.data))
                ]
                transformation_vector = np.concatenate(transformation_vector_list)
                transformation_vector = transformation_vector[untrusted_cols]

                # Assemble the matrix
                xg_jacobian = sP.sparse.hstack(
                    [
                        sP.sparse.csr_matrix((xg_jacobian.shape[0], np.sum(self.petrophysical_trust_region[self._region_of_interest]))),
                        xg_jacobian[:,untrusted_cols].multiply(transformation_vector)
                    ],
                    format="csr"
                )

                full_jacobian = sP.sparse.vstack(
                    [full_jacobian, xg_jacobian],
                    format="csr"
                )
                full_rhs = np.concatenate([full_rhs, xg_rhs])

            if self.verbose:
                logger.info("Time taken to calculate the full jacobian and rhs: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Calculating the full jacobian and rhs.----------")

            if self.verbose:
                logger.info("----------Start: Solving linear system.----------")
            # Calculate the model update
            model_update_small = self.solve_linear_system(
                A=full_jacobian,
                b=full_rhs,
                enable_scaling=True
            )

            if self.scheme == "creeping":
                update_vector = model_update_small
            elif self.scheme == "jumping":
                #! this is not correct
                update_vector = model_update_small -\
                      self.current_model.transformed_model[self._region_of_interest]
            else:
                raise ValueError("Invalid scheme provided.")

            if self.verbose:
                logger.info("----------End: Solving linear system.----------")

            if self.verbose:
                logger.info("----------Start: Calculating model update.----------")
            start_time = time.time()

            #* Update the model
            petrophysical_model_update_tuple, new_model_tuple = self.get_model_updates_from_full_inversion_results(
                model_update_vector=update_vector,
                force_disable_clipping=False,
            )
            self.apply_model_updates(
                new_model_tuple=new_model_tuple,
            )

            if self.verbose:
                logger.info("Time taken to calculate model update: %.2f seconds.", time.time()-start_time)
                logger.info("----------End: Model update.----------")

            # Save updates and sizes
            self._tracking_dict[self._current_iteration]["petrophysical_trust_region"] = self.petrophysical_trust_region.copy()
            self._tracking_dict[self._current_iteration]["petrophysical_model_update_size"] = np.array([np.linalg.norm(update) for update in petrophysical_model_update_tuple]).copy()
            self._tracking_dict[self._current_iteration]["petrophysical_model_update"] = petrophysical_model_update_tuple
            self._tracking_dict[self._current_iteration]["final_model"] = self._current_model.copy()
            self._current_iteration += 1
            if self.verbose:
                logger.info("Time taken for iteration %s: %.2f seconds.", self._current_iteration, time.time()-start_time_iteration)
                logger.info("----------End: Iteration.----------")

        if self.verbose:
            logger.info("----------Start: Finalising inversion.----------")

        if self._current_iteration not in self._tracking_dict:
            self._tracking_dict[self._current_iteration] = {}
        #* Save the final models
        self._tracking_dict[self._current_iteration]["models"] = self._current_model.copy()

        #* Calculate the final data misfit
        iwjacobian_data_list, iwrhs_data_list, data_misfit_list, iwresponse_data_list, chi_squared_list = \
            self.get_individual_inversion_jacobians_and_rhs()
        weighted_jacobian_data, rhs_data, data_misfit, data_misfit_list = \
            self.get_full_petrophysical_jacobian_and_rhs(
                weighted_jacobian_list=iwjacobian_data_list,
                weighted_response_list=iwresponse_data_list
            )
        self._tracking_dict[self._current_iteration]["data_misfit"] = data_misfit_list
        self._tracking_dict[self._current_iteration]["chi_squared"] = chi_squared_list

        #* Calculate the final model regularisation misfit
        iwjacobian_regularisation_list, iwrhs_regularisation_list, model_misfit_list = \
            self.get_individual_model_regularisation_jacobian_and_rhs()
        self._tracking_dict[self._current_iteration]["single_model_regularisation_misfit"] = model_misfit

        #* Calculate the final XG misfit
        if self._inversion_settings["add_xg_for_untrusted_region"]:
            xg_jacobian_misfit = self._xg_regularisation.get_phi(
                physics_and_data=self.data,
                model_info_list=self._current_model.get_individual_geo_model_instances(),
                weighted=True,
            )
            self._tracking_dict[self._current_iteration]["dual_model_regularisation_misfit"] = np.linalg.norm(xg_jacobian_misfit)

        #* Save the final petro trust region
        self._tracking_dict[self._current_iteration]["petrophysical_trust_region"] = self.petrophysical_trust_region.copy()
        self._tracking_dict[self._current_iteration]["final_model"] = self._current_model.copy()

        #* Adjust maximum iterations
        if self._current_iteration == self._maximum_iterations:
            logger.info("Maximum number of iterations reached.")
        else:
            logger.info("Finished Gauss-Newton inversion after %s iterations. Overwriting maximum iterations.", self._current_iteration)
            self._maximum_iterations = self._current_iteration
        logger.info("----------End: Finalising inversion.----------")
        logger.info("Gauss-Newton inversion finished.")

    def get_individual_inversion_jacobians_and_rhs(self):
        """
        Compute individual (per-method) weighted jacobians and RHS vectors.

        Returns
        -------
        weighted_jacobian_list : list
            List of sparse jacobian matrices (one per dataset) in inversion domain.
        individual_rhs_data_list : list
            List of RHS vectors corresponding to each jacobian.
        data_misfit_list : list
            L2 norms of data misfits per dataset.
        weighted_response_list : list
            Modelled (predicted) weighted responses for each dataset.
        chi_squared_list : list
            Chi-squared metrics per dataset.
        """
        # Initiate weighted jacobian and response
        weighted_jacobian_list = []
        weighted_response_list = []
        geophysical_models = self.current_model.get_individual_geo_model_instances()
        for model_no, model in enumerate(geophysical_models):
            data_for_model = self.data[model_no]
            weighted_jacobian, weighted_response =data_for_model.get_jacobian_and_response(
                model=model,
                domain="default",
            )
            transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(method_number=model_no)
            weighted_jacobian_inversion = weighted_jacobian.multiply(
                transformation_vector
            )
            weighted_jacobian_list.append(weighted_jacobian_inversion)
            weighted_response_list.append(weighted_response)

        # Assemble the full jacobian matrix
        weighted_jacobian_list = [
            weighted_jacobian.tocsr() for weighted_jacobian in weighted_jacobian_list
        ]

        # Create a weighted observed data vector
        weighted_observed_data_list = [
            self.data[i].weight * self.data[i]._data_observed
            for i in range(self._number_of_datasets)
        ]

        individual_weighted_residual_vectors = [
            weighted_observed_data - weighted_response
            for weighted_observed_data, weighted_response in zip(weighted_observed_data_list, weighted_response_list)
        ]

        # Calculate the data misfit
        data_misfit_list = [
            np.linalg.norm(wobs-wres)
            for wobs, wres in zip(weighted_observed_data_list, weighted_response_list)
        ]

        # Calculate chi squared list
        err_list = [
            data.get_err() for data in self.data
        ]

        data_weight_list = [d.weight for d in self.data]

        chi_squared_list = [
            (w_obs.size)**-1 * np.sum(((w_obs-w_resp)/(err*w_obs))**2) for w_obs, w_resp, err, weight in zip(weighted_observed_data_list, weighted_response_list, err_list, data_weight_list)
        ]

        # Create rhs data
        if self.scheme == "creeping":
            individual_rhs_data_list = individual_weighted_residual_vectors
        elif self.scheme == "jumping":
            # Assemble transformed_model_vector
            individual_rhs_data_list = []
            for i in range(self._number_of_datasets):
                transformed_model_vector = self._current_model[i].transformed_model[self.region_of_interest].copy()
                individual_rhs_data_list.append(
                    individual_weighted_residual_vectors[i] + weighted_jacobian_list[i] @ transformed_model_vector
                )
        else:
            raise ValueError("Invalid scheme provided.")
        return weighted_jacobian_list, individual_rhs_data_list, data_misfit_list, weighted_response_list, chi_squared_list

    def get_full_petrophysical_jacobian_and_rhs(self, weighted_jacobian_list=None, weighted_response_list=None):
            """
            Assemble full petrophysical jacobian and RHS from individual components.

            Parameters
            ----------
            weighted_jacobian_list : list, optional
                Precomputed individual jacobians (default: None -> recompute).
            weighted_response_list : list, optional
                Precomputed individual responses (default: None -> recompute).

            Returns
            -------
            jacobian_full : scipy.sparse matrix
                Full assembled jacobian for the petrophysical inversion.
            rhs_data : ndarray
                Right-hand side for the inversion system.
            data_misfit : float
                Global data misfit (L2 norm).
            data_misfit_list : list
                Per-dataset misfit values.
            """
            if weighted_jacobian_list is None:
                assert weighted_response_list is None, "Weighted response list must be None if weighted jacobian list is None."

            if weighted_response_list is None:
                assert weighted_jacobian_list is None, "Weighted jacobian list must be None if weighted response list is None."

            if weighted_jacobian_list is None:
                weighted_jacobian_list, _, _, weighted_response_list, _ = self.get_individual_inversion_jacobians_and_rhs()

            jacobian_full = self.individual_matrices_rhs_to_full_matrices_rhs(
                individual_matrices=weighted_jacobian_list,
            )

            # Create a weighted observed data vector
            weighted_observed_data_list = [
                self.data[i].weight * self.data[i]._data_observed
                for i in range(self._number_of_datasets)
            ]

            weighted_response = np.concatenate(weighted_response_list)
            weighted_observed_data_vector = np.concatenate(weighted_observed_data_list)
            weighted_residual_vector = weighted_observed_data_vector - weighted_response

            # Calculate the data misfit
            data_misfit_list = [
                np.linalg.norm(wobs-wres)
                for wobs, wres in zip(weighted_observed_data_list, weighted_response_list)
            ]
            data_misfit = np.linalg.norm(weighted_residual_vector)

            # Create rhs data
            if self.scheme == "creeping":
                rhs_data = weighted_residual_vector
            elif self.scheme == "jumping":
                # Assemble transformed_model_vector
                transformed_model_vector = self._current_model[0].transformed_model[self.region_of_interest[self.petrophysical_trust_region]]
                transformed_model_vector = np.concatenate(
                    [
                        transformed_model_vector,
                        self._current_model[0].transformed_model[self.region_of_interest[~self.petrophysical_trust_region]],
                        self._current_model[1].transformed_model[self.region_of_interest[~self.petrophysical_trust_region]],
                    ]
                )
                rhs_data = weighted_residual_vector + jacobian_full @\
                    transformed_model_vector
            else:
                raise ValueError("Invalid scheme provided.")
            return jacobian_full, rhs_data, data_misfit, data_misfit_list

    def get_individual_model_regularisation_jacobian_and_rhs(self):
        """
        Compute regularisation jacobians and RHS per geophysical model.

        Returns
        -------
        jacobian_list : list or None
            List of jacobian matrices for each model, or None if no regularisation.
        rhs_list : list or None
            List of RHS vectors for each model, or None if no regularisation.
        model_regularisation_misfit_list : list
            Misfit contributions from each regularisation term.
        """
        if self._single_model_regularisation == []:
            return None, None, None
        
        # Collect all the jacobians and rhs
        jacobian_list = []
        rhs_list = []
        model_regularisation_misfit_list = []

        #* Get inversion domain jacobian and rhs for each model
        geophysical_models = self.current_model.get_individual_geo_model_instances()
        for model_no, model in enumerate(geophysical_models):
            data_weight = self.data[model_no].weight

            jacobian_list_model = []
            rhs_list_model = []
            model_regularisation_misfit_model = []

            for reg_no, reg in enumerate(self._single_model_regularisation[model_no]):
                if self.verbose:
                    logger.info("Calculating regularisation jacobian and rhs for model %s, regularisation %s.", model_no, reg_no)
                #* Get individual balancing factor for regularisation
                jacobian = reg.get_jacobian(
                    physics_and_data=self.data,
                    model_info=model,
                    domain="default",
                    ) * data_weight
                
                transformation_vector = self.current_model.get_transformed_model_gradient_from_geo(method_number=model_no)
                weighted_jacobian_inversion = jacobian.multiply(
                    transformation_vector
                )
                jacobian_list_model.append(weighted_jacobian_inversion)

                phi = reg.get_phi(
                    physics_and_data=self.data,
                    model_info=model,
                    weighted=True,
                    ) * data_weight
                model_regularisation_misfit_model.append(phi)

                if self.scheme == "creeping":
                    rhs = reg.get_rhs_creeping(
                        physics_and_data=self.data,
                        model_info=model
                        ) * data_weight
                elif self.scheme == "jumping":
                    rhs = reg.get_rhs_jumping(
                        physics_and_data=self.data,
                        model_info=model,
                        domain="inversion"
                        ) * data_weight
                else:
                    raise ValueError("Invalid scheme provided.")
                rhs_list_model.append(rhs)

            #* Finalise the model regularisation for the model
            jacobian = sP.sparse.vstack(blocks=jacobian_list_model, format="csr")
            rhs = np.concatenate(rhs_list_model, axis=0).flatten()

            #* Remove coupled rows if necessary [coupling between petro and non petro region] / [coupling between two regions]
            if self._inversion_settings["decouple_regularisation_trustregion"]:
                if model_no==0 and self.verbose:
                    logger.info("Decoupling regularisation between trusted and untrusted regions.")
                jacobian, rows_to_keep = self.remove_rows_coupling_trusted_untrusted(jacobian)

                #* Remove the rows from the rhs
                rhs = rhs[rows_to_keep]
                model_regularisation_misfit_model = np.concatenate(model_regularisation_misfit_model)[rows_to_keep]

                #* Remove coupling between regions if necessary
                jacobian, rhs, rows_to_keep = self.remove_coupling_from_regions(
                    matrix=jacobian,
                    rhs=rhs
                )
                #* Remove the rows from the model regularisation misfit
                model_regularisation_misfit_model = model_regularisation_misfit_model[rows_to_keep]

                #* Calculate the model regularisation misfit
                model_regularisation_misfit_model = np.linalg.norm(model_regularisation_misfit_model)
            else:
                model_regularisation_misfit_model = np.linalg.norm(np.concatenate(model_regularisation_misfit_model))

            jacobian_list.append(jacobian.copy())
            rhs_list.append(rhs.copy())
            model_regularisation_misfit_list.append(model_regularisation_misfit_model)
        return jacobian_list, rhs_list, model_regularisation_misfit_list

    def get_full_model_regularisation_jacobian_and_rhs(self, weighted_jacobian_list=None, weighted_rhs_list=None, model_regularisation_misfit_list=None):
        """
        Assemble full regularisation jacobian and RHS from individual regularisation blocks.

        Parameters
        ----------
        weighted_jacobian_list : list, optional
            Individual regularisation jacobians.
        weighted_rhs_list : list, optional
            Individual regularisation RHS vectors.
        model_regularisation_misfit_list : list, optional
            Per-model misfit list (used to compute total misfit).

        Returns
        -------
        regularisation_jacobian : scipy.sparse matrix or None
        regularisation_rhs : ndarray or None
        model_regularisation_misfit : float
        """
        if weighted_jacobian_list is None:
            assert weighted_rhs_list is None, "Individual rhs must be None if individual matrices are None."
            assert model_regularisation_misfit_list is None, "Model regularisation misfit list must be None if individual matrices are None."
        if weighted_rhs_list is None:
            assert weighted_jacobian_list is None, "Individual matrices must be None if individual rhs is None."
            assert model_regularisation_misfit_list is None, "Model regularisation misfit list must be None if individual rhs is None."
        if model_regularisation_misfit_list is None:
            assert weighted_jacobian_list is None, "Individual matrices must be None if model regularisation misfit list is None."
            assert weighted_rhs_list is None, "Individual rhs must be None if model regularisation misfit list is None."
        
        if weighted_jacobian_list is None:
            weighted_jacobian_list, weighted_rhs_list, model_regularisation_misfit_list = self.get_individual_model_regularisation_jacobian_and_rhs()

        regularisation_jacobian, regularisation_rhs = self.individual_matrices_rhs_to_full_matrices_rhs(
            individual_matrices=weighted_jacobian_list,
            individual_rhs=weighted_rhs_list
        )

        # Calculate the model regularisation misfit
        model_regularisation_misfit = np.linalg.norm(model_regularisation_misfit_list)

        return regularisation_jacobian, regularisation_rhs, model_regularisation_misfit

    def individual_matrices_rhs_to_full_matrices_rhs(self, individual_matrices, individual_rhs=None):
        """
        Convert a list of per-method matrices and RHS into the block-structured
        full matrix used for the decoupled petrophysical inversion.

        Parameters
        ----------
        individual_matrices : list
            List of sparse matrices (one per geophysical method) in inversion domain.
        individual_rhs : list, optional
            Corresponding RHS vectors. If None, only the matrix is returned.

        Returns
        -------
        full_matrices : scipy.sparse matrix
            Block-assembled full matrix.
        full_rhs : ndarray, optional
            Concatenated RHS vector if `individual_rhs` was provided.
        """
        #* Part 1 - common part
        full_matrices_trusted = sP.sparse.vstack(
            blocks=[
                individual_matrices[:,self.petrophysical_trust_region[self.region_of_interest]]
                for individual_matrices in individual_matrices
                ],
            format="csr"
        )
        #* Part 2 - model specific part
        full_matrices_untrusted = sP.sparse.block_diag(
            mats =[
                individual_matrices[:,~self.petrophysical_trust_region[self.region_of_interest]]
                for individual_matrices in individual_matrices],
            format="csr"
        )
        #* Assemble the full matrix
        full_matrices = sP.sparse.hstack(
            blocks=[full_matrices_trusted, full_matrices_untrusted],
            format="csr"
        )

        if individual_rhs is None:
            return full_matrices
        else:
            full_rhs = np.concatenate(individual_rhs)
            return full_matrices, full_rhs

    def remove_rows_coupling_trusted_untrusted(self, matrix, mode="individual"):
        """
        Remove rows that couple trusted and untrusted petrophysical regions.

        Parameters
        ----------
        matrix : scipy.sparse matrix
            Matrix to process.
        mode : {'composite','individual'}
            Behaviour for detecting coupled rows.

        Returns
        -------
        matrix : scipy.sparse matrix
            Matrix with coupled/zero rows removed.
        rows_to_keep : ndarray
            Indices of rows kept from the original matrix.
        """
        assert mode in ["composite", "individual"], "Invalid mode provided."
        number_of_trusted_cells = np.sum(self.petrophysical_trust_region)
        zeros_rows_index = [
            row
            for row in range(matrix.shape[0])
            if np.all(matrix[row,:].toarray() == 0)
        ]

        if mode=="composite":
            coupled_row_indices = [
                row
                for row in range(matrix.shape[0])
                if np.any(np.logical_and(np.any(matrix[row,:number_of_trusted_cells].toarray() != 0), np.any(matrix[row, number_of_trusted_cells:].toarray()!= 0)))
            ]
        elif mode=="individual":
                coupled_row_indices = [
                row
                for row in range(matrix.shape[0])
                if np.any(matrix[row,self.petrophysical_trust_region[self.region_of_interest]].toarray() != 0) and np.any(matrix[row,~self.petrophysical_trust_region[self.region_of_interest]].toarray()!= 0)
            ]
        else:
            raise ValueError("Invalid mode provided.")
        n_prior_rows = matrix.shape[0]
        rows_to_remove = np.concatenate([zeros_rows_index, coupled_row_indices])
        rows_to_keep = np.setdiff1d(np.arange(matrix.shape[0]), rows_to_remove)
        matrix = matrix[rows_to_keep,:]
        if self.verbose:
            logger.info(
                "Prior rows to removal: %s Remaining rows: %s Zero rows: %s Coupled rows: %s",
                n_prior_rows, matrix.shape[0], len(zeros_rows_index), len(coupled_row_indices)
            )
        return matrix, rows_to_keep

    def remove_coupling_from_regions(self, matrix, rhs, xg=False):
        """
        Remove rows that represent coupling between specified decoupling regions.

        Parameters
        ----------
        matrix : scipy.sparse matrix
        rhs : ndarray
        xg : bool
            If True, treat XG (cross-gradient) indexing differently.

        Returns
        -------
        matrix, rhs, rows_to_keep
        """
        rows_to_keep = np.arange(matrix.shape[0])
        if self._decouple_regularisation_regions is not None:
            rows_to_remove = []
            index_matrix = matrix.toarray().copy()
            if xg:
                decoupled_vec = np.tile(self._decouple_regularisation_regions[0][self.region_of_interest], (1,2))
                index_matrix = (np.abs(index_matrix)>0) * decoupled_vec
            else:
                index_matrix = (np.abs(index_matrix)>0) * self._decouple_regularisation_regions[0][self.region_of_interest]
            for decoupled_pair in self._decouple_regularisation_regions[1]:
                coupled_row_indices_temp = [
                    row
                    for row in range(matrix.shape[0])
                    if np.any(index_matrix[row] == decoupled_pair[0]) and np.any(index_matrix[row] == decoupled_pair[1])
                ]
                rows_to_remove.extend(coupled_row_indices_temp)
            rows_to_keep = np.setdiff1d(np.arange(matrix.shape[0]), rows_to_remove)
            logger.info("Removing %s coupled rows [region].", len(rows_to_remove))
            matrix = matrix[rows_to_keep]
            rhs = rhs[rows_to_keep]
        return matrix, rhs, rows_to_keep

    def get_individual_model_updates_wo_xg(self, iwjacobian_data_list, iwrhs_data_list, iwjacobian_regularisation_list, iwrhs_regularisation_list, clip_update=False):
        """
        Compute per-model petrophysical updates without using cross-gradient (XG).

        Parameters
        ----------
        iwjacobian_data_list, iwrhs_data_list : list
            Individual data jacobians and RHS lists.
        iwjacobian_regularisation_list, iwrhs_regularisation_list : list
            Individual regularisation jacobians and RHS lists.
        clip_update : bool
            If True, apply clipping to the per-model update.

        Returns
        -------
        individual_model_update_list : list
            List of full-size petrophysical update vectors (one per model).
        """
        individual_model_update_list = []
        for model_no, _ in enumerate(self._current_model.model_list_geo):
            #* Get the rhs and jacobian for the single model
            #* Assembly the full jacobian
            jacobian_single = sP.sparse.vstack(
                blocks=[
                iwjacobian_data_list[model_no],
                iwjacobian_regularisation_list[model_no],
                ],
                format="csr"
            )
            #* Assemble the full rhs
            rhs_single = np.concatenate([iwrhs_data_list[model_no], iwrhs_regularisation_list[model_no]])

            #* Scale the system
            update_vector_small_single = self.solve_linear_system(
                A=jacobian_single,
                b=rhs_single,
                enable_scaling=True
            )

            #* Temporary update the model to get update in petrophysical domain
            petrophysical_model_old = self.current_model.model_petro.copy()
            transformed_model_petro, _ = self.current_model.transformed_model

            petrophysical_model_new = self.current_model.inversion_transformation_petro.backward(
                transformed_model_petro + update_vector_small_single[self.petrophysical_trust_region[self.region_of_interest]],
            )
            
            petrophysical_model_update = petrophysical_model_new - petrophysical_model_old[self.region_of_interest[self.petrophysical_trust_region]]

            if clip_update:
                petrophysical_model_update = self.clip_individual_model(
                    model_update=petrophysical_model_update,
                    bounds=self._maximum_update_per_step[0],
                )

            big_petro_model_update = np.zeros(self._mesh_info.mesh.cellCount())
            big_petro_model_update[
                np.logical_and(
                    self.region_of_interest,
                    self.petrophysical_trust_region
                )
            ] = petrophysical_model_update
            #* Add calculated update to list
            individual_model_update_list.append(big_petro_model_update.copy())
        return individual_model_update_list

    def get_individual_model_updates_xg(self, iwjacobian_data_list, iwrhs_data_list, iwjacobian_regularisation_list, iwrhs_regularisation_list, clip_update=False):
        """
        Compute per-model petrophysical updates including cross-gradient terms.

        This assembles a block-diagonal system plus the XG coupling rows, solves
        for the combined update and splits the solution into per-model updates.
        """
        individual_model_update_list = []
        jacobian_list = []
        rhs_list = []
        for model_no, model in enumerate(self.current_model.model_list_geo):
            #* Get the rhs and jacobian for the single model
            #* Assembly the full jacobian
            jacobian_single = sP.sparse.vstack(
                blocks=[
                iwjacobian_data_list[model_no][:, self.region_of_interest],
                iwjacobian_regularisation_list[model_no][:, self.region_of_interest],
                ],
                format="csr"
            )
            jacobian_list.append(jacobian_single)
            #* Assemble the full rhs
            rhs_single = np.concatenate([iwrhs_data_list[model_no], iwrhs_regularisation_list[model_no]])
            rhs_list.append(rhs_single)

        #* Add cross gradient terms
        xg_jacobian = self._xg_reference_regularisation.get_jacobian(
            physics_and_data=self.data,
            model_info_list=self.current_model.get_individual_geo_model_instances(),
            domain="default",
        )
        transformation_gradient_list = [
            self.current_model.get_transformed_model_gradient_from_geo(
                method_number=model_no,
            ) for model_no in range(self._number_of_datasets)
        ]
        transformation_gradient_list = np.concatenate(transformation_gradient_list)
        xg_jacobian = xg_jacobian.multiply(transformation_gradient_list).tocsr()

        xg_rhs = self._xg_reference_regularisation.get_rhs_creeping(
            physics_and_data=self.data,
            model_info_list=self.current_model.get_individual_geo_model_instances(),
        )

        #* Decouple the xg jacobian
        xg_jacobian = xg_jacobian.tocsr()
        xg_jacobian, rows_to_keep = self.remove_rows_coupling_trusted_untrusted(xg_jacobian)

        #* Remove the rows from the rhs
        xg_rhs = xg_rhs[rows_to_keep]

        #* Remove coupling between regions if necessary
        xg_jacobian, xg_rhs, rows_to_keep = self.remove_coupling_from_regions(
            matrix=xg_jacobian,
            rhs=xg_rhs,
            xg=True
        )

        #* Assemble the full system
        full_jacobian = sP.sparse.vstack(
            blocks = [
                sP.sparse.block_diag(
                    mats = jacobian_list,
                    format="csr",
                ),
                xg_jacobian,
            ],
            format="csr"
        )

        self._tracking_dict[self._current_iteration]["ixg_jacobian"] = xg_jacobian.copy()
        self._tracking_dict[self._current_iteration]["ixg_rhs"] = xg_rhs.copy()

        full_rhs = np.concatenate(rhs_list + [xg_rhs])

        #* Scale the system
        update_vector = self.solve_linear_system(
            A=full_jacobian,
            b=full_rhs,
            enable_scaling=True
        )

        model_size = np.sum(self._region_of_interest)
        for model_no, model in enumerate(self.current_model.model_list_geo):
            petrophysical_model_old = self.current_model.model_petro.copy()
            transformed_model_petro, _ = self.current_model.transformed_model

            #* Temporary update the model to get update in petrophysical domain
            update_vector_small_single = update_vector[model_no*model_size:(model_no+1)*model_size]
            petrophysical_model_new = self.current_model.inversion_transformation_petro.backward(
                transformed_model_petro + update_vector_small_single[self.petrophysical_trust_region[self.region_of_interest]],
            )
            
            petrophysical_model_update = petrophysical_model_new - petrophysical_model_old[self.region_of_interest[self.petrophysical_trust_region]]

            if clip_update:
                petrophysical_model_update = self.clip_individual_model(
                    model_update=petrophysical_model_update,
                    bounds=self._maximum_update_per_step[0],
                )

            big_petro_model_update = np.zeros(self._mesh_info.mesh.cellCount())
            big_petro_model_update[
                np.logical_and(
                    self.region_of_interest,
                    self.petrophysical_trust_region
                )
            ] = petrophysical_model_update
            #* Add calculated update to list
            individual_model_update_list.append(big_petro_model_update.copy())
        return individual_model_update_list

    def get_model_updates_from_full_inversion_results(self, model_update_vector, force_disable_clipping=False):
        """
        Convert the full inversion update vector into petrophysical and geophysical
        model updates and optionally clip them.

        Parameters
        ----------
        model_update_vector : ndarray
            Full solution vector from the linearised inversion.
        force_disable_clipping : bool
            If True, skip clipping of updates.

        Returns
        -------
        model_update_tuple : tuple
            (petrophysical_update, list_of_geo_updates)
        new_model_tuple : tuple
            New model values after applying updates.
        """
        model_update_list = []
        #* To use setter mechanic, build full update vectors
        model_size = self._region_of_interest.size

        trust_and_roi = np.logical_and(self.petrophysical_trust_region, self._region_of_interest)
        untrust_and_roi = np.logical_and(~self.petrophysical_trust_region, self._region_of_interest)

        size_of_trust_and_roi = np.sum(trust_and_roi)
        size_of_untrust_and_roi = np.sum(untrust_and_roi)

        transformed_model_petro, transformed_models_geo = self.current_model.transformed_model

        #* Calculate petrophysical model update
        petro_model_update = np.zeros_like(self.current_model.model_petro)
        new_petro_model = self.current_model.model_petro.copy()

        new_petro_model_on_roi = self.current_model.inversion_transformation_petro.backward(
            transformed_model_petro + model_update_vector[:size_of_trust_and_roi]
        )
        new_petro_model[self._region_of_interest[self.petrophysical_trust_region]] = new_petro_model_on_roi

        petro_model_update = new_petro_model - self.current_model.model_petro

        #* Calculate model updates geo
        model_update_list_geo = []
        for model_no, small_model_geo in enumerate(self.current_model.model_list_geo_small):
            #* Get the model update in geo domain
            geo_model_update = np.zeros_like(small_model_geo)
            new_geo_model_small = small_model_geo.copy()

            transformed_model_update = model_update_vector[
                size_of_trust_and_roi + model_no * size_of_untrust_and_roi:
                size_of_trust_and_roi + (model_no + 1) * size_of_untrust_and_roi
            ]
            new_geo_model_on_roi = self.current_model.inversion_transformation_list_geo[model_no].backward(
                transformed_models_geo[model_no] + transformed_model_update
            )

            new_geo_model_small[self._region_of_interest[~self.petrophysical_trust_region]] = new_geo_model_on_roi

            geo_model_update = new_geo_model_small - small_model_geo
            model_update_list_geo.append(geo_model_update)

        model_update_tuple = (petro_model_update, model_update_list_geo)

        if not force_disable_clipping:
            model_update_tuple = self.clip_model_updates(
                model_update_tuple
            )

        petro_model_update, model_update_list_geo = model_update_tuple
        new_model_tuple = (
            self.current_model.model_petro + petro_model_update,
            [
                small_model_geo + geo_model_update
                for small_model_geo, geo_model_update in zip(self.current_model.model_list_geo_small, model_update_list_geo)
            ]
        )
        return model_update_tuple, new_model_tuple

    def apply_model_updates(self, new_model_tuple):
        """
        Apply model updates using the current model's `set_model` API.

        Parameters
        ----------
        new_model_tuple : tuple
            Tuple containing the updated petrophysical model and geo models.
        """
        self.current_model.set_model(new_model_tuple)
        return None

    def clip_model_updates(self, model_update_tuple):
        """
        Clip petrophysical and geophysical update vectors element-wise according
        to `self._maximum_update_per_step` per model.

        Returns
        -------
        update_tuple_clipped : tuple
            Clipped updates in the same structure as `model_update_tuple`.
        """
        #* Update the petro model
        petro_model_update = model_update_tuple[0]
        if petro_model_update.size:
            logger.info("Clipping petrophysical model.")
            petro_model_update_clipped = self.clip_individual_model(
                model_update=petro_model_update,
                bounds=self._maximum_update_per_step[0]
            )
        else:
            petro_model_update_clipped = petro_model_update.copy()

        #* Update the geo models
        geo_model_update_list = model_update_tuple[1]
        geo_model_update_list_clipped = []
        for model_no, geo_model_update in enumerate(geo_model_update_list):
            if geo_model_update.size:
                logger.info("Clipping geo model %s.", model_no)
                geo_model_update_clipped = self.clip_individual_model(
                    model_update=geo_model_update,
                    bounds=self._maximum_update_per_step[model_no+1]
                )
            else:
                geo_model_update_clipped = geo_model_update.copy()
            geo_model_update_list_clipped.append(geo_model_update_clipped)
        
        #* Assemble the clipped model update tuple
        update_tuple_clipped = (
            petro_model_update_clipped,
            geo_model_update_list_clipped
        )
        return update_tuple_clipped

    def clip_individual_model(self, model_update, bounds):
        """
        Clip a single model update vector to provided `bounds` and optionally
        print diagnostics when `self.verbose` is True.

        Parameters
        ----------
        model_update : ndarray
        bounds : tuple
            (min, max) bounds for clipping.

        Returns
        -------
        update_vector_clipped : ndarray
        """
        if self.verbose and model_update.size:
            minimum_update_cell = np.argmin(model_update)
            maximum_update_cell = np.argmax(model_update)
            logger.info(
                "Updates before clipping: Size: %.2e. Minimum: %.2e. Maximum: %.2e. Median: %.2e.",
                np.linalg.norm(model_update), model_update[minimum_update_cell], model_update[maximum_update_cell], np.median(model_update)
            )

        #* Decide if clipping is required
        clipping_required = False
        update_too_small_vector = model_update < bounds[0]
        update_too_big_vector = model_update > bounds[1]

        if any(update_too_small_vector):
            clipping_required = True
        if any(update_too_big_vector):
            clipping_required = True
        if self.verbose and clipping_required:
            logger.info("Too small: #%s, too big: #%s. Clipping required.", np.sum(update_too_small_vector), np.sum(update_too_big_vector))
        if not clipping_required and self.verbose:
            logger.info("No clipping required.")

        #* Clip the update
        if clipping_required:
            update_vector_clipped = np.clip(model_update, bounds[0], bounds[1])
            if self.verbose and update_vector_clipped.size:
                minimum_update_cell = np.argmin(update_vector_clipped)
                maximum_update_cell = np.argmax(update_vector_clipped)
                logger.info(
                    "Updates after clipping: Size: %.2e. Minimum: %.2e at cell %s. Maximum: %.2e at cell %s. Median: %.2e.",
                    np.linalg.norm(update_vector_clipped), update_vector_clipped[minimum_update_cell], minimum_update_cell, update_vector_clipped[maximum_update_cell], maximum_update_cell, np.median(update_vector_clipped)
                )
        else:
            update_vector_clipped = model_update
        return update_vector_clipped

    def connected_components_from_meshinfo_petrophysical_trust_region(self, new_petrophysical_trust_region, region="trusted"):
        """
        Compute connected components of either the trusted or untrusted subgraph
        of the mesh defined by `new_petro_trust_region`.

        Returns
        -------
        connected_components : list of ndarrays
            List of node indices for each connected component.
        labels : ndarray
            Label per node (-1 for excluded nodes).
        adjacency_matrix : scipy.sparse matrix
            Sparse adjacency matrix used for the graph search.
        """
        assert region in ["trusted", "untrusted"], "Region must be either 'trusted' or 'untrusted'."
        #* Build the adjacency matrix of the mesh
        cni_list = self.mesh_info.cell_neighbour_info
        row_indices = [np.repeat(i, len(cni.neighbour_cells)) for i, cni in enumerate(cni_list)]
        col_indices = [cni.neighbour_cells for cni in cni_list]
        row_indices = np.concatenate(row_indices)
        col_indices = np.concatenate(col_indices)

        if region == "untrusted":
            trusted_or_untrusted = np.logical_and(~new_petrophysical_trust_region[row_indices], ~new_petrophysical_trust_region[col_indices])
        elif region == "trusted":
            trusted_or_untrusted = np.logical_and(new_petrophysical_trust_region[row_indices], new_petrophysical_trust_region[col_indices])

        row_indices = row_indices[trusted_or_untrusted]
        col_indices = col_indices[trusted_or_untrusted]
        data = np.ones_like(row_indices)

        adjacency_matrix = sP.sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh_info.mesh.cellCount(), self.mesh_info.mesh.cellCount())
        )

        #* Get the connected components of the untrusted region
        connected_components, labels = sP.sparse.csgraph.connected_components(
            adjacency_matrix,
            directed=False,
            return_labels=True,
            connection="weak")

        #* Remove trusted cells from labels - these are unconnected nodes
        if region == "untrusted":
            labels[new_petrophysical_trust_region] = -1
        elif region == "trusted":
            labels[~new_petrophysical_trust_region] = -1
        # indices_of_trusted_cells = np.where(self.petrophysical_trust_region)[0]
        # labels[indices_of_trusted_cells] = -1

        #* Remove unconnected nodes from connected components
        label_list = np.unique(labels)
        connected_components = [np.where(labels == label)[0] for label in label_list if label != -1]

        return connected_components, labels, adjacency_matrix

    def remove_small_components_from_petrophysical_trust_region(self, new_petrophysical_trust_region, min_size, criterion="area"):
        """
        Remove small components from the petrophysical trust region.
        
        Parameters:
        - new_petrophysical_trust_region: np.ndarray, the petrophysical trust region.
        - min_size: int, the minimum size of a component to keep.
        - criterion: str, the criterion to use for removing small components. Options are "count" or "area".
        
        Returns:
        - new_petrophysical_trust_region: np.ndarray, the new petrophysical trust region.
        """
        connected_components_untrusted, _, _ = self.connected_components_from_meshinfo_petrophysical_trust_region(
            new_petrophysical_trust_region=new_petrophysical_trust_region,
            region="untrusted"
            )

        old_petrophysical_trust_region = self.petrophysical_trust_region.copy()
        n_cells_to_untrusted_old = np.sum(~new_petrophysical_trust_region) - np.sum(~self.petrophysical_trust_region)

        #* Remove too small untrusted components
        for component in connected_components_untrusted:
            if criterion == "count":
                # Use the number of cells in the component
                size = len(component)
            elif criterion == "area":
                # Use the area of the component
                size = np.sum([self.mesh_info.cell_neighbour_info[comp].cell_area for comp in component])
                if self.verbose:
                    logger.info("Component has %s cells with area %.2f.", len(component), size)
            else:
                raise ValueError("Criterion must be either 'count' or 'area'.")

            # Check if the size meets the minimum size requirement
            #* Set a component to trusted, if its size is too small and they have been trusted before
            if size < min_size and np.all(self.petrophysical_trust_region[component]):
                new_petrophysical_trust_region[component] = True

        n_cells_to_untrusted_new = np.sum(~new_petrophysical_trust_region) - np.sum(~self.petrophysical_trust_region)
        if self.verbose:
            logger.info(
                "Rejected changing cells to untrusted for %s cells from the petrophysical trust region [untrusted too small]. Remaining cells: %s.",
                np.sum(n_cells_to_untrusted_old) - np.sum(n_cells_to_untrusted_new), np.sum(new_petrophysical_trust_region)
            )

        connected_components_trusted, _, _ = self.connected_components_from_meshinfo_petrophysical_trust_region(
            new_petrophysical_trust_region=new_petrophysical_trust_region,
            region="trusted"
            )

        n_cells_trusted_old = np.sum(new_petrophysical_trust_region)
        #* Remove too small trusted components
        for component in connected_components_trusted:
            if criterion == "count":
                # Use the number of cells in the component
                size = len(component)
            elif criterion == "area":
                # Use the area of the component
                size = np.sum([self.mesh_info.cell_neighbour_info[comp].cell_area for comp in component])
                if self.verbose:
                    logger.info("Component has %s cells with area %.2f.", len(component), size)
            else:
                raise ValueError("Criterion must be either 'count' or 'area'.")
            
            #* Set a component to untrusted, if its size is too small (trusted hole in a untrusted region)
            if size < min_size:
                new_petrophysical_trust_region[component] = False

        n_cells_trusted_new = np.sum(new_petrophysical_trust_region)
        if self.verbose:
            logger.info(
                "Rejected keeping cells trusted for %s cells from the petrophysical trust region [trusted too small]. Remaining cells: %s.",
                n_cells_trusted_old - n_cells_trusted_new, np.sum(new_petrophysical_trust_region)
            )
        if self.verbose:
            logger.info(
                "Removed %s cells from the petrophysical trust region [trusted too small]. Remaining cells: %s.",
                np.sum(old_petrophysical_trust_region) - np.sum(new_petrophysical_trust_region), np.sum(new_petrophysical_trust_region)
            )

        self.petrophysical_trust_region = new_petrophysical_trust_region.copy()
        return

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
        ax.set_ylim(0, 1.1 * max(max(data_misfit)))
        return fig, ax

    def plot_model_regularisation_misfit_history(self, ax=None, figsize=(10, 6)):
        """ Plots the single model regularisation misfit history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
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
        else:
            fig = ax.get_figure()

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
            include_initial_models=False,
            include_geophysical_models=False,
            figsize=None,
            **kwargs
            ):
        """ Plot the current model."""
        number_of_axes = self._number_of_datasets
        if include_initial_models:
            initial_models = self.initial_models
        if include_geophysical_models:
            geophysical_models = [
                self.data.get_geophysical_model(self._current_model[i], i)
                for i in range(self._number_of_datasets)
                ]

        figlist = []
        axslist = []

        figsize = (5 * number_of_axes, 5) if figsize is None else figsize

        # Plot the initial models
        if include_initial_models:
            fig, axs = plt.subplots(1, number_of_axes, figsize=figsize)
            fig.suptitle("Initial Petrophysical Models")
            for i in range(self._number_of_datasets):
                ax = axs[i]
                initial_models[i].plot_model(ax=ax, **kwargs)
                ax.set_title(f"Initial Model {i+1}")
            figlist.append(fig)
            axslist.append(axs)

        # Plot the current model
        fig, axs = plt.subplots(1, number_of_axes, figsize=figsize)
        fig.suptitle("Current Petrophysical Models")
        for i in range(self._number_of_datasets):
            ax = axs[i]
            self.current_model[i].plot_model(ax=ax, **kwargs)
            ax.set_title("Current Petrophysical model")
        figlist.append(fig)
        axslist.append(axs)

        # Plot the geophysical models
        data_set = 0
        if include_geophysical_models:
            fig, axs = plt.subplots(1, number_of_axes, figsize=figsize)
            fig.suptitle("Geophysical Models")
            for i in range(self._number_of_datasets):
                ax= axs[i]
                model_temp = self.current_model[0].copy()
                model_temp.model = geophysical_models[i]
                model_temp.plot_model(ax=ax, **kwargs)
                ax.set_title(f"Geophysical Model {data_set+1}")
                data_set += 1
            figlist.append(fig)
            axslist.append(axs)
        return figlist, axslist

    def assemble_iteration_vector_from_tracking_dict(self, key):
        """ Assembles a vector with iteration numbers and a vector with the corresponding values."""
        iteration_vector, value_vector = [], []

        for iteration, iteration_dict in self._tracking_dict.items():
            if key in iteration_dict and isinstance(iteration, (int, float)):
                iteration_vector.append(iteration)
                value_vector.append(iteration_dict[key])
        return iteration_vector, value_vector

    def show_region_of_interest(self, markersize=1, marker="o", mode="triang"):
        """ Shows the region of interest for all models."""
        fig, ax = plt.subplots(1, figsize=(5, 5))
        mesh_info = self.mesh_info
        mesh_info.show_region_of_interest(ax=ax, markersize=markersize, marker=marker, mode=mode)
        return fig, ax
    
    def show_petro_trust_region(self, ax=None, figsize=(5,5), markersize=1, marker="o", mode="triang"):
        """ Shows the petrophysical trust region for all models."""
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig = ax.get_figure()
        mesh_info = self.mesh_info
        region_of_interest_save = mesh_info.region_of_interest.copy()
        mesh_info._region_of_interest = self.petro_trust_region
        mesh_info.show_region_of_interest(ax=ax, markersize=markersize, marker=marker, mode=mode)
        ax.set_title("Petrophysical Trust Region")
        mesh_info._region_of_interest = region_of_interest_save.copy()
        return fig, ax


def update_petro_from_diverging_model_updates(list_models, list_model_updates, current_petrophysical_trust_region, level=0.1, difference="relative", only_shrink_trust_region=False):
    """ Updates the petrophysical model from diverging model updates. The criterion for the divergence is that the
    individual model update are diverging and significant.

    Model updates are considered to be significant if they are larger than a certain percentage/absolute value of the model.
    Model updates are considered to be diverging if they are in the same direction.
    
    Parameters
    ----------
    list_models : list
        List of models.
    list_model_updates : list
        List of model updates in the petrophysical domain.
    current_petrophysical_trust_region : np.ndarray
        Current petrophysical trust region.
    percentage : float, optional
        Percentage of the model update to be considered as diverging.
    only_shrink_petrophysical_trust_region : bool, optional
        If True, only shrinks the petrophysical trust region.

    Returns
    -------
    new_petro_trust_region : np.ndarray
        New petrophysical trust region.
    diverging : np.ndarray
        Diverging model updates.
    significant : np.ndarray
        Significant model updates.
    """
    assert len(list_models) == len(list_model_updates), "Number of models and model updates must be the same."
    assert len(list_models) == 2, "Only two models are supported."
    assert difference in ["relative", "absolute"], "Difference must be either relative or absolute."
    if isinstance(level, float):
        level = np.array([level, level])
    
    #* Check if the model updates are diverging
    diverging = ~(np.sign(list_model_updates[0]) == np.sign(list_model_updates[1]))

    #* Check if the model updates are significant - model updates are significant, if at least one of the model updates is large [relative or absolute]

    if difference == "relative":
        significant_model_1 = np.abs(list_model_updates[0]) > level[0] * np.abs(np.array(list_models[0].model))
        significant_model_2 = np.abs(list_model_updates[1]) > level[1] * np.abs(np.array(list_models[1].model))
    elif difference == "absolute":
        significant_model_1 = np.abs(list_model_updates[0]) > level[0]
        significant_model_2 = np.abs(list_model_updates[1]) > level[1]
    else:
        raise ValueError("Invalid difference provided.")

    significant = significant_model_1 | significant_model_2
    
    if not np.any(significant):
        logger.info("No model updates are significant.")
        logger.info("Maximum model update 1: %s.", np.max(np.abs(list_model_updates[0])))
        logger.info("Maximum model update 2: %s.", np.max(np.abs(list_model_updates[1])))

    #*
    distrusted_cell_from_update = np.logical_and(
        diverging,
        significant
    )

    #* If the model updates are diverging and significant, update the trust region
    new_petro_trust_region = current_petrophysical_trust_region.copy()
    if only_shrink_trust_region:
        #* If only enhance 
        new_petro_trust_region = np.logical_and(
            current_petrophysical_trust_region,
            ~distrusted_cell_from_update
        )
    else:
        new_petro_trust_region = ~distrusted_cell_from_update
    return new_petro_trust_region, diverging, significant

def update_petro_from_large_difference_in_model_updates(list_models, list_model_updates, current_petrophysical_trust_region, level=0.1, difference="relative", only_shrink_petrophysical_trust_region=True):
    """ Updates the petrophysical model from diverging model updates. The criterion for the divergence is that the
    individual model update are diverging and significant.

    Model updates are considered to be significant if they are larger than a certain percentage of the model.
    Model updates are considered to be diverging if they are in the same direction.
    
    Parameters
    ----------
    list_models : list
        List of models.
    list_model_updates : list
        List of model updates in the petrophysical domain.
    current_petrophysical_trust_region : np.ndarray
        Current petrophysical trust region.
    percentage : float, optional
        Percentage of the model update to be considered as diverging.
    only_shrink_petrophysical_trust_region : bool, optional
        If True, only shrinks the petrophysical trust region.

    Returns
    -------
    new_petro_trust_region : np.ndarray
        New petrophysical trust region.
    diverging : np.ndarray
        Diverging model updates.
    significant : np.ndarray
        Significant model updates.
    """
    assert len(list_models) == len(list_model_updates), "Number of models and model updates must be the same."
    assert len(list_models) == 2, "Only two models are supported."
    assert difference in ["relative", "absolute"], "Difference must be either relative or absolute."
    assert isinstance(level, (int, float)), "Level must be a float or an int."
    assert level > 0, "Level must be greater than 0."
    
    #* Check if the model updates are significant - model updates are significant, if at least one of the model updates is large [relative or absolute]
    #* Relative is only relevant in regions in the petro trust region if only shrink trust region is True
    if difference == "relative":
        significant_model_1 = np.abs(list_model_updates[0]) > level * np.abs(np.array(list_models[0]))
        significant_model_2 = np.abs(list_model_updates[1]) > level * np.abs(np.array(list_models[1]))
        significant = significant_model_1 | significant_model_2
    elif difference == "absolute":
        significant = np.abs(list_model_updates[1]-list_model_updates[0]) > level
    else:
        raise ValueError("Invalid difference provided.")
    
    if not np.any(significant):
        logger.info("No model updates are significant.")
        logger.info("Maximum model update 1: %s.", np.max(np.abs(list_model_updates[0])))
        logger.info("Maximum model update 2: %s.", np.max(np.abs(list_model_updates[1])))

    distrusted_cell_from_update = significant

    #* If the model updates are diverging and significant, update the trust region
    new_petro_trust_region = current_petrophysical_trust_region.copy()
    if only_shrink_petrophysical_trust_region:
        #* If only enhance 
        new_petro_trust_region = np.logical_and(
            current_petrophysical_trust_region,
            ~distrusted_cell_from_update
        )
    else:
        new_petro_trust_region = ~distrusted_cell_from_update
    return new_petro_trust_region, None, significant

def update_petro_from_difference_in_updates(list_models, list_model_updates, current_petrophysical_trust_region, level=0.1, difference="relative", only_shrink_petrophysical_trust_region=True):
    """ Updates the petrophysical model from diverging model updates. The criterion for the divergence is that the
    individual model update are diverging and significant.

    Model updates are considered to be significant if they are larger than a certain percentage of the model.
    Model updates are considered to be diverging if they are in the same direction.
    
    Parameters
    ----------
    list_models : list
        List of models.
    list_model_updates : list
        List of model updates in the petrophysical domain.
    current_petrophysical_trust_region : np.ndarray
        Current petrophysical trust region.
    percentage : float, optional
        Percentage of the model update to be considered as diverging.
    only_shrink_petrophysical_trust_region : bool, optional
        If True, only shrinks the petrophysical trust region.

    Returns
    -------
    new_petro_trust_region : np.ndarray
        New petrophysical trust region.
    diverging : np.ndarray
        Diverging model updates.
    significant : np.ndarray
        Significant model updates.
    """
    assert len(list_models) == len(list_model_updates), "Number of models and model updates must be the same."
    assert len(list_models) == 2, "Only two models are supported."
    assert difference in ["relative", "absolute"], "Difference must be either relative or absolute."

    #* Check if the model updates are diverging
    diverging = ~(np.sign(list_model_updates[0]) == np.sign(list_model_updates[1]))

    #* Check if the model updates are significant - model updates are significant, if at least one of the model updates is large [relative or absolute]
    if difference == "relative":
        significant_model_1 = np.abs(list_model_updates[0]) > level[0] * np.abs(np.array(list_models[0].model))
        significant_model_2 = np.abs(list_model_updates[1]) > level[1] * np.abs(np.array(list_models[1].model))
        significant = np.logical_or(significant_model_1, significant_model_2)
    elif difference == "absolute":
        significant = np.abs(list_model_updates[1]-list_model_updates[0]) > level[0]
    else:
        raise ValueError("Invalid difference provided.")

    #*
    distrusted_cell_from_update = np.logical_and(
        diverging,
        significant
    )

    #* If the model updates are diverging and significant, update the trust region
    new_petro_trust_region = current_petrophysical_trust_region.copy()
    if only_shrink_petrophysical_trust_region:
        #* If only enhance 
        new_petro_trust_region = np.logical_and(
            current_petrophysical_trust_region,
            ~distrusted_cell_from_update
        )
    else:
        new_petro_trust_region = ~distrusted_cell_from_update
    return new_petro_trust_region, diverging, significant
