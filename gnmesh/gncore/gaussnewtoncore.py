"""
Gauss-Newton core utilities.

This module provides the shared core API used by Gauss-Newton inversion
managers. It implements the `GaussNewtonCore` class which encapsulates
configuration, solver selection (CPU/GPU), scaling utilities and helper
methods for solving the linear systems that arise during inversion.

The module is intentionally implementation-focused and should not perform
I/O. Logging is performed via the standard library ``logging`` module.

Notes
-----
The implementation supports optional CuPy acceleration when available;
when CuPy is not present the code falls back to SciPy/NumPy CPU solvers.
"""

from typing import Any

import time
import numpy as np
import scipy as sP
import logging

logger = logging.getLogger(__name__)

# Optional GPU support (cupy). If unavailable, fall back to CPU solvers.
try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.sparse.linalg as cpxla
    _HAVE_CUPY = True
except Exception:
    cp = None
    cpx = None
    cpxla = None
    _HAVE_CUPY = False


class GaussNewtonCore:
    """Shared read-only and configurable API for Gauss-Newton managers."""

    _allowed_schemes = ("creeping", "jumping")
    _allowed_scalings = ("column_sum_l1", "column_sum_l2", "column_max", "none")
    _allowed_num_solvers = ("scipy_sparse", "scipy_dense", "cupy_sparse", "cupy_dense")
    _maximum_update_per_step_mode = "single"

    _mesh_info: Any
    _region_of_interest: Any
    _data: Any
    _maximum_iterations: Any
    _current_iteration: Any
    _data_misfit_history: Any
    _model_history: Any
    _tracking_dict: Any
    _save_model_history: Any
    _scheme: Any
    _scaling: Any
    _num_solver: Any
    _terminate_on_chi2_decrease: Any
    _maximum_update_per_step: Any
    _number_of_models: Any

    @property
    def mesh_info(self):
        """Returns the mesh info object."""
        return self._mesh_info

    @property
    def region_of_interest(self):
        """Returns the region of interest."""
        return self._region_of_interest

    @property
    def data(self):
        """Returns the inversion data object."""
        return self._data

    @property
    def maximum_iterations(self):
        """Returns the maximum number of iterations."""
        return self._maximum_iterations

    @property
    def current_iteration(self):
        """Returns the current iteration."""
        return self._current_iteration

    @property
    def data_misfit_history(self):
        """Returns the data misfit history."""
        return self._data_misfit_history

    @property
    def model_history(self):
        """Returns the model history."""
        return self._model_history

    @property
    def tracking_dict(self):
        """Returns the tracking dictionary."""
        return self._tracking_dict

    @property
    def decouple_regularisation(self):
        """Returns the decoupling regularisation configuration."""
        return self._decouple_regularisation

    @decouple_regularisation.setter
    def decouple_regularisation(self, value):
        """Sets the decoupling regularisation configuration."""
        if value is None:
            self._decouple_regularisation = None
            return

        assert isinstance(value, (list, tuple)), "Decouple regularisation must be a list or tuple."
        assert len(value) == 2, "Decouple regularisation must have two elements."
        assert isinstance(value[0], np.ndarray), "Decouple regularisation must contain a numpy array as first element."
        if hasattr(self, "_mesh_info") and hasattr(self._mesh_info, "mesh"):
            assert value[0].shape[0] == self._mesh_info.mesh.cellCount(), (
                "Decouple regularisation must have the same length as the number of cells."
            )
        assert isinstance(value[1], list), "Decouple regularisation must contain a list of pair definitions."
        assert all(isinstance(reg, (list, tuple)) for reg in value[1]), (
            "Decouple regularisation must contain a list of lists or tuples."
        )
        self._decouple_regularisation = value

    @property
    def save_model_history(self):
        """Returns whether model history is saved."""
        return self._save_model_history

    @save_model_history.setter
    def save_model_history(self, value):
        """Sets whether model history is saved."""
        assert isinstance(value, bool), "save_model_history must be a boolean."
        self._save_model_history = value

    @property
    def scheme(self):
        """Returns the update scheme."""
        return self._scheme

    @scheme.setter
    def scheme(self, value):
        """Sets the update scheme."""
        assert value in self._allowed_schemes, (
            "Scheme must be one of " + ", ".join(self._allowed_schemes) + "."
        )
        self._scheme = value

    @property
    def scaling(self):
        """Returns the scaling mode."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        """Sets the scaling mode."""
        assert value in self._allowed_scalings, (
            "Scaling must be one of " + ", ".join(self._allowed_scalings) + "."
        )
        self._scaling = value

    @property
    def maximum_update_per_step(self):
        """Returns the maximum update per step."""
        return self._maximum_update_per_step

    @maximum_update_per_step.setter
    def maximum_update_per_step(self, value):
        """Sets the maximum update per step."""
        if self._maximum_update_per_step_mode == "per_model":
            if getattr(self, "_number_of_models", 1) == 1:
                assert isinstance(value, (list, tuple)), (
                    "Maximum update per step must be a tuple or list of tuple."
                )
                if isinstance(value, tuple):
                    value = [value]
            else:
                assert isinstance(value, list), (
                    "Maximum update per step must be a list or tuple."
                )

            assert len(value) == self._number_of_models, (
                "Maximum update per step must have the same length as the number of models."
            )
            for val in value:
                assert isinstance(val, tuple), "Maximum update per step must be a tuple."
                assert len(val) == 2, "Maximum update per step must have two values."
                assert all(isinstance(v, (int, float)) for v in val), (
                    "Maximum update per step must be a list of integers or floats."
                )
            self._maximum_update_per_step = value
            return

        assert isinstance(value, (list, tuple)), (
            "Maximum update per step must be a list or tuple."
        )
        assert len(value) == 2, "Maximum update per step must have two values."
        assert all(isinstance(val, (int, float)) for val in value), (
            "Maximum update per step must be a list of integers or floats."
        )
        self._maximum_update_per_step = value

    @property
    def num_solver(self):
        """Returns the numerical solver."""
        return self._num_solver

    @num_solver.setter
    def num_solver(self, value):
        """Sets the numerical solver."""
        assert value in self._allowed_num_solvers, (
            "Numerical solver must be one of "
            + ", ".join(self._allowed_num_solvers)
            + "."
        )
        self._num_solver = value

    @property
    def terminate_on_chi2_decrease(self):
        """Returns the chi^2 decrease termination threshold."""
        return self._terminate_on_chi2_decrease

    @terminate_on_chi2_decrease.setter
    def terminate_on_chi2_decrease(self, value):
        """Sets the chi^2 decrease termination threshold."""
        assert isinstance(value, float), "Termination criterion must be a float."
        assert value >= 0, "Termination criterion must be positive."
        assert value < 1, "Termination criterion must be smaller than 1."
        self._terminate_on_chi2_decrease = value

    def solve_linear_system(self, A, b, enable_scaling=True):
        """Solves Ax = b using the configured solver and scaling.

        This central implementation supports scipy and cupy backends. If cupy
        is requested but not available, it falls back to the CPU solver and
        emits a warning when `verbose` is True.
        """
        # Prepare scaling vector
        if enable_scaling:
            if isinstance(self.scaling, str):
                if self.scaling == "column_sum_l1":
                    absolute_column_sum = sP.sparse.linalg.norm(A, axis=0, ord=1)
                    scaling_vector = 1 / (absolute_column_sum + 1e-6)
                    scaling_vector = np.array(scaling_vector).squeeze()
                elif self.scaling == "column_max":
                    scaling_vector = 1 / (sP.sparse.linalg.norm(A, axis=0, ord=np.inf) + 1e-10)
                elif self.scaling == "column_sum_l2":
                    col_norms = np.asarray(A.power(2).sum(axis=0)).squeeze()
                    scaling_vector = 1 / (np.sqrt(col_norms) + 1e-10)
                else:
                    scaling_vector = np.ones(A.shape[1])
            elif isinstance(self.scaling, (list, tuple, np.ndarray)):
                assert len(self.scaling) == A.shape[1], (
                    "Scaling vector must have the same length as the number of columns in the Jacobian."
                )
                scaling_vector = np.array(self.scaling).squeeze()
            else:
                raise ValueError("Invalid scaling provided. Must be 'column_sum_l1', 'column_max' or a vector.")
            A = A.multiply(scaling_vector)
        else:
            scaling_vector = np.ones(A.shape[1])

        start_time = time.time()

        solver = self.num_solver
        # Handle cupy availability fallback
        if solver.startswith("cupy") and not _HAVE_CUPY:
            if self.verbose:
                logger.warning("Requested solver '%s' but cupy is unavailable — falling back to scipy.", solver)
            solver = "scipy_sparse" if solver == "cupy_sparse" else "scipy_dense"

        # Solve with chosen backend
        if solver == "scipy_sparse":
            result = sP.sparse.linalg.lsmr(A, b, maxiter=1e4)
            residual = result[3] / np.linalg.norm(b)
            x = result[0]
            reason = (result[1], result[2])
            condition_number = result[6]

        elif solver == "scipy_dense":
            A_dense = A.toarray()
            result = sP.linalg.lstsq(A_dense, b)
            if result[1].size == 0:
                residual = 0.0
            else:
                residual = result[1] / np.linalg.norm(b)
            x = result[0]
            reason = (None, None)
            condition_number = result[3][0] / result[3][-1]

        elif solver == "cupy_sparse":
            sparse_matrix_cp = cp.sparse.csr_matrix(A)
            b_cp = cp.array(b)
            result = cpxla.lsmr(sparse_matrix_cp, b_cp, maxiter=1e4)
            residual = result[3] / cp.linalg.norm(b_cp)
            x = cp.asnumpy(result[0])
            reason = (result[1], result[2])
            condition_number = result[6]
            # Free cupy memory pools
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        elif solver == "cupy_dense":
            A_dense_cp = cp.array(A.toarray())
            b_cp = cp.array(b)
            result = cp.linalg.lstsq(A_dense_cp, b_cp)
            if result[1].size == 0:
                residual = 0.0
            else:
                # cp returns array-like residuals
                residual = result[1][0] / cp.linalg.norm(b_cp)
            x = cp.asnumpy(result[0])
            reason = (None, None)
            condition_number = result[3][0] / result[3][-1]
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()

        else:
            raise ValueError("Invalid numerical solver provided.")

        if getattr(self, "verbose", False):
            try:
                logger.info("Solve on iteration: %s.", self.current_iteration+1)
            except Exception:
                logger.info("Solve completed.")
            try:
                logger.info("Reason for termination: %s after %s iterations.", reason[0], reason[1])
            except Exception:
                pass
            try:
                logger.info("Condition number: %.2e.", condition_number)
            except Exception:
                pass
            try:
                logger.info("Relative residual: %.2e.", residual)
            except Exception:
                pass
            logger.info("Time taken to solve the system: %.2f seconds.", time.time()-start_time)

        x = x * scaling_vector

        if self.verbose:
            logger.info("Residual of the solution: %.2e.", np.linalg.norm(A @ x - b) / np.linalg.norm(b))
        return x

    def clip_model_vector(self, model_vector_update, model_no: int = 0):
        """Clip an update vector according to `self._maximum_update_per_step`.

        Accepts an optional `model_no` used when per-model limits are configured.
        """
        if getattr(self, "verbose", False):
            try:
                model_update_vector_on_roi = model_vector_update[self._region_of_interest]
                minimum_update_cell = np.argmin(model_vector_update)
                maximum_update_cell = np.argmax(model_vector_update)
                logger.info(
                    "Updates before clipping: Size: %.2e. Minimum: %.2e at cell %s. Maximum: %.2e at cell %s. Median: %.2e.",
                    np.linalg.norm(model_update_vector_on_roi), model_vector_update[minimum_update_cell], minimum_update_cell, model_vector_update[maximum_update_cell], maximum_update_cell, np.median(model_update_vector_on_roi)
                )
            except Exception:
                pass

        # Determine clipping bounds depending on mode
        if self._maximum_update_per_step_mode == "per_model":
            bounds = self._maximum_update_per_step[model_no]
        else:
            # single shared bounds
            bounds = self._maximum_update_per_step

        lower, upper = bounds

        clipping_required = False
        update_too_small_vector = model_vector_update < lower
        update_too_big_vector = model_vector_update > upper
        if any(update_too_small_vector) or any(update_too_big_vector):
            clipping_required = True

        if getattr(self, "verbose", False) and clipping_required:
            logger.info("Too small: #%s, too big: #%s. Clipping required.", np.sum(update_too_small_vector), np.sum(update_too_big_vector))

        if clipping_required:
            update_vector_clipped = np.clip(model_vector_update, lower, upper)
            if getattr(self, "verbose", False):
                try:
                    model_update_vector_on_roi = update_vector_clipped[self._region_of_interest]
                    minimum_update_cell = np.argmin(update_vector_clipped)
                    maximum_update_cell = np.argmax(update_vector_clipped)
                    logger.info(
                        "Updates after clipping: Size: %.2e. Minimum: %.2e at cell %s. Maximum: %.2e at cell %s. Median: %.2e.",
                        np.linalg.norm(model_update_vector_on_roi), model_update_vector_on_roi[minimum_update_cell], minimum_update_cell, model_update_vector_on_roi[maximum_update_cell], maximum_update_cell, np.median(model_update_vector_on_roi)
                    )
                except Exception:
                    pass
        else:
            update_vector_clipped = model_vector_update
        return update_vector_clipped

    def remove_rows_coupling_trusted_untrusted(self, matrix, rhs, region_of_interest):
        """Remove coupled rows according to `self._decouple_regularisation`.

        Returns (matrix, rhs) with coupled rows removed when decoupling is active.
        """
        if getattr(self, "_decouple_regularisation", None) is not None:
            rows_to_remove = []
            if not isinstance(matrix, np.ndarray):
                index_matrix = matrix.toarray().copy()
            else:
                index_matrix = matrix.copy()
            mask = self._decouple_regularisation[0][region_of_interest]
            try:
                index_matrix = (np.abs(index_matrix) > 0) * np.tile(mask, reps=(1, getattr(self, "_number_of_models", 1))).squeeze()
            except Exception:
                index_matrix = (np.abs(index_matrix) > 0) * mask
            for decoupled_pair in self._decouple_regularisation[1]:
                coupled_row_indices_temp = [
                    row
                    for row in range(matrix.shape[0])
                    if np.any(index_matrix[row] == decoupled_pair[0]) and np.any(index_matrix[row] == decoupled_pair[1])
                ]
                rows_to_remove.extend(coupled_row_indices_temp)
            rows_to_keep = np.setdiff1d(np.arange(matrix.shape[0]), rows_to_remove)
            if getattr(self, "verbose", False):
                logger.info("Removing %s coupled rows.", len(rows_to_remove))
            matrix = matrix[rows_to_keep]
            rhs = rhs[rows_to_keep]
        return matrix, rhs

    def percent_decrease_in_chi2(self, iteration):
        """Returns the percent decrease in chi-squared between iterations.

        Subclasses can call `self.percent_decrease_in_chi2(iteration)` to get
        a list with the relative decreases per dataset.
        """
        assert iteration > 0, "Iteration must be greater than 0."
        chi2_list_it_act = self._tracking_dict[iteration]["chi_squared"]
        chi2_list_it_prev = self._tracking_dict[iteration-1]["chi_squared"]
        chi2_list_decrease = [
            (chi2_it_prev - chi2_it_act) / chi2_it_prev
            for chi2_it_prev, chi2_it_act in zip(chi2_list_it_prev, chi2_list_it_act)
        ]
        return chi2_list_decrease

    def assemble_iteration_vector_from_tracking_dict(self, key):
        """Assembles iteration numbers and corresponding values from tracking dict.

        Returns (iteration_vector, value_vector) where each element corresponds
        to entries in `self._tracking_dict` that contain `key` and are
        numeric iterations.
        """
        iteration_vector, value_vector = [], []
        for iteration, iteration_dict in self._tracking_dict.items():
            if key in iteration_dict and isinstance(iteration, (int, float)):
                iteration_vector.append(iteration)
                value_vector.append(iteration_dict[key])
        return iteration_vector, value_vector