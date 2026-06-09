"""
Model containers and plotting helpers for gradients, Hessians, and transformations.

The module is split into two classes:

* ModelInfo: Single-model container with gradient, Hessian, Laplacian, and plotting helpers.
* ModelInfoMixedGeoPetro: Coupled petrophysical/geophysical model container.

No module-level functions are defined.

Author: Hagen Söding
Affiliation: ETH Zürich
Email: hagen.soeding@eaps.ethz.ch
"""

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import logging
from . import spatialgradient as sG
from . import transformation as tF
from pygimli.viewer.mpl import drawModel, drawMeshBoundaries

logger = logging.getLogger(__name__)

class ModelInfo:
    """
    Class for storing model information and calculating spatial gradients.

    Parameters
    ----------
    model : np.ndarray
        The model for which to calculate the spatial gradient.

    mesh_info : MeshInfo
        The mesh information object containing the mesh and the cell neighbour information.

    taylor_order : int, optional
        The order of the Taylor expansion to use for the gradient calculation.
        Can be 1 or 2. The default is 1.

    transformation : Transformation, optional
        The transformation to apply to the model. The default is None.

    plotting_transformation : Transformation, optional
        The transformation to apply to the model for plotting. The default is None.

    Attributes
    ----------
    mesh_info : MeshInfo
        The mesh information object containing the mesh and the cell neighbour information.

    model : np.ndarray
        The model for which to calculate the spatial gradient.

    transformed_model : np.ndarray
        The transformed model.
    
    transformed_model_gradient : np.ndarray
        The gradient of the transformed model.

    spatial_gradient : np.ndarray
        The spatial gradient of the model.

    sensitivities : np.ndarray
        The sensitivities of the model.

    Methods
    -------
    plot_model(ax=None, figsize=(5,5), show_mesh=True, **kwargs)
        Plot the model.

    plot_transformed_model(ax=None, figsize=(5,5), show_mesh=True, **kwargs)
        Plot the transformed model.

    plot_spatial_gradient_field(ax=None, figsize=(5,5), show_mesh=True, scale=1e3, **kwargs)
        Plot the spatial gradient field.

    plot_absolute_value_of_spatial_gradient(ax=None, figsize=(5,5), show_mesh=True, **kwargs)
        Plot the absolute value of the spatial gradient field.

    copy()
        Copy the model information object.
    """
    def __init__(self, model, mesh_info, taylor_order=1, transformation=tF.MultiplicativeTransformation(1), plotting_transformation=tF.MultiplicativeTransformation(1)):
        if transformation is None:
            logger.info("No transformation is passed - assuming identity - set transformation if needed")
        self._transformation = transformation
        self._mesh_info = mesh_info
        self._region_of_interest = self._mesh_info.region_of_interest
        self._taylor_order = taylor_order

        if self._taylor_order==1:
            assert self._mesh_info._gn_taylor_1_set_successfully, f"TMesh not succesfully prepared for taylor order {self._taylor_order}"
        elif self._taylor_order==2:
            assert self._mesh_info._gn_taylor_2_set_successfully, f"Mesh not successfully prepared for taylor order {self._taylor_order}"

        self._transformed_model = None
        self._plotting_transformation = plotting_transformation
        self.model = model

    @property
    def mesh_info(self):
        """Mesh information property"""
        return self._mesh_info

    @property
    def model(self):
        """Model property. If transformation is set,
        the model is transformed and the transformed model is stored"""
        return self._model

    @model.setter
    def model(self, model):
        # Set model property
        self._model = model

        # Set transformed model, if transformation is given
        if self._transformation is not None:
            self._transformed_model = self._transformation.forward(self._model[self._region_of_interest])
            self._transformed_model_gradient = self._transformation.derivative_backward(
                self._transformed_model
            )

        # Set spatial gradient
        self._spatial_gradient = sG.calculate_spatial_gradient(
            model=self.model,
            mesh_info=self._mesh_info,
            taylor_order=self._taylor_order
        )

        # Set Hessian matrix, if taylor order >= 2 is enabled
        if self._taylor_order>=2:
            self._hessian_matrix = sG.calculate_hessian_matrix(
                model=self._model,
                mesh_info=self._mesh_info,
                taylor_order=self._taylor_order,
            )
        else:
            self._hessian_matrix = None

        # Set Laplacian, if taylor order >= 2 is enabled
        if self._taylor_order>=2:
            self._laplacian = sG.calculate_laplacian_from_hessian_matrix_model(
                hessian_matrix_model=self._hessian_matrix
            )

        # Model is updated, so the sensitivities are not up to date
        self._sensitivities_updated_since_last_model_update = False

    @property
    def transformed_model(self):
        """Transformed model property. If transformation is set,
        the model is transformed and the untransformed model is stored"""
        return self._transformed_model

    @transformed_model.setter
    def transformed_model(self, transformed_model):
        if not self._transformation is None:
            composite_vector = self._model.copy()
            composite_vector[self._region_of_interest] = self._transformation.backward(transformed_model)
            self.model = composite_vector

    @property
    def transformed_model_gradient(self):
        """Transformed model gradient property. If transformation is set,
        the model is transformed and the transformed model gradient is stored"""
        return self._transformed_model_gradient

    @property
    def spatial_gradient(self):
        """Spatial gradient property."""
        return self._spatial_gradient

    @property
    def hessian_matrix(self):
        return self._hessian_matrix

    @property
    def laplacian(self):
        return self._laplacian

    @property
    def sensitivities(self):
        """Sensitivities property. If sensitivities are not calculated, it will return None.
        The sensitivities are always with respect to the untransformed parameters."""
        if hasattr(self, "_sensitivities"):
            if not self._sensitivities_updated_since_last_model_update:
                logger.warning("WARNING - sensitivities have not been calculated since the last model \
                      update and may be faulty")
            if self._sensitivities is None:
                logger.warning("WARNING - sensitivities are None")
            return self._sensitivities

        logger.info("Sensitivities not yet set - set it first")
        return

    @sensitivities.setter
    def sensitivities(self, manager):
        manager.fop.createJacobian(self._model)
        sensitivities = manager.fop.jacobian()
        if isinstance(sensitivities, pg.matrix.RSparseMapMatrix):
            self._sensitivities = pg.utils.sparseMatrix2Dense(sensitivities)
        else:
            self._sensitivities = np.array(sensitivities)
        self._sensitivities_updated_since_last_model_update = True

    @property
    def taylor_order(self):
        """Taylor order property"""
        return self._taylor_order

    def plot_model(self, ax=None, figsize=(5,5), show_mesh=True, disable_plotting_transform=False, **kwargs):
        """Plot model"""
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
        else:
            fig = ax.get_figure()
        
        if self._plotting_transformation is not None:
            model_to_plot = self._plotting_transformation.forward(self._model)
        else:
            model_to_plot = self._model
        if disable_plotting_transform:
            model_to_plot = self._model
        _ = drawModel(ax, self._mesh_info.mesh, data=model_to_plot, **kwargs)

        if show_mesh:
            drawMeshBoundaries(
                ax,
                self._mesh_info.mesh,
                hideMesh=False,
                lw=0.3,
                color="0.2",
                fitView=False,
            )
        return fig, ax

    def plot_transformed_model(self, ax=None, figsize=(5,5), show_mesh=True, **kwargs):
        """Plot transformed model"""
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
        else:
            fig = ax.get_figure()
        _ = drawModel(ax, self._mesh_info.mesh, data=self._transformed_model, **kwargs)

        if show_mesh:
            drawMeshBoundaries(
                ax,
                self._mesh_info.mesh,
                hideMesh=False,
                lw=0.3,
                color="0.2",
                fitView=False,
            )
        return fig, ax

    def plot_spatial_gradient_field(
        self,
        ax=None,
        figsize=(5,5),
        show_mesh=True,
        scale=1e3,
        **kwargs
    ):
        """Plot spatial gradient field"""
        fig, ax = sG.plot_gradient_field(
            spatial_gradient=self._spatial_gradient,
            mesh=self._mesh_info.mesh,
            dimension=self._mesh_info.dimension,
            scale=scale,
            show_mesh=show_mesh,
            ax=ax,
            figsize=figsize,
            **kwargs
        )
        return fig, ax

    def plot_absolute_value_of_spatial_gradient(
        self,
        ax=None,
        figsize=(5,5),
        show_mesh=True,
        **kwargs
    ):
        """Plot absolute value of spatial gradient field"""
        fig, ax = sG.plot_absolute_value_of_gradient_field_from_vectors(
            spatial_gradient=self._spatial_gradient,
            mesh=self._mesh_info.mesh,
            show_mesh=show_mesh,
            ax=ax,
            figsize=figsize,
            **kwargs
        )
        return fig, ax

    def plot_laplacian_field(
        self,
        ax=None,
        figsize=(5,5),
        show_mesh=True,
        **kwargs
    ):
        """Plot Laplacian field"""
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
        else:
            fig = ax.get_figure()
        if self._taylor_order>=2:
            if "cMax" not in kwargs:
                kwargs["cMax"] = np.max(np.abs(self._laplacian))
                kwargs["cMin"] = -kwargs["cMax"]
            if "cMap" not in kwargs:
                kwargs["cMap"] = "seismic"
            gci = drawModel(ax, self._mesh_info.mesh, data=self._laplacian, **kwargs)
            if "cMap" in kwargs:
                gci.set_cmap(kwargs["cMap"])
            if "cmap" in kwargs:
                gci.set_cmap(kwargs["cmap"])
            if "cMin" in kwargs:
                gci.set_clim(kwargs["cMin"], None)
            if "cMax" in kwargs:
                gci.set_clim(None, kwargs["cMax"])
            if "cmin" in kwargs:
                gci.set_clim(kwargs["cmin"], None)
            if "cmax" in kwargs:
                gci.set_clim(None, kwargs["cmax"])
        else:
            raise ValueError("Laplacian is not calculated for taylor order 1")
        if show_mesh:
            drawMeshBoundaries(
                ax,
                self._mesh_info.mesh,
                hideMesh=False,
                lw=0.3,
                color="0.2",
                fitView=False,
            )
        return fig, ax
    
    def plot_hessian_matrix_overview(
        self,
        cmap="turbo",
        show_mesh=False,
        log=False,
        sensor_positions=None,
        figsize=(15, 10),
        **kwargs
        ):
        """Plot Hessian matrix overview"""
        assert self._taylor_order>=2, "Hessian matrix is not calculated for taylor order 1"
        fig, axs = sG.plot_hessian_matrix_overview(
            hessian_matrix_list=self._hessian_matrix,
            mesh=self._mesh_info.mesh,
            cmap=cmap,
            log=log,
            sensor_positions=sensor_positions,
            show_mesh=show_mesh,
            **kwargs
        )
        return fig, axs
    
    def copy(self):
        """Copy model. Mesh_info and transformation are passed as reference, 
        since we dont expect them to change."""
        # While the model is copied, mesh_info and transformation is passed as reference,
        # since we dont expect them to change.
        model_new = ModelInfo(
            model=self.model.copy(),
            mesh_info=self._mesh_info,
            taylor_order=self._taylor_order,
            transformation=self._transformation,
            plotting_transformation=self._plotting_transformation)
        # Preserve sensitivities when they are already available.
        if hasattr(self, "_sensitivities") and self.sensitivities is not None:
            model_new.sensitivities = self.sensitivities.copy()
        return model_new

class ModelInfoMixedGeoPetro:
    """Container for one petrophysical model and multiple coupled geophysical models."""

    def __init__(
            self,
            model_petro,
            model_list_geo,
            mesh_info,
            taylor_order=1,
            transformation_list_petro=None,
            inversion_transformation_petro=tF.MultiplicativeTransformation(1),
            inversion_transformation_list_geo=None,
            plotting_transformation=tF.MultiplicativeTransformation(1),
            petrophysical_trust_region=None
    ):

        if transformation_list_petro is None:
            transformation_list_petro = [tF.MultiplicativeTransformation(1) for _ in model_list_geo]
        self._transformation_list_petro = transformation_list_petro
        self._inversion_transformation_petro = inversion_transformation_petro

        if inversion_transformation_list_geo is None:
            inversion_transformation_list_geo = [tF.MultiplicativeTransformation(1) for _ in model_list_geo]
        self._inversion_transformation_list_geo = inversion_transformation_list_geo

        if len(inversion_transformation_list_geo) != len(model_list_geo):
            raise ValueError("Length of inversion_transformation_list_geo must match length of model_list_geo")

        self._mesh_info = mesh_info
        self._region_of_interest = self._mesh_info.region_of_interest
        self._taylor_order = taylor_order
        if self._taylor_order == 1:
            assert self._mesh_info._gn_taylor_1_set_successfully, f"Mesh not succesfully prepared for taylor order {self._taylor_order}"
        elif self._taylor_order == 2:
            assert self._mesh_info._gn_taylor_2_set_successfully, f"Mesh not successfully prepared for taylor order {self._taylor_order}"

        self._plotting_transformation = plotting_transformation

        if petrophysical_trust_region is not None:
            self._petrophysical_trust_region = petrophysical_trust_region
        else:
            self._petrophysical_trust_region = np.ones(self._mesh_info.mesh.cellCount(), dtype=bool)

        self.model = (model_petro, model_list_geo)

    @property
    def mesh_info(self):
        """Mesh information property"""
        return self._mesh_info

    def set_model(self, model_list_tuple):
        # Validate the input layout before storing any state.
        number_of_untrusted_cells = np.sum(~self._petrophysical_trust_region)

        assert len(model_list_tuple) == 2, "Model must be a tuple of (model_petro, model_list_geo)"
        model_petro, model_list_geo = model_list_tuple
        assert isinstance(model_petro, np.ndarray), "Model petro must be a numpy array"
        assert isinstance(model_list_geo, list), "Model list geo must be a list"
        assert len(model_list_geo) == len(self._transformation_list_petro), "Length of model_list_geo must match length of transformation_list_geo"

        #! Check for None
        length_of_geo_models = [len(model) for model in model_list_geo]
        assert all(length == number_of_untrusted_cells for length in length_of_geo_models), \
            "All geo models must have the same length as the number of untrusted cells"
        
        # Store the split model representation.

        self._model_petro = model_petro.copy()
        self._model_list_geo_small= [model.copy() for model in model_list_geo]

        # The transformed view is derived on demand from the stored split models.

    model = property(fset=set_model, doc="Model property. Returns a vector of the size of the full model with nans for the untrusted cells.")

    @property
    def model_petro(self):
        """Petrophysical model property. Model is restricted to the trust region."""
        return self._model_petro

    @property
    def model_list_geo(self):
        """Geophysical model list property. If transformation is set,
        the model is transformed and the transformed model is stored"""
        model_list_geo = []
        for i, model_geo in enumerate(self._model_list_geo_small):
            model_temp = np.ones(self._mesh_info.mesh.cellCount())
            model_temp[self._petrophysical_trust_region] = self._transformation_list_petro[i].forward(self.model_petro)
            model_temp[~self._petrophysical_trust_region] = model_geo
            model_list_geo.append(model_temp)
        return model_list_geo

    @property
    def model_list_geo_small(self):
        """Geophysical model list property. This is the model without the petrophysical transformation applied."""
        return self._model_list_geo_small

    @property
    def transformed_model(self):
        """Transformed model property. If transformation is set,
        the model is transformed and the untransformed model is stored"""
        
        transformed_petro_model = self._inversion_transformation_petro.forward(self._model_petro[self._region_of_interest[self._petrophysical_trust_region]])

        transformed_geo_model_list = []
        for i, model_geo in enumerate(self._model_list_geo_small):
            transformed_geo_model = self._inversion_transformation_list_geo[i].forward(model_geo[self._region_of_interest[~self._petrophysical_trust_region]])
            transformed_geo_model_list.append(transformed_geo_model)
        return (transformed_petro_model, transformed_geo_model_list)

    @property
    def transformed_model_gradient(self):
        """Transformed model gradient property. If transformation is set,
        the model is transformed and the transformed model gradient is stored"""
        transformed_model_petro, transformed_model_geo_list = self.transformed_model

        transformed_petro_model_gradient = self._inversion_transformation_petro.derivative_backward(
            transformed_model_petro
        )

        transformed_geo_model_gradient_list = []
        for i, transformed_model_geo in enumerate(transformed_model_geo_list):
            transformed_geo_model_gradient = self._inversion_transformation_list_geo[i].derivative_backward(transformed_model_geo)
            transformed_geo_model_gradient_list.append(transformed_geo_model_gradient)
        return (transformed_petro_model_gradient, transformed_geo_model_gradient_list)

    def get_transformed_model_gradient_from_geo(self, method_number):
        """Transformed model gradient property. If transformation is set,
        the model is transformed and the transformed model gradient is stored"""
        transformed_petro_model_gradient, transformed_geo_model_gradient_list = self.transformed_model_gradient
        # Chain-rule correction from geophysical to petrophysical parameterisation.
        petro_transformation_gradient = self._transformation_list_petro[method_number].derivative_forward(
            self._model_petro[self._region_of_interest[self._petrophysical_trust_region]]
        )
        double_transformed_petro_gradient = transformed_petro_model_gradient * petro_transformation_gradient
        return_vector = np.zeros(np.sum(self._region_of_interest), dtype=float)
        return_vector[self._petrophysical_trust_region[self._region_of_interest]] = double_transformed_petro_gradient
        return_vector[~self._petrophysical_trust_region[self._region_of_interest]] = transformed_geo_model_gradient_list[method_number]
        return return_vector

    @property
    def petrophysical_trust_region(self):
        """Petrophysical trust region property"""
        return self._petrophysical_trust_region

    @petrophysical_trust_region.setter
    def petrophysical_trust_region(self, petro_trust_region):
        """Set petrophysical trust region property"""
        if self.petrophysical_trust_region is not None:
            if len(petro_trust_region) != self._mesh_info.mesh.cellCount():
                raise ValueError("Length of petro_trust_region must match length of mesh")
            if not isinstance(petro_trust_region, np.ndarray):
                raise ValueError("petro_trust_region must be a numpy array")
            if np.any(np.logical_and(petro_trust_region, ~self._petrophysical_trust_region)):
                raise ValueError("petro_trust_region must not contain cells that have been distrusted before")

        # Re-split the current model into the new trust-region layout.
        model_list_geo = self.model_list_geo
        self._petrophysical_trust_region = petro_trust_region

        new_petro_model = self._transformation_list_petro[0].backward(model_list_geo[0][petro_trust_region])
        new_geo_model_list = []
        for i, model_geo in enumerate(model_list_geo):
            new_geo_model = model_geo[~petro_trust_region]
            new_geo_model_list.append(new_geo_model)

        self.model = (new_petro_model, new_geo_model_list)

    def get_individual_geo_model_instances(self):
        """Get individual model instances for petrophysical and geophysical models."""
        model_list = []
        for i, model_geo in enumerate(self.model_list_geo):
            model_temp = ModelInfo(
                model=model_geo.copy(),
                mesh_info=self._mesh_info,
                taylor_order=self._taylor_order,
                transformation= self._inversion_transformation_list_geo[i],
            )
            model_list.append(model_temp)
        return model_list
    
    @property
    def inversion_transformation_petro(self):
        """Inversion transformation for petrophysical model"""
        return self._inversion_transformation_petro
    
    @property
    def inversion_transformation_list_geo(self):
        """Inversion transformation for geophysical models"""
        return self._inversion_transformation_list_geo

    def copy(self):
        """Copy model. Mesh_info and transformation are passed as reference, 
        since we dont expect them to change."""
        # While the model is copied, mesh_info and transformation is passed as reference,
        # since we dont expect them to change.
        model_new = ModelInfoMixedGeoPetro(
            model_petro=self.model_petro.copy(),
            model_list_geo=[model.copy() for model in self._model_list_geo_small],
            mesh_info=self._mesh_info,
            taylor_order=self._taylor_order,
            transformation_list_petro=self._transformation_list_petro,
            inversion_transformation_petro=self._inversion_transformation_petro,
            inversion_transformation_list_geo=self._inversion_transformation_list_geo,
            plotting_transformation=self._plotting_transformation,
            petrophysical_trust_region=self.petrophysical_trust_region.copy()
        )
        return model_new