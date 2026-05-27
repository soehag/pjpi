"""Anisotropic smoothing helpers.

Note
----
This module is experimental and not fully implemented or exhaustively tested.
In particular, `aniso_phi` is a placeholder and currently raises
`NotImplementedError`. Use these routines with caution and add tests
before using in production workflows.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygimli as pg
from functools import partial
from .regularisationcore import Regularisation
from gnmesh.meshtools.spatialgradient import calculate_spatial_gradient

def aniso_homogeneous_kernel(physics_and_data, model_info, anisotropy_matrix_field):
    """
    Calculate the homogeneous contribution to the anisotropic smoothing kernel.
    The homogeneous kernel is non dependent on the change of the anisotropy matrix 
    throught the model domain.
    The homogeneous kernel is given as:
        trace(C * partial / partial m H)
    """
    assert model_info.taylor_order >=2, "The taylor order must be greater than 1"

    mesh_info = model_info.mesh_info
    dimension = mesh_info.dimension

    assert anisotropy_matrix_field[0].shape[0] == dimension, "The anisotropy matrix must have the same dimension as the model"
    assert anisotropy_matrix_field[0].shape[1] == dimension, "The anisotropy matrix must have the same dimension as the model"
    assert dimension ==2, "The anisotropic smoothing kernel is only implemented for 2D models. Construction site is the Hessian sensitivities for 3D"

    # Get the mesh information
    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    homogeneous_kernel = np.zeros((no_of_model_parameters, no_of_model_parameters))

    for i in range(no_of_model_parameters):
        # Get Hessian matrix sensitivities
        cell_neighbours, hessian_matrix_sensitivities = mesh_info.cell_neighbour_info[i].get_hessian_mesh_sensitivities()
        for j in range(dimension):
            # Calculate the dot product of the anisotropy matrix and the spatial gradient
            # cheeky way of first and second column of heassian
            anisotropy_matrix_at_cell = anisotropy_matrix_field[i]
            hessian_matrix_sensitivities_column = hessian_matrix_sensitivities[j:j+2,:]
            dot_product = np.dot(anisotropy_matrix_at_cell[j], hessian_matrix_sensitivities_column)
            # Calculate the homogeneous kernel
            homogeneous_kernel[i, cell_neighbours] += dot_product
    return np.sqrt(area_of_cells) * homogeneous_kernel

def aniso_heterogeneous_kernel(physics_and_data, model_info, divergence_of_anisotropy_matrix_field=None):
    """
    Calculate the heterogeneous contribution to the anisotropic smoothing kernel. The heterogeneous kernel
    just arises when the anisotropic matrix changes within the model domain.
    The heterogeneous kernel is given as:
        partial / partial m [ nabla * C * nabla m] = nabla * C * O
    """

    assert model_info.taylor_order >=2, "The taylor order must be greater than 1"

    mesh_info = model_info.mesh_info

    no_of_model_parameters = mesh_info.mesh.cellCount()

    area_of_cells = np.array(
        [cni.cell_area for cni in mesh_info.cell_neighbour_info]
    )

    heterogeneous_kernel = np.zeros((no_of_model_parameters, no_of_model_parameters))

    for i in range(no_of_model_parameters):
        if not divergence_of_anisotropy_matrix_field is None:
            matrix_divergence = divergence_of_anisotropy_matrix_field[i]
            # Get the Gradient matrix sensitivities
            cell_neighbours, gradient_matrix_sensitivities = mesh_info.cell_neighbour_info[i].get_gradient_mesh_sensitivities()
            heterogeneous_kernel[i, cell_neighbours] = matrix_divergence @ gradient_matrix_sensitivities
    return np.sqrt(area_of_cells) * heterogeneous_kernel

def aniso_kernel_full(physics_and_data, model_info, anisotropy_matrix_field, divergence_of_anisotropy_matrix_field=None):
    """
    Calculate the full anisotropic smoothing kernel
    """
    homogeneous_kernel = aniso_homogeneous_kernel(physics_and_data, model_info, anisotropy_matrix_field)
    heterogeneous_kernel = aniso_heterogeneous_kernel(physics_and_data, model_info, divergence_of_anisotropy_matrix_field)
    return homogeneous_kernel + heterogeneous_kernel

def anisokernel_jacobian(physics_and_data, model_info, anisotropy_matrix_field, divergence_of_anisotropy_matrix_field=None):
    """
    Calculate the jacobian of the anisotropic smoothing kernel
    """
    assert model_info.taylor_order >=2, "The taylor order must be greater than 1"

    return aniso_kernel_full(
        physics_and_data=physics_and_data,
        model_info=model_info,
        anisotropy_matrix_field=anisotropy_matrix_field,
        divergence_of_anisotropy_matrix_field=divergence_of_anisotropy_matrix_field
    )

def aniso_phi(physics_and_data, model_info, anisotropy_matrix_field, model_transformation_regularisation=None):
    """
    Calculate the anisotropic smoothing kernel
    """
    raise NotImplementedError("aniso_phi is not yet implemented")

# kernel plotting routines

def plot_kernel(model_info, kernel, cell=0, ax=None, figsize=(5,5), cbar_clip_percentage=.8):
    """
    Plot the kernel
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    mesh_info = model_info.mesh_info
    cmax = np.max(np.abs(kernel[cell,:])) * cbar_clip_percentage
    cmin = -cmax
    _ = pg.show(
        mesh_info.mesh,
        kernel[cell,:],
        ax=ax,
        label=f"Kernel for cell {cell} - row sum: {np.sum(kernel[cell,:]):.2f}",
        cMin=cmin,
        cMax=cmax,
        cMap="seismic")
    return fig, ax

def plot_homogeneous_kernel(model_info, anisotropy_matrix_field, cell=0, ax=None, figsize=(5,5), cbar_clip_percentage=.8):
    """
    Plot the homogeneous kernel
    """
    homogeneous_kernel = aniso_homogeneous_kernel(
        physics_and_data=None,
        model_info=model_info,
        anisotropy_matrix_field=anisotropy_matrix_field
    )
    return plot_kernel(model_info, homogeneous_kernel, cell=cell, ax=ax, figsize=figsize, cbar_clip_percentage=cbar_clip_percentage)

def plot_heterogeneous_kernel(model_info, divergence_of_anisotropy_matrix_field, cell=0, ax=None, figsize=(5,5), cbar_clip_percentage=.8):
    """
    Plot the heterogeneous kernel
    """
    heterogeneous_kernel = aniso_heterogeneous_kernel(
        physics_and_data=None,
        model_info=model_info,
        divergence_of_anisotropy_matrix_field=divergence_of_anisotropy_matrix_field
    )
    return plot_kernel(model_info, heterogeneous_kernel, cell=cell, ax=ax, figsize=figsize, cbar_clip_percentage=cbar_clip_percentage)

def plot_all_kernels(model_info, anisotropy_matrix_field, divergence_of_anisotropy_matrix_field, cell=0, ax=None, figsize=(15,5), cbar_clip_percentage=.8):
    """
    Plot the kernel
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = ax[0].get_figure()

    homogeneous_kernel = aniso_homogeneous_kernel(
        physics_and_data=None,
        model_info=model_info,
        anisotropy_matrix_field=anisotropy_matrix_field
    )
    heterogeneous_kernel = aniso_heterogeneous_kernel(
        physics_and_data=None,
        model_info=model_info,
        divergence_of_anisotropy_matrix_field=divergence_of_anisotropy_matrix_field
    )
    full_kernel = aniso_kernel_full(
        physics_and_data=None,
        model_info=model_info,
        anisotropy_matrix_field=anisotropy_matrix_field,
        divergence_of_anisotropy_matrix_field=divergence_of_anisotropy_matrix_field
    )

    plot_kernel(model_info, homogeneous_kernel, cell=cell, ax=ax[0], cbar_clip_percentage=cbar_clip_percentage)
    plot_kernel(model_info, heterogeneous_kernel, cell=cell, ax=ax[1], cbar_clip_percentage=cbar_clip_percentage)
    plot_kernel(model_info, full_kernel, cell=cell, ax=ax[2], cbar_clip_percentage=cbar_clip_percentage)

    return fig, ax


def plot_aniso_diffusion_tensor_field(model_info, anisotropy_matrix_field, ax=None, figsize=(5,5), scale=200):
    """
    Plot the diffusion tensor as quiver plot."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    mesh_info = model_info.mesh_info
    dimension = mesh_info.dimension
    no_of_model_parameters = mesh_info.mesh.cellCount()
    _ = pg.show(mesh_info.mesh, ax=ax, showMesh=True)

    # get cell centers x and y
    cell_centers = np.array([cni.cell_center for cni in mesh_info.cell_neighbour_info])
    x = cell_centers[:,0]
    y = cell_centers[:,1]

    # calculate eigenvalue decomposition of the anisotropy matrix
    eigendecomposition = [sp.linalg.eigh(anisotropy_matrix_field[i]) for i in range(no_of_model_parameters)]
    # quiver first eigenvector
    u = np.array(
        [eigen_obj[0][0]*eigen_obj[1][0,0] for eigen_obj in eigendecomposition]
    )
    v = np.array(
        [eigen_obj[0][0]*eigen_obj[1][1,0] for eigen_obj in eigendecomposition]
    )
    ax.quiver(x,y,u,v, scale=scale, color='r', label='First eigenvector', angles='xy', scale_units='xy')
    # quiver second eigenvector
    u = np.array(
        [eigen_obj[0][1]*eigen_obj[1][0,1] for eigen_obj in eigendecomposition]
    )
    v = np.array(
        [eigen_obj[0][1]*eigen_obj[1][1,1] for eigen_obj in eigendecomposition]
    )
    ax.quiver(x,y,u,v, scale=scale, color='b', label='Second eigenvector', angles='xy', scale_units='xy')
    return fig, ax

class AnisotropicSmoothing(Regularisation):
    """
    Anisotropic smoothing regularisation
    """

    def __init__(self, anisotropy_matrix_field, divergence_of_anisotropy_matrix_field=None):
        self.anisotropy_matrix_field = anisotropy_matrix_field
        self.divergence_of_anisotropy_matrix_field = divergence_of_anisotropy_matrix_field
        
        aniso_jacobian_object = partial(
            anisokernel_jacobian,
            anisotropy_matrix_field=self.anisotropy_matrix_field,
            divergence_of_anisotropy_matrix_field=self.divergence_of_anisotropy_matrix_field
        )

        aniso_phi_object = partial(
            aniso_phi,
            anisotropy_matrix_field=self.anisotropy_matrix_field
        )
        super().__init__(
            calculate_jacobian=aniso_jacobian_object,
            calculate_phi=aniso_phi_object,
            static_jacobian=True
        )

    def plot_homogeneous_kernel(self, model_info, cell=0, ax=None, figsize=(5,5), cbar_clip_percentage=.8):
        """
        Plot the homogeneous kernel
        """
        fig, ax = plot_homogeneous_kernel(
            model_info,
            self.anisotropy_matrix_field,
            cell=cell,
            ax=ax,
            figsize=figsize,
            cbar_clip_percentage=cbar_clip_percentage
        )
        return fig, ax

    def plot_heterogeneous_kernel(self, model_info, cell=0, ax=None, figsize=(5,5), cbar_clip_percentage=.8):
        """
        Plot the heterogeneous kernel
        """
        fig, ax = plot_heterogeneous_kernel(
            model_info,
            self.divergence_of_anisotropy_matrix_field,
            cell=cell,
            ax=ax,
            figsize=figsize,
            cbar_clip_percentage=cbar_clip_percentage
        )
        return fig, ax

    def plot_kernel(self, model_info, cell=0, figsize=(15,5), cbar_clip_percentage=.8):
        """
        Plot the homogeneous and heterogeneous kernel
        """
        fig, ax = plot_all_kernels(
            model_info,
            self.anisotropy_matrix_field,
            self.divergence_of_anisotropy_matrix_field,
            cell=cell,
            figsize=figsize,
            cbar_clip_percentage=cbar_clip_percentage
        )
        return fig, ax
    
    def plot_diffusion_tensor_field(self, model_info, ax=None, figsize=(5,5), scale=200):
        """
        Plot the diffusion tensor field
        """
        fig, ax = plot_aniso_diffusion_tensor_field(
            model_info,
            self.anisotropy_matrix_field,
            ax=ax,
            figsize=figsize,
            scale=scale
        )
        return fig, ax
