"""Demo: ModelInfo utilities showcase.

Creates a mesh, builds `MeshInfo` and `ModelInfo` (Taylor order 1 and 2)
and visualises gradients, transformed model and Hessian overview.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gS
import pygimli as pg
import gnmesh.meshtools as mI
import gnmesh.meshtools.modelinfo as modI
from pygimli.viewer.mpl import drawModel, drawMeshBoundaries
import gnmesh.meshtools.transformation as tF


def main():
    area = 0.001
    model_choice = "quadratic"
    circle = pg.meshtools.createCircle(pos=[0, 0], radius=1, area=area, nSegments=100)

    # (optional) add additional polygon/circles for more interesting mesh shapes
    # The test version used an optional pygimli case; omit here for clarity.

    mesh = pg.meshtools.createMesh(circle, area=area)

    # Create a model on cell centers
    cell_centers = np.array([np.array(cell.center())[0:2] for cell in mesh.cells()])
    gradient_direction = np.array([-1, 1]) / np.sqrt(2)

    if model_choice == "linear":
        variation = 100
        offset = 700
        model = variation * (cell_centers @ gradient_direction) + offset
    elif model_choice == "quadratic":
        model = np.array([cell_center @ np.array([[1, 0], [0, 40]]) @ cell_center for cell_center in cell_centers])
    else:
        model = pg.solver.parseArgToArray([[1, 800.0], [2, 500.0], [3, 150.0]], mesh.cellCount(), mesh)

    mi = mI.MeshInfo(
        mesh=mesh,
        initialise_gn2=True,
        neighbour_function=mI.meshinfo.get_n_closest_neighbours_function_for_mesh(mesh=mesh, n=8)
        )
    
    print(f"Minimum neighbours: {np.min([len(cni.neighbour_cells) for cni in mi.cell_neighbour_info])}")

    minimum_model_value = np.min(model)
    maximum_model_value = np.max(model)

    transformation = tF.LogarithmicBarrierTransformationTwoSided(
        lower_barrier=minimum_model_value * 0.9,
        upper_barrier=maximum_model_value * 1.1,
    )

    mod_gn_1 = modI.ModelInfo(
        model=model,
        mesh_info=mi,
        taylor_order=1,
        transformation=transformation,
        )
    
    mod_gn_2 = modI.ModelInfo(
        model=model,
        mesh_info=mi,
        taylor_order=2,
        transformation=transformation,
        )
    
    # Plot the model
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    gs_model = gS.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.1)
    ax_model = fig.add_subplot(gs_model[0, 0])
    cax_model = fig.add_subplot(gs_model[0, 1])
    mod_gn_1.plot_model(show_mesh=True, ax=ax_model)
    mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=minimum_model_value, vmax=maximum_model_value),
        cmap="viridis",
    )
    mappable.set_array([])
    fig.colorbar(mappable, cax=cax_model)
    ax_model.set_title("Model")


    # Plot the transformed model
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    gs_transformed = gS.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.1)
    ax_transformed = fig.add_subplot(gs_transformed[0, 0])
    cax_transformed = fig.add_subplot(gs_transformed[0, 1])
    mod_gn_1.plot_transformed_model(show_mesh=True, ax=ax_transformed)
    transformed_model = mod_gn_1.transformed_model
    transformed_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=np.min(transformed_model), vmax=np.max(transformed_model)),
        cmap="viridis",
    )
    transformed_mappable.set_array([])
    fig.colorbar(transformed_mappable, cax=cax_transformed)
    ax_transformed.set_title("Transformed model")

    # Plot absolute value of spatial gradient
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    gs_gradient = gS.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.1)
    ax_gradient = fig.add_subplot(gs_gradient[0, 0])
    cax_gradient = fig.add_subplot(gs_gradient[0, 1])
    mod_gn_1.plot_absolute_value_of_spatial_gradient(ax=ax_gradient, show_colorbar=False, cmap="inferno")
    abs_norm_of_spatial_gradient = np.linalg.norm(mod_gn_1.spatial_gradient, axis=1)
    gradient_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=np.min(abs_norm_of_spatial_gradient), vmax=np.max(abs_norm_of_spatial_gradient)),
        cmap="inferno",
    )
    gradient_mappable.set_array([])
    fig.colorbar(gradient_mappable, cax=cax_gradient)
    ax_gradient.set_title("Absolute value of spatial gradient")

    # Plot Laplacian field with a colorbar next to it.
    fig_laplacian = plt.figure(figsize=(6, 5), constrained_layout=True)
    gs_laplacian = gS.GridSpec(1, 2, figure=fig_laplacian, width_ratios=[1, 0.05], wspace=0.1)
    ax_laplacian = fig_laplacian.add_subplot(gs_laplacian[0, 0])
    cax_laplacian = fig_laplacian.add_subplot(gs_laplacian[0, 1])
    laplacian_vector = mod_gn_2.laplacian
    mod_gn_2.plot_laplacian_field(
        ax=ax_laplacian,
        cMin=-1e5,
        cMax=1e5,
        cmap="turbo",
        cmin=np.min(laplacian_vector)-1,
        cmax=np.max(laplacian_vector)+1
        )
    laplacian_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=np.min(laplacian_vector)-1, vmax=np.max(laplacian_vector)+1),
        cmap="turbo",
    )
    laplacian_mappable.set_array([])
    cbar_laplacian = fig_laplacian.colorbar(laplacian_mappable, cax=cax_laplacian)
    ax_laplacian.set_title("Laplacian field (Taylor order 2)")

    # Plot Hessian overview (six components) in a new figure with GridSpec

    fig_hess, ax_hess = mod_gn_2.plot_hessian_matrix_overview()
    fig_hess.canvas.draw()

    plt.show(block=True)


if __name__ == "__main__":
    main()
