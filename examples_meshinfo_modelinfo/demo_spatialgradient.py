"""Demo: spatial gradient calculation and visualization.

Creates a circular mesh, a simple linear model and shows gradient field.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pygimli as pg
import gnmesh.meshtools as mI
import gnmesh.meshtools.spatialgradient as sG
from pygimli.viewer.mpl import drawModel, drawSelectedMeshBoundaries
import logging

logger = logging.getLogger(__name__)


def main():
    circle = pg.meshtools.createCircle(pos=[0, 0], radius=1, area=0.1, nSegments=100)
    mesh = pg.meshtools.createMesh(circle)
    figures = []

    # linear model in a fixed direction
    cell_centers = np.array([np.array(cell.center())[0:2] for cell in mesh.cells()])
    grad_dir = np.array([-1, 1]) / np.sqrt(2)
    model = 100 * (cell_centers @ grad_dir) + 700

    mi = mI.MeshInfo(
        mesh=mesh,
        initialise_gn1=True,
        initialise_gn2=True,
        neighbour_function=mI.meshinfo.get_n_closest_neighbours_function_for_mesh(
            mesh=mesh,
            n=8,
            )
        )
    gradient = sG.calculate_spatial_gradient(model=model, mesh_info=mi)

    # Single figure with three subplots and horizontal colorbars beneath using GridSpec
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], hspace=0.25, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])   # model
    ax1 = fig.add_subplot(gs[0, 1])   # vectors
    ax2 = fig.add_subplot(gs[0, 2])   # gradient norm
    cax0 = fig.add_subplot(gs[1, 0])  # colorbar for model (horizontal)
    cax1 = fig.add_subplot(gs[1, 1])  # spare colorbar axis (middle)
    cax2 = fig.add_subplot(gs[1, 2])  # colorbar for gradient norm (horizontal)

    fig.suptitle("Spatial gradient demo: model, vectors, and norm", fontsize=12)

    # Left: model
    gci = drawModel(ax0, mesh, data=model, showMesh=True)
    if gci is not None:
        try:
            fig.colorbar(gci, cax=cax0, orientation="horizontal").set_label("Model value")
        except Exception:
            pass
    drawSelectedMeshBoundaries(ax0, mesh.boundaries(), linewidth=0.3, color="0.2")
    ax0.set_title("Model")

    # Middle: vector gradient field
    sG.plot_gradient_field(spatial_gradient=gradient, mesh=mesh, scale=1e3, show_mesh=True, ax=ax1)
    ax1.set_title("Gradient (vectors)")
    # middle colorbar axis unused
    try:
        cax1.axis("off")
    except Exception:
        pass

    # Right: gradient norm (compute here to control colorbar placement)
    grad_abs = np.linalg.norm(gradient, axis=1)
    cMin = np.mean(grad_abs)-1
    cMax = np.mean(grad_abs)+1
    
    _ = sG.plot_absolute_value_of_gradient_field_from_vectors(
        spatial_gradient=gradient,
        mesh=mesh,
        ax=ax2,
        show_colorbar=False,
        cmin=cMin,
        cmax=cMax,
        cmap="copper",
    )

    # Create an explicit mappable so the last plot can get its own colorbar.
    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=cMin, vmax=cMax), cmap="copper")
    # mappable.set_array([])
    cbar2 = fig.colorbar(mappable, cax=cax2, orientation="horizontal")
    cbar2.set_ticks([cMin, np.mean(grad_abs), cMax])
    cbar2.set_label("Gradient norm")

    # Plot hessian matrix overview in a new figure
    hessian_matrix_list = sG.calculate_hessian_matrix(
        model=model,
        mesh_info=mi,
        taylor_order=2,
    )
    logger.info("Hessian matrix list: %s", hessian_matrix_list)
    sG.plot_hessian_matrix_overview(
        hessian_matrix_list=hessian_matrix_list,
        mesh=mesh,
        cmap="seismic",
    )

    plt.show()
    logger.info("Gradient demo finished — check plots.")


if __name__ == "__main__":
    main()