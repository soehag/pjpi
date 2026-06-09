"""Demo: meshinfo sensitivities.

Creates a small mesh, initialises `MeshInfo`, and plots the GN1/GN2
sensitivity matrices in a single 2x2 figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import gnmesh.meshtools as mI
from pygimli.viewer.mpl import drawModel, drawMeshBoundaries
import logging

logger = logging.getLogger(__name__)



def main():
    circle = pg.meshtools.createCircle(pos=[0, 0], radius=1, area=0.1, nSegments=100)
    mesh = pg.meshtools.createMesh(circle, area=0.01)

    mi = mI.MeshInfo(
        mesh,
        neighbour_function=mI.meshinfo.get_n_closest_neighbours_function_for_mesh(mesh=mesh, n=5),
        initialise_gn2=True,
    )

    min_cell_neighbours = np.min([len(cni.neighbour_cells) for cni in mi.cell_neighbour_info])
    logger.info("Minimum number of neighbour cells: %s", min_cell_neighbours)

    # Pick a stable interior point near the lower-right quadrant for the neighbourhood example.
    px = 0.5
    py = 0.1
    cell_id = mesh.findCell([px, py]).id()

    # Show selected cell and its neighbours (like in test)
    fig_n, ax_n = plt.subplots(1, 1)
    gci = drawModel(ax_n, mesh, showMesh=True, data=np.zeros(mesh.cellCount()))
    gci.set_cmap("seismic")
    gci.set_clim(-1, 1)
    drawMeshBoundaries(ax_n, mesh, hideMesh=False, lw=0.3, color="k", fitView=False)

    cell = mesh.cells()[cell_id]
    ax_n.plot(cell.center()[0], cell.center()[1], 'ro', markersize=4)
    ax_n.hlines(py, -1, 1, linewidth=0.5, color='orange')
    ax_n.vlines(px, -1, 1, linewidth=0.5, color='orange')
    for n in mi.cell_neighbour_info[cell_id].neighbour_cells:
        n_cell = mesh.cells()[n]
        ax_n.plot(n_cell.center()[0], n_cell.center()[1], 'go', markersize=4)
    ax_n.set_title('Selected cell and neighbours')
    fig_n.suptitle('MeshInfo: selected cell neighbourhood', fontsize=12)

    ind_gn1, sens_gn1 = mi.cell_neighbour_info[cell_id].get_gradient_mesh_sensitivities()
    ind_gn2, sens_gn2 = mi.cell_neighbour_info[cell_id].get_gradient_mesh_sensitivities(order=2)

    full_sensitivity_matrix_gn1 = np.zeros((2, mesh.cellCount()))
    full_sensitivity_matrix_gn1[:, ind_gn1] = sens_gn1

    full_sensitivity_matrix_gn2 = np.zeros((2, mesh.cellCount()))
    full_sensitivity_matrix_gn2[:, ind_gn2] = sens_gn2

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    cmax = np.max(np.abs(full_sensitivity_matrix_gn1))
    cmin = -cmax

    gci = drawModel(axs[0, 0], mesh, data=full_sensitivity_matrix_gn1[0], cMin=cmin, cMax=cmax, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin, cmax)
    drawMeshBoundaries(axs[0, 0], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)

    gci = drawModel(axs[0, 1], mesh, data=full_sensitivity_matrix_gn1[1], cMin=cmin, cMax=cmax, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin, cmax)
    drawMeshBoundaries(axs[0, 1], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)

    gci = drawModel(axs[1, 0], mesh, data=full_sensitivity_matrix_gn2[0], cMin=cmin, cMax=cmax, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin, cmax)
    drawMeshBoundaries(axs[1, 0], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)

    gci = drawModel(axs[1, 1], mesh, data=full_sensitivity_matrix_gn2[1], cMin=cmin, cMax=cmax, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin, cmax)
    drawMeshBoundaries(axs[1, 1], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)

    # Column titles for components
    axs[0, 0].set_title("x component", pad=20)
    axs[0, 1].set_title("y component", pad=20)

    # Row labels for Taylor orders (left-side)
    axs[0, 0].set_ylabel("Taylor order 1", rotation=90, labelpad=10)
    axs[1, 0].set_ylabel("Taylor order 2", rotation=90, labelpad=10)

    fig.suptitle('MeshInfo: Gradient sensitivity matrices for taylor order 1 and 2', fontsize=12)

    # --- Hessian plot in a new figure (three components) ---
    ind_hessian, hessian_matrix = mi.cell_neighbour_info[cell_id].get_hessian_mesh_sensitivities()
    full_hessian_matrix = np.zeros((3, mesh.cellCount()))
    full_hessian_matrix[:, ind_hessian] = hessian_matrix

    fig_h, axs_h = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    cmax_h = np.max(np.abs(full_hessian_matrix[0]))
    cmin_h = -cmax_h

    gci = drawModel(axs_h[0], mesh, data=full_hessian_matrix[0], cMin=cmin_h, cMax=cmax_h, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin_h, cmax_h)
    drawMeshBoundaries(axs_h[0], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)
    axs_h[0].set_title("Hessian xx")

    gci = drawModel(axs_h[1], mesh, data=full_hessian_matrix[1], cMin=cmin_h, cMax=cmax_h, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin_h, cmax_h)
    drawMeshBoundaries(axs_h[1], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)
    axs_h[1].set_title("Hessian xy")

    gci = drawModel(axs_h[2], mesh, data=full_hessian_matrix[2], cMin=cmin_h, cMax=cmax_h, cMap="seismic", showMesh=True)
    if gci is not None:
        gci.set_cmap("seismic")
        gci.set_clim(cmin_h, cmax_h)
    drawMeshBoundaries(axs_h[2], mesh, hideMesh=False, lw=0.3, color="k", fitView=False)
    axs_h[2].set_title("Hessian yy")
    fig_h.suptitle('MeshInfo: Hessian sensitivity components', fontsize=12)

    logger.info("GN1 sensitivity vector length: %s; GN2: %s", len(ind_gn1), len(ind_gn2))

    # Show region of interest in the mesh plot
    mi.show_region_of_interest()
    plt.show(block=True)


if __name__ == "__main__":
    main()
