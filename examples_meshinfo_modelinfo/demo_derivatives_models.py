# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pgcupy
#     language: python
#     name: python3
# ---

# %%
""" Demo: Derivatives of models

Computes analytic gradient/hessian for different models and compares them to numerical derivatives.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import gnmesh.meshtools as mI
import gnmesh.meshtools.spatialgradient as sG
from pygimli.viewer.mpl import drawModel, drawMeshBoundaries

# %%
def main():
    # Define simple mesh

    circle = pg.meshtools.createCircle(pos=[0, 0], radius=1, area=0.1, nSegments=100)
    mesh = pg.meshtools.createMesh(circle, area=0.001)
    cell_centers = np.array([np.array(cell.center())[0:2] for cell in mesh.cells()])

    # Select one of the three analytic test models used in this demo.
    MODEL_CHOICE = "QUADRATIC" # Possible values: EDGE, LOGARITHMIC, QUADRATIC

    if MODEL_CHOICE == "LOGARITHMIC":
        alpha = 100
        a = 1.1
        beta = 100
        b = 1.1
        model = np.log(alpha*(a+cell_centers[:,0])) * np.log(beta*(b+cell_centers[:,1]))
        gradient_analytic = np.array(
            [
            [(a+cc[0])**-1 * np.log(beta*(b+cc[1])), (b+cc[1])**-1 * np.log(alpha*(a+cc[0]))]
            for cc in cell_centers
            ])

        hessian_analytic = np.array([
            [
                [-(a+cc[0])**-2 * np.log(beta*(b+cc[1])), (a+cc[0])**-1 * (b+cc[1])**-1],
                [(a+cc[0])**-1 * (b+cc[1])**-1, -(b+cc[1])**-2 * np.log(alpha*(a+cc[0]))]
            ]
            for cc in cell_centers
        ])
    elif MODEL_CHOICE == "QUADRATIC":
        Q_full = np.array([[1, 0], [0, 42]])
        G_full = np.array([20, 20])
        N_alpha = 9
        model = np.array([cc @ Q_full @ cc + G_full @ cc + N_alpha * np.sin(4 * np.sum(cc)) for cc in cell_centers])
        gradient_analytic = 2 * cell_centers @ Q_full + G_full + N_alpha * 4 * np.tile(np.cos(4*np.sum(cell_centers, axis=1)), reps=(2,1)).T
        hessian_analytic = np.array(
            [
            Q_full - N_alpha * 16 * np.sin(4*np.sum(cell_center))
            for cell_center in cell_centers
            ])
    else:
        raise ValueError("Invalid model choice. Please choose from 'EDGE', 'LOGARITHMIC', or 'QUADRATIC'.")
    
    laplacian_analytic = np.array(
    [
        np.sum(np.diag(hess_matrix_cell))
    for hess_matrix_cell in hessian_analytic
    ])

    # Calculate numerical derivatives
    mi = mI.MeshInfo(
        mesh=mesh,
        initialise_gn2=True,
        neighbour_function=mI.meshinfo.get_n_closest_neighbours_function_for_mesh(mesh=mesh, n=8)
    )

    gradient_t1 = sG.calculate_spatial_gradient(
        model=model,
        mesh_info=mi
    )

    gradient_t2 = sG.calculate_spatial_gradient(
        model=model,
        mesh_info=mi,
        taylor_order=2
    )

    hessian_numeric = sG.calculate_hessian_matrix(
        model = model,
        mesh_info=mi
    )

    # Calculate diagnostics for both numerical and analytical derivatives

    # Analytic

    eval_evec_analytic = [np.linalg.eig(hess_cell) for hess_cell in hessian_analytic]
    eval_analytic = np.array([np.array([np.max(ev[0]), np.min(ev[0])]) for ev in eval_evec_analytic])
    evec_analytic = np.array([ev[1][:,0]for ev in eval_evec_analytic])
    eval_max_analytic = eval_analytic[:,0]
    eval_min_analytic = eval_analytic[:,1]

    # Numeric
    eval_evec_numeric = [np.linalg.eig(hess_cell) for hess_cell in hessian_numeric]
    eval_numeric = np.array([np.array([np.max(ev[0]), np.min(ev[0])]) for ev in eval_evec_numeric])
    evec_numeric = np.array([ev[1][:,0]for ev in eval_evec_numeric])
    eval_max_numeric = eval_numeric[:,0]
    eval_min_numeric = eval_numeric[:,1]

    # Plot 1: Model and Gradients
    # Plot Model, analytic gradient, numeric gradient (Taylor order 1 and 2), Difference between analytic and numeric gradients
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Quiver scale needs to be smaller for the quadratic model because its gradients are larger.
    s = 500 if MODEL_CHOICE == "QUADRATIC" else 100
    # Model
    gci = drawModel(ax=axs[0,0], mesh=mesh, data=model)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=axs[0,0], mesh=mesh, color="k", linewidth=0.5)
    axs[0,0].set_title("Model")
    try:
        fig.colorbar(gci, ax=axs[0,0], fraction=0.046, pad=0.04)
    except Exception:
        pass

    # Gradient Analytic with quiver
    axs[0,1].quiver(cell_centers[:,0], cell_centers[:,1], gradient_analytic[:,0], gradient_analytic[:,1], color="r", scale=s)
    drawMeshBoundaries(ax=axs[0,1], mesh=mesh, color="k", linewidth=0.5)
    axs[0,1].set_title("Analytic Gradient")

    # Gradient Numeric Taylor order 1 with quiver
    axs[0,2].quiver(cell_centers[:,0], cell_centers[:,1], gradient_t1[:,0], gradient_t1[:,1], color="b", scale=s)
    drawMeshBoundaries(ax=axs[0,2], mesh=mesh, color="k", linewidth=0.5)
    axs[0,2].set_title("Numeric Gradient (Taylor order 1)")

    # Gradient Numeric Taylor order 2 with quiver
    axs[1,0].quiver(cell_centers[:,0], cell_centers[:,1], gradient_t2[:,0], gradient_t2[:,1], color="g", scale=s)
    drawMeshBoundaries(ax=axs[1,0], mesh=mesh, color="k", linewidth=0.5)
    axs[1,0].set_title("Numeric Gradient (Taylor order 2)")


    # Difference between analytic and numeric gradients (Taylor order 1)
    gradient_diff_t1 = np.linalg.norm(gradient_analytic - gradient_t1, axis=1)
    gci = drawModel(ax=axs[1,1], mesh=mesh, data=gradient_diff_t1)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=axs[1,1], mesh=mesh, color="k", linewidth=0.5)
    axs[1,1].set_title("Difference Analytic/Numeric Gradient (Taylor order 1)")
    try:
        fig.colorbar(gci, ax=axs[1,1], fraction=0.046, pad=0.04)
    except Exception:
        pass

    # Difference between analytic and numeric gradients (Taylor order 2)
    gradient_diff_t2 = np.linalg.norm(gradient_analytic - gradient_t2, axis=1)
    gci = drawModel(ax=axs[1,2], mesh=mesh, data=gradient_diff_t2)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=axs[1,2], mesh=mesh, color="k", linewidth=0.5)
    axs[1,2].set_title("Difference Analytic/Numeric Gradient (Taylor order 2)")
    try:
        fig.colorbar(gci, ax=axs[1,2], fraction=0.046, pad=0.04)
    except Exception:
        pass
    pg.plt.show()

    # Plot 2: Hessian diagnostics
    # Plot Norm of Hessian, Max eigenvalue, Min eigenvalue
    fig = plt.figure(figsize=(26, 8), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        8,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05],
        wspace=0.25,
        hspace=0.25,
    )
    ax00 = fig.add_subplot(gs[0, 0])
    cax00 = fig.add_subplot(gs[0, 1])
    ax01 = fig.add_subplot(gs[0, 2])
    cax01 = fig.add_subplot(gs[0, 3])
    ax02 = fig.add_subplot(gs[0, 4])
    cax02 = fig.add_subplot(gs[0, 5])
    ax03 = fig.add_subplot(gs[0, 6])
    cax03 = fig.add_subplot(gs[0, 7])
    ax10 = fig.add_subplot(gs[1, 0])
    cax10 = fig.add_subplot(gs[1, 1])
    ax11 = fig.add_subplot(gs[1, 2])
    cax11 = fig.add_subplot(gs[1, 3])
    ax12 = fig.add_subplot(gs[1, 4])
    cax12 = fig.add_subplot(gs[1, 5])
    ax13 = fig.add_subplot(gs[1, 6])
    cax13 = fig.add_subplot(gs[1, 7])

    # Norm of Hessian Analytic
    hessian_norm_analytic = np.linalg.norm(hessian_analytic, axis=(1,2))
    gci = drawModel(ax=ax00, mesh=mesh, data=hessian_norm_analytic)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax00, mesh=mesh, color="k", linewidth=0.5)
    ax00.set_title("Norm of Hessian (Analytic)")
    fig.colorbar(gci, cax=cax00)

    # Max eigenvalue of Hessian Analytic
    gci = drawModel(ax=ax01, mesh=mesh, data=eval_max_analytic)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax01, mesh=mesh, color="k", linewidth=0.5)
    ax01.set_title("Max Eigenvalue of Hessian (Analytic)")
    fig.colorbar(gci, cax=cax01)

    # Min eigenvalue of Hessian Analytic
    gci = drawModel(ax=ax02, mesh=mesh, data=eval_min_analytic)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax02, mesh=mesh, color="k", linewidth=0.5)
    ax02.set_title("Min Eigenvalue of Hessian (Analytic)")
    fig.colorbar(gci, cax=cax02)

    # Norm of Hessian Numeric
    hessian_norm_numeric = np.linalg.norm(hessian_numeric, axis=(1,2))
    gci = drawModel(ax=ax10, mesh=mesh, data=hessian_norm_numeric)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax10, mesh=mesh, color="k", linewidth=0.5)
    ax10.set_title("Norm of Hessian (Numeric)")
    fig.colorbar(gci, cax=cax10)

    # Max eigenvalue of Hessian Numeric
    gci = drawModel(ax=ax11, mesh=mesh, data=eval_max_numeric)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax11, mesh=mesh, color="k", linewidth=0.5)
    ax11.set_title("Max Eigenvalue of Hessian (Numeric)")
    fig.colorbar(gci, cax=cax11)

    # Min eigenvalue of Hessian Numeric
    gci = drawModel(ax=ax12, mesh=mesh, data=eval_min_numeric)
    gci.set_cmap("turbo")
    drawMeshBoundaries(ax=ax12, mesh=mesh, color="k", linewidth=0.5)
    ax12.set_title("Min Eigenvalue of Hessian (Numeric)")
    fig.colorbar(gci, cax=cax12)

    # Laplacian (analytic and numeric)
    laplacian_numeric = np.array([np.sum(np.diag(h)) for h in hessian_numeric])
    gci = drawModel(ax=ax03, mesh=mesh, data=laplacian_analytic)
    if gci is not None:
        gci.set_cmap("turbo")
        drawMeshBoundaries(ax=ax03, mesh=mesh, color="k", linewidth=0.5)
        ax03.set_title("Laplacian (Analytic)")
        fig.colorbar(gci, cax=cax03)
    else:
        drawMeshBoundaries(ax=ax03, mesh=mesh, color="k", linewidth=0.5)
        ax03.set_title("Laplacian (Analytic)")
        cax03.axis("off")

    gci = drawModel(ax=ax13, mesh=mesh, data=laplacian_numeric)
    if gci is not None:
        gci.set_cmap("turbo")
        drawMeshBoundaries(ax=ax13, mesh=mesh, color="k", linewidth=0.5)
        ax13.set_title("Laplacian (Numeric)")
        fig.colorbar(gci, cax=cax13)
    else:
        drawMeshBoundaries(ax=ax13, mesh=mesh, color="k", linewidth=0.5)
        ax13.set_title("Laplacian (Numeric)")
        cax13.axis("off")

    pg.plt.show()

# %%
if __name__ == "__main__":
    main()
