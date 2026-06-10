"""ERT inversion example script.

This module runs a geophysical (ERT) inversion on a prepared synthetic
crosshole mesh and data. It provides ``run_ert_geo_inversion`` and a
``__main__`` entrypoint for standalone execution.
"""

import os
import sys
from pathlib import Path
from functools import partial
import json
import logging

import matplotlib
# Use non-interactive backend for scripted figure export
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
# viewer helpers not used in this script; omit unused imports

from pygimli.physics import ert
from pygimli.physics import traveltime as tt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_helper.petrophysics import GassmannTransformation, ArchieTransformation
from scripts_helper import plotting_helpers as ph

import gnmesh.gncore.geophysical as gp
import gnmesh.gncore.petrophysical as pp
import gnmesh.gncore.petrophysical_decoupled as ppd
import gnmesh.gncore.physicsanddata as pd
import gnmesh.regularisation as reg
import gnmesh.meshtools as mths

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s", force=True)

# %%
#* Set up folder structure
def main():
    """Prepare data and run the default ERT inversion.

    This function performs filesystem setup, loads synthetic data and
    meshes, constructs petrophysical transformations and regularisation
    operators, and runs `run_ert_petro_inversion` with default parameters.
    It is intended for command-line execution only.
    """
    path_home = Path(__file__).resolve().parent.parent
    path_data_home = path_home / "data"
    path_data_synthetic = path_data_home / "synthetic"
    path_data_results = path_data_home / "results_pjpi"

    path_figures_home = path_home / "figures"
    path_figures_results = path_figures_home / "figures_pjpi"

    path_data_results.mkdir(parents=True, exist_ok=True)
    path_figures_results.mkdir(parents=True, exist_ok=True)

    # %%
    # Load data and inversion mesh
    inversion_mesh = pg.load(str(path_data_synthetic / "inversion_mesh.bms"))

    ert_data = ert.load(str(path_data_synthetic / "ert_measurement_noise_3.data"))
    ert_scheme = ert.load(str(path_data_synthetic / "ert_scheme.data"))

    tt_data = tt.load(str(path_data_synthetic / "tt_measurement_noise_3.data"))
    tt_scheme = tt.load(str(path_data_synthetic / "tt_scheme.data"))

    original_mesh_w_models = pg.load(str(path_data_synthetic / "final_mesh_with_models.bms"))
    
    # %%
    # Create transformations for reservoir and caprock

    # Parameters for Gassmann
    K_CO2 = 0.01 #Gpa
    RHO_CO2 = 231.53 #kg/m^3

    K_BRINE = 3.63 #Gpa
    RHO_BRINE = 1164.59  #kg/m^3

    K_MATRIX = 37.78 #Gpa
    RHO_MATRIX = 2670.89 #kg/m^3

    K_AIR = 1.42 * 10**5 * 10**-9 #Gpa
    K_DRY = 3.5 #Gpa for sample 1

    S_WAVE_VELOCITY_IVANOVA = 1.42
    RHO_IVANOVA = 4050

    # Parameters for Archie
    SATURATION_EXPONENT = 1.62
    CEMENTATION_EXPONENT = 2.0
    R_FLUID = 3e-2

    # Porosities and saturations

    PHI_RESERVOIR = 0.28
    PHI_CAPROCK = 0.12

    SATURATION_CAPROCK = 0.0
    SATURATION_RESERVOIR = 0.0
    SATURATION_CO2 = 0.5

    gassmann_reservoir = GassmannTransformation(
        k_co2=K_CO2,
        rho_co2=RHO_CO2,
        k_brine=K_BRINE,
        rho_brine=RHO_BRINE,
        k_matrix=K_MATRIX,
        rho_matrix=RHO_MATRIX,
        phi=PHI_RESERVOIR,
        version_fluid_mixtures="brie",
        output_parameter="slowness",
    )

    gassmann_caprock = GassmannTransformation(
        k_co2=K_CO2,
        rho_co2=RHO_CO2,
        k_brine=K_BRINE,
        rho_brine=RHO_BRINE,
        k_matrix=K_MATRIX,
        rho_matrix=RHO_MATRIX,
        phi=PHI_CAPROCK,
        version_fluid_mixtures="brie",
        output_parameter="slowness",
    )

    archie_reservoir = ArchieTransformation(
        r_fluid=R_FLUID,
        phi=PHI_RESERVOIR,
        m=CEMENTATION_EXPONENT,
        n=SATURATION_EXPONENT,
    )

    archie_caprock = ArchieTransformation(
        r_fluid=R_FLUID,
        phi=PHI_CAPROCK,
        m=CEMENTATION_EXPONENT,
        n=SATURATION_EXPONENT,
    )

    # %% Create meshinfo objects
    meshinfo_inversion = mths.MeshInfo(
        mesh=inversion_mesh,
        neighbour_function=mths.get_n_closest_neighbours_function_for_mesh(
            mesh=inversion_mesh,
            n=4,
        )
    )

    # %% Create region of interest (exclude outer mesh) and initial model vectors
    # Initial model vectors for inversion
    mean_apparent_resistivity = np.mean(np.array(original_mesh_w_models["res"]))
    saturation_from_mean_resistivity = archie_reservoir.backward(mean_apparent_resistivity)

    apparent_vp, _, _ = ph.apparent_velocity_from_data(tt_data)
    mean_apparent_vp = np.mean(apparent_vp)
    saturation_from_mean_vp = gassmann_reservoir.backward(1/mean_apparent_vp)

    initial_model_vector_res = np.ones(inversion_mesh.cellCount()) * mean_apparent_resistivity
    initial_model_vector_slowness = np.ones(inversion_mesh.cellCount()) * (1/mean_apparent_vp)
    initial_model_vector_saturation = np.ones(inversion_mesh.cellCount()) * np.mean([saturation_from_mean_resistivity, saturation_from_mean_vp])
    logger.info("Initial saturation: %s", initial_model_vector_saturation[0])

    # Region of interest for inversion (exclude outer mesh)
    BUFFER_X = 30
    BUFFER_Y = 30
    BUFFER_TRIANGLE_BOUNDARY = 20
    AREA_TRIANGLE_BOUNDARY = 100
    BOUNDARY_NODES = 20

    BOREHOLE_DISTANCE = 50

    LAYER_FROM_TOP_A = 80
    LAYER_FROM_TOP_B = 60
    inclination = (LAYER_FROM_TOP_B-LAYER_FROM_TOP_A)/(BOREHOLE_DISTANCE)

    ELECTRODE_SPACING = 5
    ELECTRODE_NUMBER = 31

    SEISMIC_STATION_SPACING = 5
    SEISMIC_STATION_NUMBER = 31

    length_boreholes = np.max([ELECTRODE_SPACING * (ELECTRODE_NUMBER-1), SEISMIC_STATION_SPACING * (SEISMIC_STATION_NUMBER-1)])

    BOREHOLE_A_X = BUFFER_X
    BOREHOLE_A_Y_MAX = -BUFFER_Y
    BOREHOLE_A_Y_MIN = BOREHOLE_A_Y_MAX - length_boreholes

    BOREHOLE_B_X = BOREHOLE_A_X + BOREHOLE_DISTANCE
    BOREHOLE_B_Y_MAX = -BUFFER_Y
    BOREHOLE_B_Y_MIN = BOREHOLE_A_Y_MAX - length_boreholes
    DOMAIN_EXTENT_X = 2 * BUFFER_X + BOREHOLE_DISTANCE
    DOMAIN_XMIN = 0
    DOMAIN_XMAX = DOMAIN_EXTENT_X

    DOMAIN_YMIN = -(2 * BUFFER_Y + length_boreholes)
    DOMAIN_YMAX = 0

    def plot_boreholes_on_mesh(ax):
        """Draw borehole location markers/lines on an axes object (non-blocking).

        This function only draws on the supplied axes. Do not call it when an
        interactive plot is required; the caller is responsible for saving or
        showing the figure.
        """
        ax.plot([BOREHOLE_A_X, BOREHOLE_A_X], [BOREHOLE_A_Y_MIN, BOREHOLE_A_Y_MAX], color="black", linestyle="--")
        ax.plot([BOREHOLE_B_X, BOREHOLE_B_X], [BOREHOLE_B_Y_MIN, BOREHOLE_B_Y_MAX], color="black", linestyle="--")

    cell_centers = np.array(inversion_mesh.cellCenters())[:,:2]
    region_of_interest = np.logical_and.reduce((
        cell_centers[:,0] >= DOMAIN_XMIN ,
        cell_centers[:,0] <= DOMAIN_XMAX ,
        cell_centers[:,1] >= DOMAIN_YMIN ,
        cell_centers[:,1] <= DOMAIN_YMAX
    ))

    meshinfo_inversion.region_of_interest = region_of_interest
    fig, ax = meshinfo_inversion.show_region_of_interest()
    plot_boreholes_on_mesh(ax)
    fig.savefig(path_figures_results / "inversion_region_of_interest.jpg", format='jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create decoupled region vector for regularisation - 1 and 2 for caprock and reservoir
    cell_centers = np.array(inversion_mesh.cellCenters())[:,:2]
    petrophysical_trust_region_on_imesh = cell_centers[:,1] < (-LAYER_FROM_TOP_A - inclination * (cell_centers[:,0]-BUFFER_X))
    decoupled_region_vector = petrophysical_trust_region_on_imesh*1+1

    # %% Set up inversion parameters for geophysical ERT inversion
    # Metaparameters for inversion
    MAXIMUM_ITERATIONS = 30

    SATURATION_MIN = 0.0
    SATURATION_MAX = 0.55

    RESISTIVITY_MIN = np.floor(np.min(original_mesh_w_models["res"])*0.9*1e1)*1e-1
    RESISTIVITY_MAX = np.ceil(np.max(original_mesh_w_models["res"])*1.1*1e1)*1e-1
    logger.info("Resistivity min: %s, max: %s", RESISTIVITY_MIN, RESISTIVITY_MAX)
    logger.info("In mesh: %s, %s", np.min(original_mesh_w_models["res"]), np.max(original_mesh_w_models["res"]))
    logger.info(f"Resistivity min: {RESISTIVITY_MIN}, max: {RESISTIVITY_MAX}")

    VELOCITY_MIN = np.floor((np.min(original_mesh_w_models["vp"])-50)*1e-1)*1e1
    VELOCITY_MAX = np.ceil((np.max(original_mesh_w_models["vp"])+50)*1e-1)*1e1
    logger.info("Velocity min: %s, max: %s", VELOCITY_MIN, VELOCITY_MAX)
    logger.info("In mesh: %s, %s", np.min(original_mesh_w_models['vp']), np.max(original_mesh_w_models['vp']))

    logger.info("Velocity min: %s, max: %s", VELOCITY_MIN, VELOCITY_MAX)
    logger.info("In mesh: %s, %s", np.min(original_mesh_w_models["vp"]), np.max(original_mesh_w_models["vp"]))

    MAX_SATURATION_UPDATE_PER_STEP = 0.1
    MAX_SLOWNESS_UPDATE_PER_STEP = 0.2
    MAX_RESISTIVITY_UPDATE_PER_STEP = 0.2

    # Regularization parameters - use pygimli matrix

    # Build regularisation operators using pygimli/gnmesh helpers.
    # `reg_manager` is a thin wrapper that exposes a `fop` with
    # region-based constraint generation used for smoothing operators.
    damping_operator = reg.DampingStepWidth

    reg_manager = tt.TravelTimeManager()
    reg_manager.setMesh(inversion_mesh)
    # ert_reg_manager.setData(ert_data)
    rm_ert = reg_manager.fop.regionManager()
    rm_ert.setConstraintType(1)
    m = reg_manager.fop.createConstraints()
    m = pg.utils.sparseMatrix2Dense(m)
    smoothing_operator = partial(
        reg.LinearOperator,
        linear_operator=m,
        rhs_vector=np.zeros(m.shape[0])
    )

    # Plotting parameters
    CMAP = "turbo"
    logScale = False
    C_MIN_RES = RESISTIVITY_MIN
    C_MAX_RES = RESISTIVITY_MAX
    REL_RESIDUAL_RES_CMAX = 0.15

    C_MIN_VP = VELOCITY_MIN
    C_MAX_VP = VELOCITY_MAX
    REL_RESIDUAL_TT_CMAX = 0.05

    C_MIN_SAT = SATURATION_MIN
    C_MAX_SAT = SATURATION_MAX

    # %% Define function to conduct inversion and save results

    tt_normalisation = np.mean(ert_data["rhoa"])/np.mean(tt_data["t"])

    def run_pjpi_inversion(
        tt_weight,
        petrophysical_trust_region=petrophysical_trust_region_on_imesh,
        mesh_info_inversion_mesh=meshinfo_inversion,
        maximum_iterations=MAXIMUM_ITERATIONS,
        max_update_per_step=[
            (-MAX_SATURATION_UPDATE_PER_STEP,MAX_SATURATION_UPDATE_PER_STEP),
            (-MAX_RESISTIVITY_UPDATE_PER_STEP, MAX_RESISTIVITY_UPDATE_PER_STEP),
            (-MAX_SLOWNESS_UPDATE_PER_STEP, MAX_SLOWNESS_UPDATE_PER_STEP),
        ],
        plot=True,
        save=True,
        force_recalculate=False,
        regularisation_weights=[
            (1e0, 1e0),
            (5e1, 1e2),
        ]
    ):
        """Run an joint petrophysical inversion on the prepared mesh and data.

        Parameters
        - `tt_weight`: relative weight for the TT data misfit in the joint inversion. The ERT data misfit weight is fixed to 1, so this parameter controls the balance between the two datasets in the joint inversion.
        - `petrophysical_trust_region`: boolean vector defining petrophysical trust region for decoupled regularisation. True values indicate cells where petrophysical regularisation is applied, False values indicate cells where only geophysical regularisation is applied and the petrophysical and geophysical models are decoupled.
        - `mesh_info_inversion_mesh`: `MeshInfo` instance defining inversion mesh.
        - `maximum_iterations`: maximum GN iterations.
        - `max_update_per_step`: tuple limiting per-iteration model updates.
        - `plot`, `save`: toggle plotting and result saving.
        - `force_recalculate`: ignore cached results when True.
        - `regularisation_weights`: list of (smoothing, damping) tuples for ERT and TT regularisation, respectively. The first tuple applies to the ERT regularisation operators, the second tuple to the TT regularisation operators. The first element of each tuple is the weight for the smoothing operator, the second element the weight for the damping operator.

        Returns
        - `(geophysical_manager, results_dict)` tuple on success.
        """
        results_dict_path = path_data_results.joinpath(f"results_ttweight_{tt_weight}.json")
        inversion_mesh=mesh_info_inversion_mesh.mesh
        if results_dict_path.exists() and not force_recalculate:
            #* Load results
            with open(results_dict_path, "r") as f:
                result_dict = json.load(f)
            logger.info("Results loaded from %s", results_dict_path)
            #* Load final model
            final_mesh_with_model_pjpi = pg.load(str(path_data_results.joinpath(f"final_model_ttweight_{tt_weight}.bms").absolute()))
            #* Load final response
            final_response_ert_pjpi = tt.load(str(path_data_results.joinpath(f"final_response_ert_ttweight_{tt_weight}.data").absolute()))
            final_response_tt_pjpi = tt.load(str(path_data_results.joinpath(f"final_response_tt_ttweight_{tt_weight}.data").absolute()))
            #* Manager set to none
            petrophysical_pjpi_inversion = None

        elif not results_dict_path.exists() or force_recalculate:
            logger.info("Invert on mesh with layer and decouple regularisation")
            decoupled_argument = (decoupled_region_vector, [(1,2)])

            model_transformation_sat_pjpi = mths.transformation.LogarithmicBarrierTransformationTwoSided(
                lower_barrier = SATURATION_MIN,
                upper_barrier = SATURATION_MAX,
            )

            model_transformation_ert_pjpi = mths.transformation.LogarithmicBarrierTransformationTwoSided(
                lower_barrier=RESISTIVITY_MIN,
                upper_barrier=RESISTIVITY_MAX,
            )

            model_transformation_tt_pjpi = mths.transformation.LogarithmicBarrierTransformationTwoSided(
                lower_barrier=1/VELOCITY_MAX,
                upper_barrier=1/VELOCITY_MIN,
            )

            initial_model_pjpi = mths.ModelInfoMixedGeoPetro(
                model_petro=np.ones(np.sum(petrophysical_trust_region)) * initial_model_vector_saturation[0],
                model_list_geo=[
                    np.ones(np.sum(~petrophysical_trust_region)) * archie_reservoir.forward(initial_model_vector_saturation[0]),
                    np.ones(np.sum(~petrophysical_trust_region)) * gassmann_reservoir.forward(initial_model_vector_saturation[0])
                ],
                mesh_info=mesh_info_inversion_mesh,
                transformation_list_petro=[
                    archie_reservoir,
                    gassmann_reservoir
                ],
                inversion_transformation_petro=model_transformation_sat_pjpi,
                inversion_transformation_list_geo=[
                    model_transformation_ert_pjpi,
                    model_transformation_tt_pjpi,
                ],
                petrophysical_trust_region=petrophysical_trust_region
            )

            geophysical_data_ert = pd.pyhsics_and_data_geophysical(
                manager=ert.ERTManager(),
                data_container=ert_data,
                data_observed_field_name="rhoa",
                weight=1.0,
            )

            geophysical_data_tt = pd.pyhsics_and_data_geophysical(
                manager=tt.TravelTimeManager(),
                data_container=tt_data,
                data_observed_field_name="t",
                weight=tt_weight * tt_normalisation,
            )

            smoothing_pjpi_ert = smoothing_operator()
            damping_pjpi_ert = damping_operator()
            smoothing_pjpi_tt = smoothing_operator()
            damping_pjpi_tt = damping_operator()

            # Smoothing is applied on the geophysical models and translated to the inversion domain. This is why we use the parameters extracted from the geophysical-only inversions here.
            # Alternatively regularisation can be applied in the petrophysical domain and geophysical domain separately, but this requires 
            # careful tuning 
            smoothing_pjpi_ert.weight = regularisation_weights[0][0]
            damping_pjpi_ert.weight = regularisation_weights[0][1]
            smoothing_pjpi_tt.weight = regularisation_weights[1][0]
            damping_pjpi_tt.weight = regularisation_weights[1][1]

            petrophysical_pjpi_inversion = ppd.GaussNewtonPetrophysicalDecoupled(
                mesh_info=mesh_info_inversion_mesh,
                petrophysical_trust_region=petrophysical_trust_region,
                geophysical_data_list=[
                    geophysical_data_ert,
                    geophysical_data_tt
                    ],
                initial_model=initial_model_pjpi,
                single_model_regularisation=[
                    (smoothing_pjpi_ert, damping_pjpi_ert),
                    (smoothing_pjpi_tt, damping_pjpi_tt),
                    ],
                decouple_regularisation_regions=None, # decoupling happens already through the petrophysical_trust_region and the decoupled region vector, so no need to add an additional operator here
                maximum_iterations=maximum_iterations,
                verbose=True,
                inversion_settings={
                    "decouple_regularisation_trustregion": True,
                    "domain": "petrophysical",
                    "add_xg_for_untrusted_region": False,
                    "update_petro_trust_region": False,
                    "fix_coupled_region": np.array([True] * inversion_mesh.cellCount()),
                    "individual_updates_with_xg": False,
                    "minimum_petro_component_size": 0.0,
                },
            )
            petrophysical_pjpi_inversion.maximum_update_per_step = max_update_per_step
            petrophysical_pjpi_inversion.terminate_on_chi2_decrease = 0.01
            petrophysical_pjpi_inversion.num_solver = "scipy_sparse"
            # Conditioning of the system can be a problem - play around with scaling and preconditioners to improve results.
            # The alternative approach used here is to stabilise the inversion by introducing rather restrictive clipping
            
            # petrophysical_pjpi_inversion.scaling="none"
            # petrophysical_pjpi_inversion.scaling="column_max"
            petrophysical_pjpi_inversion.run()

            #* Prepare final model
            final_mesh_with_model_pjpi = inversion_mesh.copy()
            petrophysical_model_vector_pjpi = np.ones(inversion_mesh.cellCount()) * np.nan
            petrophysical_model_vector_pjpi[petrophysical_pjpi_inversion.petrophysical_trust_region] = petrophysical_pjpi_inversion.current_model.model_petro
            final_mesh_with_model_pjpi["sat"] = petrophysical_model_vector_pjpi
            final_mesh_with_model_pjpi["res"] = petrophysical_pjpi_inversion.current_model.model_list_geo[0]
            final_mesh_with_model_pjpi["vp"] = 1/petrophysical_pjpi_inversion.current_model.model_list_geo[1]

            ert_man_final = ert.ERTManager()
            ert_man_final.setMesh(mesh=inversion_mesh)
            ert_man_final.setData(data=ert_data)

            tt_man_final = tt.TravelTimeManager()
            tt_man_final.setMesh(mesh=inversion_mesh)
            tt_man_final.setData(data=tt_data)

            final_response_ert_pjpi_vector = ert_man_final.fop.response(final_mesh_with_model_pjpi["res"])
            final_response_ert_pjpi = ert_data.copy()
            final_response_ert_pjpi["rhoa"] = final_response_ert_pjpi_vector
            
            final_response_tt_pjpi_vector = tt_man_final.fop.response(1/final_mesh_with_model_pjpi["vp"])
            final_response_tt_pjpi = tt_data.copy()
            final_response_tt_pjpi["t"] = final_response_tt_pjpi_vector

            rel_error_ert_pjpi = np.linalg.norm(np.array(ert_data["rhoa"])-np.array(final_response_ert_pjpi["rhoa"]))/np.linalg.norm(np.array(ert_data["rhoa"]))
            chi2_ert_pjpi = ph.data_to_chi_squared(ert_data, final_response_ert_pjpi, "rhoa")

            logger.info("ERT: Relative error: %s, for smoothing %s and damping %s", rel_error_ert_pjpi, regularisation_weights[0][0], regularisation_weights[0][1])
            logger.info("ERT: Chi squared: %s for smoothing %s and damping %s", chi2_ert_pjpi, regularisation_weights[0][0], regularisation_weights[0][1])

            rel_error_tt_pjpi = np.linalg.norm(np.array(tt_data["t"])-np.array(final_response_tt_pjpi["t"]))/np.linalg.norm(np.array(tt_data["t"]))
            chi2_tt_pjpi = ph.data_to_chi_squared(tt_data, final_response_tt_pjpi, "t")

            logger.info("TT: Relative error: %s, for smoothing %s and damping %s", rel_error_tt_pjpi, regularisation_weights[1][0], regularisation_weights[1][1])
            logger.info("TT: Chi squared: %s for smoothing %s and damping %s", chi2_tt_pjpi, regularisation_weights[1][0], regularisation_weights[1][1])

            data_misfit = [petrophysical_pjpi_inversion.tracking_dict[iteration]["data_misfit"] for iteration in range(0, petrophysical_pjpi_inversion.maximum_iterations+1)]
            chi_squared_history = [
                petrophysical_pjpi_inversion.tracking_dict[iteration]["chi_squared"] for iteration in range(0, petrophysical_pjpi_inversion.maximum_iterations+1)
            ]
            single_model_regularisation_misfit = [petrophysical_pjpi_inversion.tracking_dict[iteration]["single_model_regularisation_misfit"] for iteration in range(0, petrophysical_pjpi_inversion.maximum_iterations+1)]

            if save:
                final_response_ert_pjpi.save(str(path_data_results.joinpath(f"final_response_ert_ttweight_{tt_weight}.data").absolute()))
                final_response_tt_pjpi.save(str(path_data_results.joinpath(f"final_response_tt_ttweight_{tt_weight}.data").absolute()))
                final_mesh_with_model_pjpi.save(
                    str(path_data_results.joinpath(
                    f"final_model_ttweight_{tt_weight}.bms"
                    ).absolute())
                    )

            result_dict = {
                "inversion_name": "pjpi",
                "inversion_domain": "petro",
                "joint_inversion": False,
                "rel_error_ert": rel_error_ert_pjpi,
                "rel_error_tt": rel_error_tt_pjpi,
                "final_chi2_ert": chi2_ert_pjpi,
                "final_chi2_tt": chi2_tt_pjpi,
                "smoothing_ert": regularisation_weights[0][0],
                "damping_ert": regularisation_weights[0][1],
                "smoothing_tt": regularisation_weights[1][0],
                "damping_tt": regularisation_weights[1][1],
                "iterations": petrophysical_pjpi_inversion.maximum_iterations,
                "data_misfit": data_misfit,
                "chi_squared_history": chi_squared_history,
                "single_model_regularisation_misfit": single_model_regularisation_misfit,
            }

            if save:
                with open(results_dict_path.absolute(), "w") as f:
                    json.dump(
                        result_dict,
                        f,
                        indent=4,
                        )
        else:
            return None
        
        #* Plotting results
        if plot:
            #* Plot final models
            fig, axs = plt.subplots(1, 6, figsize=(25, 8), layout="constrained")
            _=pg.show(original_mesh_w_models, data="sat", cMap=CMAP, logScale=logScale, cMin=C_MIN_SAT, cMax=C_MAX_SAT, ax=axs[0], label="Saturation")
            _=pg.show(inversion_mesh, data=final_mesh_with_model_pjpi["sat"], cMap=CMAP, logScale=logScale, cMin=C_MIN_SAT, cMax=C_MAX_SAT, ax=axs[1], label="Saturation")
            _=pg.show(original_mesh_w_models, data="res", cMap=CMAP, logScale=logScale, cMin=C_MIN_RES, cMax=C_MAX_RES, ax=axs[2], label="Resistivity")
            _=pg.show(inversion_mesh, data=final_mesh_with_model_pjpi["res"], cMap=CMAP, logScale=logScale, cMin=C_MIN_RES, cMax=C_MAX_RES, ax=axs[3], label="Resistivity")
            _=pg.show(original_mesh_w_models, data="vp", cMap=CMAP, logScale=logScale, cMin=C_MIN_VP, cMax=C_MAX_VP, ax=axs[4], label="Velocity")
            _=pg.show(inversion_mesh, data=final_mesh_with_model_pjpi["vp"], cMap=CMAP, logScale=logScale, cMin=C_MIN_VP, cMax=C_MAX_VP, ax=axs[5], label="Velocity")

            for ax in axs:
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_xlim(DOMAIN_XMIN, DOMAIN_XMAX)
                ax.set_ylim(DOMAIN_YMIN, DOMAIN_YMAX)
                plot_boreholes_on_mesh(ax)

            axs[0].set_title("Original model saturation")
            axs[1].set_title(f"Final saturation model PJPI with TT weight {tt_weight}")
            axs[2].set_title("Original model resistivity")
            axs[3].set_title(f"Final resistivity model PJPI with TT weight {tt_weight}")
            axs[4].set_title("Original model velocity")
            axs[5].set_title(f"Final velocity model PJPI with TT weight {tt_weight}")
            if save:
                fig.savefig(
                    str(path_figures_results.joinpath(f"final_models_ttweight_{tt_weight}.jpg").absolute()),
                    format='jpg',
                    dpi=300,
                    bbox_inches='tight'
                    )
                plt.close(fig)
            
            #* Plot misfit history
            fig, ax = ph.plot_misfits_from_results_dict(
                results_dict=result_dict,
                fields_to_plot=["data", "single"]
            )
            ax.set_title("Misfit history")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Misfit")
            ax.set_yscale("log")
            if save:
                fig.savefig(
                    str(path_figures_results.joinpath(f"misfit_history_ttweight_{tt_weight}.jpg").absolute()),
                    format='jpg',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close(fig)

            #* Plot final misfits - TT
            residual_container_ert = ert_data.copy()
            residual_container_ert["rhoa"] = np.abs(final_response_ert_pjpi["rhoa"] - ert_data["rhoa"]) / np.abs(ert_data["rhoa"])
            data_matrices, offsets = ph.gather_datamatrices_by_offset(residual_container_ert)
            fig, axs = ph.plot_datamatrices_by_offset(
                data_matrices=data_matrices[3:],
                offsets=offsets[3:],
                cMap="turbo",
                cMin=0.0,
                cMax=REL_RESIDUAL_RES_CMAX,
                figsize=(20, 5),
                layout='constrained',
            )
            fig.suptitle(f"Relative residuals by offset for ERT PJPI with TT weight {tt_weight}")
            if save:
                fig.savefig(str(path_figures_results.joinpath(f"final_residual_ert_ttweight_{tt_weight}.jpg").absolute()), format='jpg', dpi=300, bbox_inches='tight')
                plt.close(fig)

            #* Plot final misfits - TT
            residual_container_tt = tt_data.copy()
            residual_container_tt["t"] = np.abs(final_response_tt_pjpi["t"] - tt_data["t"]) / np.abs(tt_data["t"])
            fig, ax = ph.plot_apparent_velocities_from_data(
                data_tt = residual_container_tt,
                field="t",
                cMin=0.0,
                cMax=REL_RESIDUAL_TT_CMAX,
                cMap="turbo",
            )
            ax.set_title(f"Relative residuals TT PJPI with TT weight {tt_weight}")
            if save:
                fig.savefig(str(path_figures_results.joinpath(f"final_residual_tt_ttweight_{tt_weight}.jpg").absolute()), format='jpg', dpi=300, bbox_inches='tight')
                plt.close(fig)

            return petrophysical_pjpi_inversion, result_dict

    # Run a default inversion when invoked as a script.
    for tt_weight in [.1, 1, 10]:
        run_pjpi_inversion(
            tt_weight=tt_weight,
        )


if __name__ == "__main__":
    main()