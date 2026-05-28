"""Build cross-hole example geometry and PLCs used by downstream examples.

This script constructs a layered domain with a CO2 plume and boundary PLCs,
exports illustrative figures to `setup_figures/` and supplies helper
functions for mesh creation and sensor placement used by the examples.

Edit only the geometry parameters near the top of the file; helper functions
below encapsulate the PLC and mesh creation logic.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.viewer.mpl import drawMesh, drawPLC, drawBoundaryMarkers

import scipy as sP
from pygimli.physics import ert
from pygimli.physics import traveltime as tt

from petrophysics import Transformation, GassmannTransformation, ArchieTransformation
from parsing import parse_mesh_to_saturation_model, parse_mesh_to_resistivity_model, parse_mesh_to_vp_model
from plotting_helpers import gather_datamatrices_by_offset, plot_datamatrices_by_offset, plot_apparent_velocities_from_data

# Use non-interactive backend for scripted figure export
matplotlib.use("Agg")


# %%
#* Set up folder structure

path_home = Path(__file__).resolve().parent
path_data = path_home / "data"
setup_figures = path_home / "setup_figures"

path_home.mkdir(parents=True, exist_ok=True)
path_data.mkdir(parents=True, exist_ok=True)
setup_figures.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Set up survey parameters

# %% [markdown]
# ## Geometry Parameters
# %%
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

sensor_positions_A_y = np.linspace(BOREHOLE_A_Y_MAX, BOREHOLE_A_Y_MIN, ELECTRODE_NUMBER)
sensor_positions_B_y = np.linspace(BOREHOLE_B_Y_MAX, BOREHOLE_B_Y_MIN, ELECTRODE_NUMBER)

# %%
def plot_boreholes_on_mesh(ax):
    """Draw borehole location markers/lines on an axes object (non-blocking).

    This function only draws on the supplied axes. Do not call it when an
    interactive plot is required; the caller is responsible for saving or
    showing the figure.
    """
    ax.plot([BOREHOLE_A_X, BOREHOLE_A_X], [BOREHOLE_A_Y_MIN, BOREHOLE_A_Y_MAX], color="black", linestyle="--")
    ax.plot([BOREHOLE_B_X, BOREHOLE_B_X], [BOREHOLE_B_Y_MIN, BOREHOLE_B_Y_MAX], color="black", linestyle="--")

# %% [markdown]
# ## MARKER Parametrisation
# %%
MARKER_CAPROCK = 0
MARKER_SANDSTONE = 1
MARKER_CO2 = 2
MARKER_BOUNDARY = 3
# EMPTY = 0

# %% [markdown]
# ## Create Geometry
# %%
DOMAIN_EXTENT_X = 2 * BUFFER_X + BOREHOLE_DISTANCE
DOMAIN_XMIN = 0
DOMAIN_XMAX = DOMAIN_EXTENT_X

DOMAIN_YMIN = -(2 * BUFFER_Y + length_boreholes)
DOMAIN_YMAX = 0

domain = mt.createWorld(start=[DOMAIN_XMIN, DOMAIN_YMIN], end=[DOMAIN_XMAX, DOMAIN_YMAX], marker=MARKER_CAPROCK, worldMarker=False)

height_layer_left = DOMAIN_YMAX - (LAYER_FROM_TOP_A - inclination * BUFFER_X)
height_layer_right = DOMAIN_YMAX - (LAYER_FROM_TOP_B + inclination * BUFFER_X)

height_layer_left_with_buffer = DOMAIN_YMAX - (LAYER_FROM_TOP_A - inclination * (BUFFER_X+BUFFER_TRIANGLE_BOUNDARY))
height_layer_right_with_buffer = DOMAIN_YMAX - (LAYER_FROM_TOP_B + inclination * (BUFFER_X+BUFFER_TRIANGLE_BOUNDARY))

# %%
x_vec = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, 20)
y_vec = height_layer_left + -(inclination) * (x_vec - (DOMAIN_XMIN))
verts_line = np.array([x_vec, y_vec]).T
layer_boundary_plc = mt.createPolygon(verts_line, isClosed=False, marker=MARKER_BOUNDARY, boundaryMarker=0)
domain_with_layer = mt.mergePLC([domain, layer_boundary_plc])

fig, ax = plt.subplots()

gci = drawPLC(
    ax=ax,
    mesh=domain_with_layer,
    markers=True,
    showMesh=True,
    colorBar=False,
)
plot_boreholes_on_mesh(ax)
_=ax.set_title("Domain with layer")
fig.savefig(setup_figures / "domain_with_layer.jpg", format='jpg', dpi=300, bbox_inches='tight')


# %%
# Model CO2 plume as quadratic function with distance from borehole
# Conditions: ax^2 + by + c
# a < 0 (concave parabola), apex at borehole midpoint
# c = layer height at borehole midpoint
# Form: p(x) = a (x - x0)^2 + y0


CO2_THICKNESS = 15
PLUME_SHAPE_PARAMETER = -10e-2

borehole_x_middle = BOREHOLE_A_X + BOREHOLE_DISTANCE / 2
layer_height_middle = (height_layer_left + height_layer_right) / 2 - CO2_THICKNESS
# Shape parameter for the quadratic plume profile (negative -> concave)

def lower_co2_plume_boundary(x, borehole="A"):
    """Quadratic lower boundary of the CO2 plume referenced to a borehole.

    Parameters
    - x: array_like or float, x-coordinate(s)
    - borehole: 'A' or 'B' selects reference borehole for horizontal offset
    """
    if borehole == "A":
        x_dist = np.abs(x - BOREHOLE_A_X)
        x_shift = np.abs(borehole_x_middle - BOREHOLE_A_X)
    elif borehole == "B":
        x_dist = np.abs(x - BOREHOLE_B_X)
        x_shift = np.abs(borehole_x_middle - BOREHOLE_B_X)
    else:
        raise ValueError("Borehole must be either 'A' or 'B'")

    # Parabolic profile around the borehole midpoint with tilt from inclination
    plume_profile = PLUME_SHAPE_PARAMETER * (x_dist - x_shift) ** 2
    plume_profile += layer_height_middle - inclination * (x - borehole_x_middle)
    return plume_profile

EXTEND_LAYER_OUTSIDE_PLUME = True
co2_plume_lower_boundary = lower_co2_plume_boundary

if EXTEND_LAYER_OUTSIDE_PLUME:
    x_params = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, 50)
    y_params_A = co2_plume_lower_boundary(x_params, borehole="A")
    y_params_B = co2_plume_lower_boundary(x_params, borehole="B")
    y_params = np.zeros_like(x_params)
    y_params[x_params < borehole_x_middle] = y_params_A[x_params < borehole_x_middle]
    y_params[x_params >= borehole_x_middle] = y_params_B[x_params >= borehole_x_middle]

    LEFT_MAXIMUM = 5
    RIGHT_MAXIMUM = 105

    x_left_out = x_params[x_params < LEFT_MAXIMUM]
    x_right_out = x_params[x_params > RIGHT_MAXIMUM]
    y_params[x_params < LEFT_MAXIMUM] = layer_height_middle - inclination * (x_left_out-borehole_x_middle)
    y_params[x_params > RIGHT_MAXIMUM] = layer_height_middle - inclination * (x_right_out-borehole_x_middle)

# Plot intermediate PLCs: reservoir, CO2 plume and caprock
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
# Create reservoir plc
verts_reservoir = [
    [x, y] for x, y in zip(x_params, y_params)
]
verts_reservoir = verts_reservoir + [[DOMAIN_XMAX, DOMAIN_YMIN], [DOMAIN_XMIN, DOMAIN_YMIN]]
plc_reservoir = mt.createPolygon(
    verts_reservoir, isClosed=True, marker=MARKER_SANDSTONE, boundaryMarker=0, markerPosition=[borehole_x_middle, DOMAIN_YMIN+10]
)

gci_reservoir = drawPLC(
    ax=axs[0],
    mesh=plc_reservoir,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[0].set_title("Reservoir")

# Create CO2 plume plc
verts_co2 = [
    [x, y] for x, y in zip(x_params, y_params)
]
verts_co2 = verts_co2 + [[DOMAIN_XMAX, height_layer_right], [DOMAIN_XMIN, height_layer_left]]
plc_co2 = mt.createPolygon(
    verts_co2, isClosed=True, marker=MARKER_CO2, boundaryMarker=0, markerPosition=[BOREHOLE_A_X, height_layer_left]
)
_=drawPLC(
    ax=axs[1],
    mesh=plc_co2,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[1].set_title("CO2 Plume")

# Create caprock plc
verts_caprock = [
        [DOMAIN_XMIN, DOMAIN_YMAX],
        [DOMAIN_XMAX, DOMAIN_YMAX],
        [DOMAIN_XMAX, height_layer_right],
        [DOMAIN_XMIN, height_layer_left],
]
plc_caprock = mt.createPolygon(
    verts_caprock, isClosed=True, marker=MARKER_CAPROCK, boundaryMarker=0, markerPosition=[borehole_x_middle, DOMAIN_YMAX-10]
)
_=drawPLC(
    ax=axs[2],
    mesh=plc_caprock,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[2].set_title("Caprock")#; plot_boreholes_on_mesh(axs[2])

for ax in axs:
    plot_boreholes_on_mesh(ax)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_aspect("equal")
    ax.set_xlim(DOMAIN_XMIN-BUFFER_X, DOMAIN_XMAX+BUFFER_X)
    ax.set_ylim(DOMAIN_YMIN-BUFFER_Y, DOMAIN_YMAX+BUFFER_Y)

fig.savefig(setup_figures / "intermediate_plcs.jpg", format='jpg', dpi=300, bbox_inches='tight')

#* Create boundary plcs
fig, axs = plt.subplots(1, 6, figsize=(25,5))
verts_caprock_boundaries = [
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, DOMAIN_YMAX + BUFFER_TRIANGLE_BOUNDARY],
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, DOMAIN_YMAX + BUFFER_TRIANGLE_BOUNDARY],
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, height_layer_right_with_buffer],
    [DOMAIN_XMAX, height_layer_right],
    [DOMAIN_XMAX, DOMAIN_YMAX],
    [DOMAIN_XMIN, DOMAIN_YMAX],
    [DOMAIN_XMIN, height_layer_left],
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, height_layer_left_with_buffer],
]
plc_caprock_boundary = mt.createPolygon(
    verts_caprock_boundaries, isClosed=True, marker=MARKER_CAPROCK, boundaryMarker=0
)
_=drawPLC(
    ax=axs[0],
    mesh=plc_caprock_boundary,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[0].set_title("Caprock Boundary")
_=plot_boreholes_on_mesh(axs[0])

verts_reservoir_boundaries = [
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, height_layer_left_with_buffer-CO2_THICKNESS],
    [DOMAIN_XMIN, height_layer_left-CO2_THICKNESS],
    [DOMAIN_XMIN, DOMAIN_YMIN],
    [DOMAIN_XMAX, DOMAIN_YMIN],
    [DOMAIN_XMAX, height_layer_right-CO2_THICKNESS],
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, height_layer_right_with_buffer-CO2_THICKNESS],
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, DOMAIN_YMIN-BUFFER_TRIANGLE_BOUNDARY],
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, DOMAIN_YMIN-BUFFER_TRIANGLE_BOUNDARY],
]
plc_reservoir_boundary = mt.createPolygon(
    verts_reservoir_boundaries, isClosed=True, marker=MARKER_SANDSTONE, boundaryMarker=0
)
_=drawPLC(
    ax=axs[1],
    mesh=plc_reservoir_boundary,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[1].set_title("Reservoir Boundary")
_=plot_boreholes_on_mesh(axs[1])

verts_co2_boundaries_left = [
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, height_layer_left_with_buffer],
    [DOMAIN_XMIN, height_layer_left],
    [DOMAIN_XMIN, height_layer_left-CO2_THICKNESS],
    [DOMAIN_XMIN-BUFFER_TRIANGLE_BOUNDARY, height_layer_left_with_buffer-CO2_THICKNESS],
]

verts_co2_boundaries_right = [
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, height_layer_right_with_buffer],
    [DOMAIN_XMAX, height_layer_right],
    [DOMAIN_XMAX, height_layer_right-CO2_THICKNESS],
    [DOMAIN_XMAX+BUFFER_TRIANGLE_BOUNDARY, height_layer_right_with_buffer-CO2_THICKNESS],
]

plc_co2_boundary_left = mt.createPolygon(
    verts_co2_boundaries_left, isClosed=True, marker=MARKER_CO2, boundaryMarker=0
)
plc_co2_boundary_right = mt.createPolygon(
    verts_co2_boundaries_right, isClosed=True, marker=MARKER_CO2, boundaryMarker=0
)
plc_co2_boundaries = mt.mergePLC([plc_co2_boundary_left, plc_co2_boundary_right])

_=drawPLC(
    ax=axs[2],
    mesh=plc_co2_boundary_left,
    markers=True,
    showMesh=True,
    colorBar=False,
)
_ = drawPLC(
    ax=axs[2],
    mesh=plc_co2_boundary_right,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[2].set_title("CO2 Boundary")
_=plot_boreholes_on_mesh(axs[2])

#* Create full domain without boundary
plc_boundaries = mt.mergePLC([plc_caprock_boundary, plc_reservoir_boundary, plc_co2_boundaries])
domain_with_co2_plume = mt.mergePLC([plc_caprock, plc_reservoir, plc_co2])
domain_with_boundaries = mt.mergePLC([domain_with_co2_plume, plc_boundaries])

_ = drawPLC(
    ax=axs[3],
    mesh=plc_boundaries,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[3].set_title("Boundary PLCs")
plot_boreholes_on_mesh(axs[3])

_ = drawPLC(
    ax=axs[4],
    mesh=domain_with_co2_plume,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[4].set_title("Full Domain without Boundaries")
plot_boreholes_on_mesh(axs[4])

_ = drawPLC(
    ax=axs[5],
    mesh=domain_with_boundaries,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[5].set_title("Full Domain with Boundaries")
plot_boreholes_on_mesh(axs[5])

for ax in axs:
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_aspect("equal")
    ax.set_xlim(DOMAIN_XMIN-BUFFER_X, DOMAIN_XMAX+BUFFER_X)
    ax.set_ylim(DOMAIN_YMIN-BUFFER_Y, DOMAIN_YMAX+BUFFER_Y)

fig.savefig(setup_figures / "boundary_plcs_and_full_domain.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %%
#* Plot geometry overview with bright colors and without markers to make it more visually appealing for presentation
domain_with_co2_plume_to_plot = domain_with_co2_plume.copy()
domain_with_co2_plume_to_plot = mt.createMesh(domain_with_co2_plume_to_plot, quality=34, area=1000)

# Define a discrete colormap with 3 colors
cmap = mcolors.ListedColormap(['red', 'green', 'blue'])
fig, ax = plt.subplots(1,1, figsize=(4, 8))
ax, cbar = ret = pg.show(domain_with_co2_plume_to_plot, data=domain_with_co2_plume_to_plot.cellMarkers(), markers=False, cMap=cmap, ax=ax)
cbar.remove()
_=ax.set_ylabel("Depth [m]")
_=ax.set_xlabel("Distance [m]")
plot_boreholes_on_mesh(ax)
fig.savefig(setup_figures / "geometry_colored.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %%
def add_boundary_mesh(mesh, boundary_area=AREA_TRIANGLE_BOUNDARY, quality=33, set_boundary_marker=True):
    """Merge interior mesh with boundary meshes and set boundary markers.

    Parameters
    - mesh: interior mesh to which boundary meshes will be attached.
    - boundary_area: target triangle area for boundary mesh generation.
    - quality: quality parameter forwarded to pygimli meshing.
    - set_boundary_marker: if True, reset and set boundary markers

    Returns
    - out_mesh: merged mesh (interior + boundaries)
    - list of boundary meshes [reservoir, co2, caprock, merged]
    """
    mesh_internal = mesh.copy()
    if set_boundary_marker:
        for boundary in mesh_internal.boundaries():
            boundary.setMarker(0)

    boundary_mesh_reservoir = mt.createMesh(plc_reservoir_boundary, quality=quality, area=boundary_area)
    boundary_mesh_co2 = mt.createMesh(plc_co2_boundaries, quality=quality, area=boundary_area)
    boundary_mesh_caprock = mt.createMesh(plc_caprock_boundary, quality=quality, area=boundary_area)
    boundary_mesh = mt.mergeMeshes([boundary_mesh_reservoir, boundary_mesh_co2, boundary_mesh_caprock])
    out_mesh = mt.mergeMeshes([mesh_internal, boundary_mesh])

    # Set outermost boundary markers to -2, internal boundaries to 0
    def boundary_outside(boundary):
        return (
            boundary.center().x() == DOMAIN_XMIN - BUFFER_TRIANGLE_BOUNDARY
            or boundary.center().x() == DOMAIN_XMAX + BUFFER_TRIANGLE_BOUNDARY
            or boundary.center().y() == DOMAIN_YMIN - BUFFER_TRIANGLE_BOUNDARY
            or boundary.center().y() == DOMAIN_YMAX + BUFFER_TRIANGLE_BOUNDARY
        )

    if set_boundary_marker:
        for boundary in out_mesh.boundaries():
            if boundary_outside(boundary):
                boundary.setMarker(-2)
            else:
                boundary.setMarker(0)

    return out_mesh, [boundary_mesh_reservoir, boundary_mesh_co2, boundary_mesh_caprock, boundary_mesh]

## Add utility functions for outer mesh generation and sensor placement

def add_sensor_positions_to_plc(
    plc,
    sensor_positions,
    refinement=0.2,
    minimum_distance=0.2,
    alsoupper=True,
    direction="y",
):
    """Insert sensor nodes into a PLC while respecting a minimum spacing.

    Parameters
    - plc: PLC-like object (pygimli PLC) to which nodes will be added.
    - sensor_positions: sequence of (x,y) positions for sensors.
    - refinement: fraction of electrode spacing used to offset duplicated nodes.
    - minimum_distance: minimum allowed distance (in units of electrode spacing).
    - alsoupper: if True, also add an upper-shifted node for each sensor.
    - direction: 'x' or 'y' indicating primary spacing direction.

    Returns
    - plc_temp: copy of the input PLC with added nodes.
    """
    plc_temp = plc.copy()
    skipped_count = 0
    nodes_prior = plc_temp.nodeCount()

    for node in sensor_positions:
        # Get all existing node positions (2D)
        node_positions = np.array([np.array(node_temp.pos())[:2] for node_temp in plc_temp.nodes()])

        # Create a node at the sensor location if not already present
        plc_temp.createNodeWithCheck(node, edgeCheck=True)

        # Define shift vector based on specified direction
        if direction == "x":
            shift_vector = np.array([refinement * ELECTRODE_SPACING, 0])
        elif direction == "y":
            shift_vector = np.array([0, refinement * ELECTRODE_SPACING])
        else:
            raise ValueError("Direction must be either 'x' or 'y'")

        # Try adding a lower-shifted node if spacing allows
        node_to_add = np.array(node)[:2] - shift_vector
        distances = np.linalg.norm(node_positions - node_to_add[:2], axis=1, ord=1)

        if np.min(distances) <= (minimum_distance * ELECTRODE_SPACING):
            skipped_count += 1
        else:
            plc_temp.createNodeWithCheck(node_to_add, edgeCheck=True)

        # Optionally add an upper-shifted node as well
        if alsoupper:
            node_to_add = np.array(node)[:2] + shift_vector
            distances = np.linalg.norm(node_positions - node_to_add[:2], axis=1, ord=1)
            if np.min(distances) <= (minimum_distance * ELECTRODE_SPACING):
                skipped_count += 1
            else:
                plc_temp.createNodeWithCheck(node_to_add, edgeCheck=True)

    print(f"Skipped {skipped_count} nodes due to minimum distance")
    nodes_after = plc_temp.nodeCount()
    print(f"Added {nodes_after - nodes_prior} nodes")
    return plc_temp

## Create test meshes for testing the boundary mesh generation and sensor position addition

# %%
inner_mesh_test = pg.meshtools.createMesh(domain_with_co2_plume, quality=33, area=20)
inner_mesh_test, boundary_meshes_test = add_boundary_mesh(inner_mesh_test)
boundary_mesh_reservoir, boundary_mesh_co2, boundary_mesh_caprock, boundary_mesh = boundary_meshes_test

fig, axs = plt.subplots(1, 6, figsize=(25, 5))

_ = drawMesh(
    ax=axs[0],
    mesh=inner_mesh_test,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[0].set_title("Full Mesh - interior")
plot_boreholes_on_mesh(axs[0])

_ = drawMesh(
    ax=axs[1],
    mesh=boundary_mesh_reservoir,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[1].set_title("Reservoir Boundary")
plot_boreholes_on_mesh(axs[1])

_ = drawMesh(
    ax=axs[2],
    mesh=boundary_mesh_co2,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[2].set_title("CO2 Boundary")
plot_boreholes_on_mesh(axs[2])

_ = drawMesh(
    ax=axs[3],
    mesh=boundary_mesh_caprock,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[3].set_title("Caprock Boundary")
plot_boreholes_on_mesh(axs[3])

_ = drawMesh(
    ax=axs[4],
    mesh=boundary_mesh,
    markers=True,
    showMesh=True,
    colorBar=False,
)
axs[4].set_title("All Boundaries")
plot_boreholes_on_mesh(axs[4])

_ = pg.show(inner_mesh_test, markers=True, showMesh=True, colorBar=False, fillRegions=True, ax=axs[5])
axs[5].set_title("Full Mesh - interior + boundaries")
plot_boreholes_on_mesh(axs[5])

for ax in axs:
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_aspect("equal")
    ax.set_xlim(DOMAIN_XMIN-BUFFER_X, DOMAIN_XMAX+BUFFER_X)
    ax.set_ylim(DOMAIN_YMIN-BUFFER_Y, DOMAIN_YMAX+BUFFER_Y)

fig.savefig(setup_figures / "test_boundary_meshes.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% [markdown]
# Create acquisitions setup


# %% [markdown]
##  ERT scheme

ert_scheme = pg.DataContainerERT()
sensor_A = np.array([[BOREHOLE_A_X, sensor_positions_A_y_temp] for sensor_positions_A_y_temp in sensor_positions_A_y])
sensor_B = np.array([[BOREHOLE_B_X, sensor_positions_B_y_temp] for sensor_positions_B_y_temp in sensor_positions_B_y])
for sen in sensor_A:
    ert_scheme.createSensor(sen)
for sen in sensor_B:
    ert_scheme.createSensor(sen)

A=[]
B=[]

M=[]
N=[]

electrode_difference_list = [-3, -2, -1, 1, 2, 3]

for current_1_num, current_1_sens in enumerate(sensor_A):
    for current_2_num, current_2_sens in enumerate(sensor_B):
        a_temp = current_1_num
        b_temp = current_2_num

        for electrode_difference in electrode_difference_list:
            m_temp = a_temp + electrode_difference
            n_temp = b_temp + electrode_difference

            if m_temp >= 0 and n_temp >= 0 and m_temp < ELECTRODE_NUMBER and n_temp < ELECTRODE_NUMBER:
                A.append(a_temp)
                B.append(b_temp+ELECTRODE_NUMBER)
                M.append(m_temp)
                N.append(n_temp+ELECTRODE_NUMBER)
                # print(f"m: {m_temp}, n: {n_temp}, a: {a_temp}, b: {b_temp}")

sensor_positions = ert_scheme.sensorPositions()


ert_scheme.resize(len(M))
ert_scheme["m"] = M
ert_scheme["n"] = N
ert_scheme["a"] = A
ert_scheme["b"] = B
ert_scheme["valid"] = np.ones_like(M)
ert_scheme["k"] = np.ones_like(M)
ert_scheme.registerSensorIndex("m")
ert_scheme.registerSensorIndex("n")
ert_scheme.registerSensorIndex("a")
ert_scheme.registerSensorIndex("b")

domain_with_co2_plume_and_sensors = add_sensor_positions_to_plc(domain_with_co2_plume, sensor_positions, refinement=0.2)
k_factor_mesh = pg.meshtools.createMesh(domain_with_co2_plume_and_sensors, quality=33, area=10)
k_factor_mesh, _ = add_boundary_mesh(k_factor_mesh)

fig, ax = plt.subplots(1, 1, figsize=(4, 8))
print("hi")
drawMesh(
    ax=ax,
    mesh=k_factor_mesh,
    markers=True,
    showMesh=True,
    colorBar=False,
    boundaryMarkers=True,
)
drawBoundaryMarkers(
    ax=ax,
    mesh=k_factor_mesh,
)
ax.set_title("Mesh for geometric factor calculation")
plot_boreholes_on_mesh(ax)
fig.savefig(setup_figures / "k_factor_mesh.jpg", format='jpg', dpi=300, bbox_inches='tight')
print("there")
ert_scheme["k"]=ert.createGeometricFactors(ert_scheme, numerical=True, mesh=k_factor_mesh, verbose=True)

ert_scheme.save(str(path_data / "ert_scheme.dat"))

# %% [markdown]
##  TT scheme
tt_scheme = pg.DataContainer()
sensor_A = np.array([[BOREHOLE_A_X, sensor_positions_A_y_temp] for sensor_positions_A_y_temp in sensor_positions_A_y])
sensor_B = np.array([[BOREHOLE_B_X, sensor_positions_B_y_temp] for sensor_positions_B_y_temp in sensor_positions_B_y])
for sen in sensor_A:
    tt_scheme.createSensor(sen)
for sen in sensor_B:
    tt_scheme.createSensor(sen)

one_two_way = "one"
if one_two_way == "one":
    number_of_rays = SEISMIC_STATION_NUMBER **2
    tt_scheme.resize(number_of_rays)
elif one_two_way == "two":
    number_of_rays = SEISMIC_STATION_NUMBER **2 * 2
    tt_scheme.resize(number_of_rays)
else:
    raise ValueError("One or two way travel time")

offset=0
source_indices = np.arange(SEISMIC_STATION_NUMBER)+offset
receiver_indices = np.arange(SEISMIC_STATION_NUMBER)+SEISMIC_STATION_NUMBER+offset

source_indices_long = np.repeat(source_indices, SEISMIC_STATION_NUMBER)
receiver_indices_long = np.tile(receiver_indices, SEISMIC_STATION_NUMBER)

if one_two_way == "two":
    source_indices_long = np.append(source_indices_long, receiver_indices_long)
    receiver_indices_long = np.append(receiver_indices_long, source_indices_long)

tt_scheme["s"] = source_indices_long
tt_scheme["g"] = receiver_indices_long
tt_scheme["valid"] = np.ones_like(source_indices_long)
tt_scheme.registerSensorIndex("s")
tt_scheme.registerSensorIndex("g")

tt_scheme.save(str(path_data / "tt_scheme.dat"))

# %% [markdown]
##  Plot acquisition geometry
fig, axs = plt.subplots(1, 2, figsize=(10, 10))

a = np.array(ert_scheme["a"])
b = np.array(ert_scheme["b"])
m = np.array(ert_scheme["m"])
n = np.array(ert_scheme["n"])
sensor_positions_ert = ert_scheme.sensorPositions()

# Find configurations to plot
first_configuration = np.where(
    (a == 1) & (b == 1 + ELECTRODE_NUMBER) & (m == 2) & (n == 2 + ELECTRODE_NUMBER)
)[0][0]
print(f"First configuration indices: {first_configuration}")

second_configuration = np.where(
    (a == 11) & (b == 11 + ELECTRODE_NUMBER) & (m == 13) & (n == 13 + ELECTRODE_NUMBER)
)[0][0]
print(f"Second configuration indices: {second_configuration}")


third_configuration = np.where(
    (a == 21) & (b == 21 + ELECTRODE_NUMBER) & (m == 24) & (n == 24 + ELECTRODE_NUMBER)
)[0][0]
print(f"Third configuration indices: {third_configuration}")

# First plot ERT scheme
configurations_to_plot = [first_configuration, second_configuration, third_configuration]

_ = pg.show(
    domain_with_co2_plume,
    markers=True,
    showMesh=True,
    ax=axs[0]
    )

for num, config in enumerate(configurations_to_plot):
    a_temp = a[config]
    b_temp = b[config]
    m_temp = m[config]
    n_temp = n[config]

    label_current = "Current Electrodes" if num == 0 else None
    label_potential = "Potential Electrodes" if num == 0 else None
    # Plot injection electrode positions
    axs[0].scatter(
        [sensor_positions_ert[a_temp].x(), sensor_positions_ert[b_temp].x()],
        [sensor_positions_ert[a_temp].y(), sensor_positions_ert[b_temp].y()],
        color="red",
        label=label_current,
    )

    # Plot measurement electrode positions
    axs[0].scatter(
        [sensor_positions_ert[m_temp].x(), sensor_positions_ert[n_temp].x()],
        [sensor_positions_ert[m_temp].y(), sensor_positions_ert[n_temp].y()],
        color="blue",
        label=label_potential,
    )

plot_boreholes_on_mesh(axs[0])
axs[0].set_title("ERT Acquisition Geometry")
axs[0].legend()


# Second plot TT scheme
g = np.array(tt_scheme["g"])
s = np.array(tt_scheme["s"])
sensor_positions_tt = tt_scheme.sensorPositions()

shotposition_to_plot = 1

_ = pg.show(
    domain_with_co2_plume,
    markers=True,
    showMesh=True,
    ax=axs[1]
    )

# Plot geophone positions
scheme_indices = np.argwhere(s == shotposition_to_plot).flatten() 
for position_vector in [sensor_positions_tt[g[i]] for i in scheme_indices]:
        axs[1].plot(position_vector.x(), position_vector.y(), 'bo')

# Plot shot position
axs[1].plot(sensor_positions_tt[shotposition_to_plot].x(), sensor_positions_tt[shotposition_to_plot].y(), 'ro')
plot_boreholes_on_mesh(axs[1])

fig.savefig(setup_figures / "acquisition_geometry.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% [markdown]
# Initialise petrophysical parametrisation for Gassmann's equation
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
PHI = 0.28

saturation_vector_co2 = np.linspace(0.01, 0.99, 100)

gassmann_wood = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="wood",
)

gassmann_wood_hs = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="wood",
    version_gassmann="hs",
)

gassmann_domenico = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="domenico",
)

gassmann_domenico_hs = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="domenico",
    version_gassmann="hs",
)

gassmann_brie = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="brie",
)

gassmann_brie_hs = GassmannTransformation(
    k_co2=K_CO2,
    rho_co2=RHO_CO2,
    k_brine=K_BRINE,
    rho_brine=RHO_BRINE,
    k_matrix=K_MATRIX,
    rho_matrix=RHO_MATRIX,
    phi=PHI,
    version_fluid_mixtures="brie",
    version_gassmann="hs",
)

# First plot - saturation vs bulk modulus for the three Gassmann versions
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(saturation_vector_co2*100, gassmann_wood.bulk_modulus_co2_brine_mixture(saturation_vector_co2), label="Wood", color="blue")
ax.plot(saturation_vector_co2*100, gassmann_domenico.bulk_modulus_co2_brine_mixture(saturation_vector_co2), label="Domenico", color="green")
ax.plot(saturation_vector_co2*100, gassmann_brie.bulk_modulus_co2_brine_mixture(saturation_vector_co2), label="Brie", color="red")

# Add lines for K_AIR, K_DRY and K_MATRIX
ax.axhline(K_AIR, color="gray", linestyle="--", label="Air")
ax.axhline(K_DRY, color="orange", linestyle="--", label="Dry Rock")
# ax.axhline(K_MATRIX, color="purple", linestyle="--", label="Matrix")
ax.set_xlabel("CO2 Saturation (%)")
ax.set_ylabel("Bulk Modulus (GPa)")
ax.set_title("Bulk Modulus of CO2-Brine Mixture vs Saturation")
ax.legend()

fig.savefig(setup_figures / "gassmann_transformations.jpg", format='jpg', dpi=300, bbox_inches='tight')

# Second plot - saturation vs velocity for the three Gassmann versions
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(saturation_vector_co2*100, gassmann_wood.saturation_to_vp(saturation_vector_co2), label="Wood", color="blue")
ax.plot(saturation_vector_co2*100, gassmann_domenico.saturation_to_vp(saturation_vector_co2), label="Domenico", color="green")
ax.plot(saturation_vector_co2*100, gassmann_brie.saturation_to_vp(saturation_vector_co2), label="Brie", color="red")

# Sanity check for my own calculations
ax.plot(saturation_vector_co2*100, gassmann_wood_hs.saturation_to_vp(saturation_vector_co2), label="Wood HS", color="black", linestyle="--")
ax.plot(saturation_vector_co2*100, gassmann_domenico_hs.saturation_to_vp(saturation_vector_co2), label="Domenico HS", color="yellow", linestyle="--")
ax.plot(saturation_vector_co2*100, gassmann_brie_hs.saturation_to_vp(saturation_vector_co2), label="Brie HS", color="purple", linestyle="--")

ax.set_xlabel("CO2 Saturation (%)")
ax.set_ylabel("P-wave Velocity (m/s)")
ax.set_title("P-wave Velocity of CO2-Brine Mixture vs Saturation")
ax.legend()

fig.savefig(setup_figures / "gassmann_transformations_velocity.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% [markdown]
# Initialise petrophysical parametrisation for Archie's law

SATURATION_EXPONENT = 1.62
CEMENTATION_EXPONENT = 2.0
R_FLUID = 3e-2

archie = ArchieTransformation(
    r_fluid=R_FLUID,
    phi=PHI,
    m=CEMENTATION_EXPONENT,
    n=SATURATION_EXPONENT,
)

# Plot saturation vs resistivity for Archie's law
saturation_vector = np.linspace(0.01, 0.99, 100)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(saturation_vector*100, archie.sat_to_res_function(saturation_vector), label="Archie", color="blue")
ax.set_yscale("log")
ax.set_xlabel("CO2 Saturation (%)")
ax.set_ylabel("Resistivity (Ohm-m)")
ax.set_title("Resistivity of CO2-Brine Mixture vs Saturation")
ax.legend()

fig.savefig(setup_figures / "archie_transformation.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% [markdown]
# Create mesh
final_mesh = mt.createMesh(domain_with_co2_plume_and_sensors, area=10, quality=33)
final_mesh, _ = add_boundary_mesh(final_mesh)

final_mesh.save(str(path_data / "final_mesh_wo_models.mesh"))

# %% Create transformations for reservoir and caprock
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

saturation_vector = np.linspace(0.01, 0.55, 100)
fig, ax_res = plt.subplots(figsize=(6,4))

ax_res.plot(saturation_vector*100, archie_reservoir.sat_to_res_function(saturation_vector), label="Reservoir", color="red", linestyle="-")
ax_res.plot(saturation_vector*100, archie_caprock.sat_to_res_function(saturation_vector), label="Caprock", color="red", linestyle="--")

ax_vp = ax_res.twinx()

ax_vp.plot(saturation_vector*100, gassmann_reservoir.saturation_to_vp(saturation_vector), label="Reservoir", color="blue", linestyle="-")
ax_vp.plot(saturation_vector*100, gassmann_caprock.saturation_to_vp(saturation_vector), label="Caprock", color="blue", linestyle="--")

ax_res.set_xlabel("CO2 Saturation (%)")
ax_vp.set_ylabel("P-wave Velocity (m/s)")
ax_res.set_ylabel("Resistivity (Ohm-m)")
ax_res.set_title("Petrophysical Transformations for Reservoir and Caprock")

# Create combined legend just two lines for reservoir(-) and caprock(--)
lines = [plt.Line2D([0], [0], color="black", linestyle="-"), plt.Line2D([0], [0], color="black", linestyle="--")]
labels = ["Reservoir", "Caprock"]
ax_res.legend(lines, labels, loc="upper center")

ax_res.set_xlim(0, 55)
ax_vp.set_ylim(2600, 3500)
# Format vp velocity to km/s
ax_vp.set_yticklabels([f'{x/1000:.1f}' for x in ax_vp.get_yticks()])
ax_res.set_ylim(0, 9)

fig.savefig(setup_figures / "petrophysical_transformations.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% Parse models to mesh
final_mesh_with_models = final_mesh.copy()
parse_mesh_to_saturation_model(
    mesh=final_mesh_with_models,
    saturation_caprock=SATURATION_CAPROCK,
    saturation_co2plume=SATURATION_CO2,
    saturation_reservoir=SATURATION_RESERVOIR,
)

parse_mesh_to_vp_model(
    mesh=final_mesh_with_models,
    seis_transformation_caprock=gassmann_caprock,
    seis_transformation_reservoir=gassmann_reservoir,
    saturation_caprock=SATURATION_CAPROCK,
    saturation_reservoir=SATURATION_RESERVOIR,
    saturation_co2plume=SATURATION_CO2,
)

parse_mesh_to_resistivity_model(
    mesh=final_mesh_with_models,
    ert_transformation_caprock=archie_caprock,
    ert_transformation_reservoir=archie_reservoir,
    saturation_caprock=SATURATION_CAPROCK,
    saturation_reservoir=SATURATION_RESERVOIR,
    saturation_co2plume=SATURATION_CO2,
)

final_mesh_with_models.save(str(path_data / "final_mesh_with_models.mesh"))

CMIN_SATURATION = 0.0
CMAX_SATURATION = 0.55
CMIN_RESISTIVITY = 0.0
CMAX_RESISTIVITY = 2.1
CMIN_VP = 2.6e3
CMAX_VP = 3.5e3

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
_ = pg.show(
    mesh=final_mesh_with_models,
    data=final_mesh_with_models["saturation"],
    label="Saturation",
    ax=ax[0],
    cMap="turbo",
    cMin=CMIN_SATURATION,
    cMax=CMAX_SATURATION,
)

_ = pg.show(
    mesh=final_mesh_with_models,
    data=final_mesh_with_models["res"],
    label="Resistivity",
    ax=ax[1],
    cMap="turbo",
    cMin=CMIN_RESISTIVITY,
    cMax=CMAX_RESISTIVITY,
)

_ = pg.show(
    mesh=final_mesh_with_models,
    data=final_mesh_with_models["vp"],
    label="P-wave Velocity",
    ax=ax[2],
    cMap="turbo",
    cMin=CMIN_VP,
    cMax=CMAX_VP,
)

for axi in ax:
    axi.set_xlabel("Distance [m]")
    axi.set_ylabel("Depth [m]")
    axi.set_aspect("equal")
    axi.set_xlim(DOMAIN_XMIN-BUFFER_X, DOMAIN_XMAX+BUFFER_X)
    axi.set_ylim(DOMAIN_YMIN-BUFFER_Y, DOMAIN_YMAX+BUFFER_Y)
    plot_boreholes_on_mesh(axi)

fig.savefig(setup_figures / "final_models_on_mesh.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% Simulate data
for rel_noise in [0.0, 0.03]:
    data_dongle=f"noise_{rel_noise*100:.0f}"
    #* Simulate ERT data
    print(f"Creating data for case {data_dongle}")
    ert_data = ert.simulate(mesh=final_mesh_with_models, scheme=ert_scheme, res=final_mesh_with_models["res"], verbose=True, noiseLevel=rel_noise, noiseAbs=0.0, seed=1337)
    ert_data["err"] = rel_noise
    ert_data.save(str(path_data.joinpath(f"ert_measurement_{data_dongle}.data")))
        
    #* Simulate TT data
    tt_data = tt.simulate(mesh=final_mesh_with_models, scheme=tt_scheme, vel=final_mesh_with_models["vp"], verbose=True, noiseLevel=rel_noise, noiseAbs=0.0, seed=1337)
    tt_data["err"] = rel_noise
    tt_data.save(str(path_data.joinpath(f"tt_measurement_{data_dongle}.data")))

# %% Plot simulated data for one of the noiseless datasets
# Load data
tt_data_noiseless = tt.load(str(path_data.joinpath("tt_measurement_noise_3.data")))
ert_data_noiseless = ert.load(str(path_data.joinpath("ert_measurement_noise_3.data")))

fig, axs = plot_apparent_velocities_from_data(tt_data_noiseless, cMin=2.6e3, cMax=3.5e3)
fig.savefig(setup_figures / "tt_apparent_velocities.jpg", format='jpg', dpi=300, bbox_inches='tight')

data_matrices, offsets = gather_datamatrices_by_offset(ert_data_noiseless)
fig, axs = plot_datamatrices_by_offset(data_matrices[3:], offsets[3:], cMap="turbo", cMin=0.2, cMax=2.1)
fig.savefig(setup_figures / "ert_datamatrices_by_offset.jpg", format='jpg', dpi=300, bbox_inches='tight')

# %% Create inversion mesh and save it
AREA_INVERSION = 30
REFINEMENT_INVERSION = 0.5

inversion_plc = domain_with_layer.copy()
inversion_plc = add_sensor_positions_to_plc(
    plc=inversion_plc,
    sensor_positions=sensor_positions,
    refinement=REFINEMENT_INVERSION,
    minimum_distance=0.49,
    alsoupper=False,
)

inversion_mesh = mt.createMesh(inversion_plc, quality=33, area=AREA_INVERSION)
inversion_mesh, _ = add_boundary_mesh(inversion_mesh)

# Remove markers
for c in inversion_mesh.cells():
    c.setMarker(0)

#* Check minimum number of neighbour cells
minimum_cell_neighbour_count = np.inf
for cell in inversion_mesh.cells():
    neighbour_cells = []
    for j in range(cell.neighborCellCount()):
        try:
            neighbour_cells.append(cell.neighborCell(j).id())
        except AttributeError:
            pass
            # print(f"Cell {cell.id()} has no neighbour at position {j}")
    if len(neighbour_cells) < minimum_cell_neighbour_count:
        minimum_cell_neighbour_count = len(neighbour_cells)
print(f"Minimum number of neighbour cells: {minimum_cell_neighbour_count}")

# Check if all sensor position are nodes
node_positions = np.array([node.pos() for node in inversion_mesh.nodes()])
node_check=True
for sensor in sensor_positions:
    if sensor not in node_positions:
        node_check=False
print(f"Node check for: {node_check}")

# Print inversion mesh with sensor positions
fig, ax = plt.subplots(1, 1, figsize=(4, 8))
_ = pg.show(
    mesh=inversion_mesh,
    markers=True,
    showMesh=True,
    ax=ax,
    colorBar=False,
)
ax.set_title("Inversion Mesh with Sensor Positions")
ax.scatter(
    [sensor.x() for sensor in sensor_positions],
    [sensor.y() for sensor in sensor_positions],
    color="red",
    label="Sensor Positions",
)
ax.legend()

fig.savefig(setup_figures / "inversion_mesh.jpg", format='jpg', dpi=300, bbox_inches='tight')
inversion_mesh.save(str(path_data / "inversion_mesh.mesh"))