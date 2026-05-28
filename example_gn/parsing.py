"""Helpers to parse pygimli meshes into petrophysical models.

This module provides small utilities that convert a mesh with region
markers into per-cell property vectors (saturation, resistivity, P-wave
velocity) and attach them to the mesh via `mesh.addData(...)` when desired.

The functions expect transformation objects conforming to the
`Transformation` interface used elsewhere in the examples: they must
implement a `.forward(x)` method that maps saturation -> physical value.
"""

import numpy as np
import pygimli as pg

# Region markers used in the example geometry
MARKER_CAPROCK = 0
MARKER_SANDSTONE = 1
MARKER_CO2 = 2
MARKER_BOUNDARY = 3

def parse_mesh_to_saturation_model(
        mesh,
        saturation_caprock=0.0,
        saturation_co2plume=0.5,
        saturation_reservoir=0.0,
        marker_caprock=MARKER_CAPROCK,
        marker_co2plume=MARKER_CO2,
        marker_reservoir=MARKER_SANDSTONE,
        overwrite=True
):
    """
    Parse a mesh to a saturation model.

    Parameters
    ----------
    mesh : pygimli.core._pygimli_.Mesh
        Mesh to parse. The mesh must have per-cell `marker()` values that
        indicate the geological region (caprock, reservoir, CO2 plume).
    saturation_caprock : float
        Saturation value assigned to cells with `marker_caprock`.
    saturation_co2plume : float
        Saturation of CO2 plume.
    saturation_reservoir : float
        Saturation of reservoir.
    overwrite : bool
        Overwrite existing saturation values.

    Returns
    -------
    numpy.ndarray
        1D array with one saturation value per mesh cell (index == cell.id()).
        The array is also attached to the mesh with `mesh.addData("saturation", arr)`
        when `overwrite=True`.
    """
    n_cells = mesh.cellCount()
    saturation_model = np.zeros(n_cells)
    for cell in mesh.cells():
        if cell.marker() == marker_caprock:
            saturation_model[cell.id()] = saturation_caprock
        elif cell.marker() == marker_co2plume:
            saturation_model[cell.id()] = saturation_co2plume
        elif cell.marker() == marker_reservoir:
            saturation_model[cell.id()] = saturation_reservoir
        else:
            raise ValueError("Unknown marker")
    
    if overwrite:
        mesh.addData("sat", saturation_model)
    return saturation_model

def parse_mesh_to_resistivity_model(
    mesh,
    ert_transformation_caprock,
    ert_transformation_reservoir,
    saturation_caprock=0.0,
    saturation_co2plume=0.5,
    saturation_reservoir=0.0,
    marker_caprock=MARKER_CAPROCK,
    marker_co2plume=MARKER_CO2,
    marker_reservoir=MARKER_SANDSTONE,
    overwrite=True
):
    """
    Parse a mesh to a resistivity model.

    Parameters
    ----------
    mesh : pygimli.core._pygimli_.Mesh
        Mesh to parse. Requires cell `marker()` values.
    ert_transformation_caprock : object
        Transformation object mapping saturation -> resistivity for caprock.
        Must implement `.forward(saturation)`.
    ert_transformation_reservoir : object
        Transformation object mapping saturation -> resistivity for the
        reservoir region.
    saturation_caprock : float
        Saturation of caprock.
    saturation_co2plume : float
        Saturation of CO2 plume.
    saturation_reservoir : float
        Saturation of reservoir.
    overwrite : bool
        Overwrite existing saturation values.

    Returns
    -------
    pygimli.core._pygimli_.RVector
        Resistivity model.
    """

    # First add saturation model to mesh if not already present
    if not "sat" in mesh.dataKeys():
        saturation_model = parse_mesh_to_saturation_model(
            mesh=mesh,
            saturation_caprock=saturation_caprock,
            saturation_co2plume=saturation_co2plume,
            saturation_reservoir=saturation_reservoir,
            marker_caprock=marker_caprock,
            marker_co2plume=marker_co2plume,
            marker_reservoir=marker_reservoir,
            overwrite=overwrite
        )
    else:
        saturation_model = mesh["sat"]
        print("Saturation available in mesh - skipping saturation parsing")
    
    n_cells = mesh.cellCount()
    resistivity_model = np.zeros(n_cells)
    for cell in mesh.cells():
        if cell.marker() == marker_caprock:
            resistivity_model[cell.id()] = ert_transformation_caprock.forward(saturation_model[cell.id()])
        elif cell.marker() == marker_co2plume:
            resistivity_model[cell.id()] = ert_transformation_reservoir.forward(saturation_model[cell.id()])
        elif cell.marker() == marker_reservoir:
            resistivity_model[cell.id()] = ert_transformation_reservoir.forward(saturation_model[cell.id()])
        else:
            raise ValueError("Unknown marker")
    
    if overwrite:
        mesh.addData("res", resistivity_model)
    return resistivity_model

def parse_mesh_to_vp_model(
    mesh,
    seis_transformation_caprock,
    seis_transformation_reservoir,
    saturation_caprock=0.0,
    saturation_co2plume=0.5,
    saturation_reservoir=0.0,
    marker_caprock=MARKER_CAPROCK,
    marker_co2plume=MARKER_CO2,
    marker_reservoir=MARKER_SANDSTONE,
    overwrite=True
):
    """
    Parse a mesh to a vp model.

    Parameters
    ----------
    mesh : pygimli.core._pygimli_.Mesh
        Mesh to parse. Requires cell `marker()` values.
    seis_transformation_caprock : object
        Transformation object mapping saturation -> P-wave velocity for caprock.
    seis_transformation_reservoir : object
        Transformation object mapping saturation -> P-wave velocity for reservoir.
    saturation_caprock : float
        Saturation of caprock.
    saturation_co2plume : float
        Saturation of CO2 plume.
    saturation_reservoir : float
        Saturation of reservoir.
    overwrite : bool
        Overwrite existing saturation values.

    Returns
    -------
    pygimli.core._pygimli_.RVector
        Resistivity model.
    """

    #* First add saturation model to mesh if not already present
    if not "saturation" in mesh.dataKeys():
        saturation_model = parse_mesh_to_saturation_model(
            mesh=mesh,
            saturation_caprock=saturation_caprock,
            saturation_co2plume=saturation_co2plume,
            saturation_reservoir=saturation_reservoir,
            overwrite=overwrite
        )
    else:
        saturation_model = mesh["saturation"]
        print("Saturation available in mesh - skipping saturation parsing")
    
    n_cells = mesh.cellCount()
    vp_model = np.zeros(n_cells)
    for cell in mesh.cells():
        if cell.marker() == marker_caprock:
            vp_model[cell.id()] = seis_transformation_caprock.forward(saturation_model[cell.id()])
        elif cell.marker() == marker_co2plume:
            vp_model[cell.id()] = seis_transformation_reservoir.forward(saturation_model[cell.id()])
        elif cell.marker() == marker_reservoir:
            vp_model[cell.id()] = seis_transformation_reservoir.forward(saturation_model[cell.id()])
        else:
            raise ValueError("Unknown marker")
    
    if overwrite:
        mesh.addData("vp", vp_model)
    return vp_model