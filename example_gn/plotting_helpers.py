import numpy as np
import matplotlib.pyplot as plt
import scipy as sP


def gather_datamatrices_by_offset(data_ert):
    """Group ERT data into offset-aligned data matrices.

    This function examines the ERT container `data_ert` and groups
    measurements by offset (m-a and n-b). For each unique offset where
    both boreholes have measurements it creates a 2D matrix with injection
    indices along rows and measurement indices along columns.

    Parameters
    ----------
    data_ert : pygimli.core._pygimli_.DataContainerERT
        ERT data container providing integer fields `a, b, m, n` and a
        data field `rhoa` with resistivity values.

    Returns
    -------
    data_matrix_list : list of numpy.ndarray
        List of 2D arrays (one per unique offset) containing `rhoa` values.
    unique_offset_values : numpy.ndarray
        Sorted array of unique offset values considered.
    """
    a = data_ert["a"]
    b = data_ert["b"]
    m = data_ert["m"]
    n = data_ert["n"]
    offset_A = m - a
    offset_B = n - b

    unique_offset_values = np.unique(np.concatenate((offset_A, offset_B)))
    unique_offset_values.sort()
    print(f"Unique offset values: {unique_offset_values}")

    data_matrix_list = []

    for offset in unique_offset_values:
        borehole_A_offset_index_vector = np.where(offset_A == offset)[0]
        borehole_B_offset_index_vector = np.where(offset_B == offset)[0]

        borehole_A_and_B_offset_index_vector = np.where((offset_A == offset) & (offset_B == offset))[0]

        a_temp = a[borehole_A_and_B_offset_index_vector]
        b_temp = b[borehole_A_and_B_offset_index_vector]
        
        minimum_injection_a = np.min(a_temp)
        minimum_injection_b = np.min(b_temp)

        data_matrix = np.zeros((len(np.unique(a_temp)), len(np.unique(b_temp))))

        i_vector = a_temp - minimum_injection_a
        j_vector = b_temp - minimum_injection_b

        data_matrix[i_vector, j_vector] = data_ert["rhoa"][borehole_A_and_B_offset_index_vector]

        data_matrix_list.append(data_matrix.copy())

    return data_matrix_list, unique_offset_values

def plot_datamatrices_by_offset(data_matrices, offsets, cMap="turbo", cMin=None, cMax=None, figsize=(20, 10), layout='constrained'):
    """
    Plot a sequence of data matrices (from `gather_datamatrices_by_offset`).

    Parameters
    ----------
    data_matrices : sequence of 2D arrays
        Matrices to plot.
    offsets : sequence
        Offset values corresponding to `data_matrices`.
    cMap, cMin, cMax : optional
        Colormap and color limits forwarded to `imshow`.
    figsize : tuple
        Figure size forwarded to `plt.subplots`.
    layout : str
        Subplot layout argument forwarded to `plt.subplots` (e.g. 'constrained').

    Returns
    -------
    fig, axs
        Matplotlib figure and axes (axs is an array of Axes objects).
    """
    fig, axs = plt.subplots(1, len(data_matrices), figsize=figsize, layout=layout)

    # Normalize axs to an array for consistent indexing
    if len(data_matrices) == 1:
        axs = [axs]

    for num, data_matrix in enumerate(data_matrices):
        ax = axs[num]
        im = ax.imshow(data_matrix, cmap=cMap, vmin=cMin, vmax=cMax, origin="upper")
        ax.set_title(f"Offset: {offsets[num]}")
        ax.set_xlabel("Borehole B")
        ax.set_ylabel("Borehole A")
        fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)

    return fig, axs

def plot_apparent_velocities_from_data(data_tt, ax=None, field="v", cMap="turbo", cMin=None, cMax=None, figsize=(5, 5)):
    """Create a matrix view of apparent seismic velocities from travel-times.

    Parameters
    ----------
    data_tt : pygimli.core._pygimli_.DataContainer
        Traveltime data with integer sensor indices `s` (sources) and `g` (geophones)
        and a time field `t` when plotting apparent velocity (`field='v'`).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None a new figure/axes pair is created using
        `figsize`.
    field : {'v','s','t'}
        Which quantity to place into the matrix: apparent velocity ('v'),
        source-geophone distance ('s') or travel time ('t').
    cMap, cMin, cMax : optional
        Colormap and value limits forwarded to `imshow`.
    figsize : tuple
        Figure size used when `ax is None`.

    Returns
    -------
    fig, ax
        Matplotlib figure and the Axes containing the matrix plot.
    """

    assert field in ["v", "s", "t"], "Field must be either 'v', 's' or 't'"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    sensor_positions = np.array(data_tt.sensorPositions()[:,:2])

    distance_matrix = sP.spatial.distance_matrix(sensor_positions, sensor_positions)

    sources = np.array(data_tt["s"])
    geophones = np.array(data_tt["g"])

    minimum_source_index = np.min(sources)
    minimum_geophone_index = np.min(geophones)

    i_vector = sources - minimum_source_index
    j_vector = geophones - minimum_geophone_index

    data_matrix = np.zeros((len(np.unique(sources)), len(np.unique(geophones))))

    distance_vector = distance_matrix[sources, geophones]

    if field == "v":
        data_matrix[i_vector, j_vector] = distance_vector / data_tt["t"]
    elif field == "s":
        data_matrix[i_vector, j_vector] = distance_vector
    elif field == "t":
        data_matrix[i_vector, j_vector] = data_tt["t"]
        
    im = ax.imshow(data_matrix, cmap=cMap, vmin=cMin, vmax=cMax, origin="upper")
    _=ax.set_title("Apparent velocities")
    _=ax.set_xlabel("Geophones")
    _=ax.set_ylabel("Sources")
    fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)
    return fig, ax