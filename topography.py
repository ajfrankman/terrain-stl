import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import earthpy as et
import rasterio as rio
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from stl import mesh


def create_solid_terrain_stl(X, Y, Z, output_file, base_thickness_mm=None):
    """
    Create a watertight STL file for the terrain with a flat base.

    Parameters:
    - X, Y, Z: 2D arrays of coordinates and elevation data.
    - output_file: Path to save the STL file.
    - base_thickness_mm: Optional, thickness of the base in millimeters (STL units). If provided,
      removes the current base below the minimum elevation and adds the specified thickness.
    """
    # Uniform scaling factor to fit within 200x200x200mm
    scale_factor = 200 / max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())

    # Scale X, Y, and Z
    X = X * scale_factor
    Y = Y * scale_factor
    Z = Z * scale_factor

    rows, cols = X.shape

    # Add a flat base if base_thickness_mm is provided
    if base_thickness_mm is not None:
        base_z = Z.min() - base_thickness_mm
    else:
        base_z = 0

    # Create vertices for the top surface and base
    top_vertices = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    base_vertices = np.stack((X.flatten(), Y.flatten(), np.full_like(Z.flatten(), base_z)), axis=-1)

    # Combine all vertices
    vertices = np.vstack((top_vertices, base_vertices))

    # Prepare faces
    faces = []

    # Terrain surface faces
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1

            # Top surface triangles
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Side walls connecting the top surface to the base
    offset = rows * cols
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Current vertices
            top_v0 = i * cols + j
            top_v1 = top_v0 + 1
            top_v2 = top_v0 + cols
            top_v3 = top_v2 + 1

            # Corresponding base vertices
            base_v0 = offset + top_v0
            base_v1 = offset + top_v1
            base_v2 = offset + top_v2
            base_v3 = offset + top_v3

            # Side wall triangles
            faces.append([top_v0, base_v0, top_v1])
            faces.append([top_v1, base_v0, base_v1])
            faces.append([top_v1, base_v1, top_v3])
            faces.append([top_v3, base_v1, base_v3])
            faces.append([top_v3, base_v3, top_v2])
            faces.append([top_v2, base_v3, base_v2])
            faces.append([top_v2, base_v2, top_v0])
            faces.append([top_v0, base_v2, base_v0])

    # Base faces
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = offset + i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1

            # Base triangles
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # Remove duplicate vertices and faces
    vertices, unique_faces = np.unique(vertices, axis=0, return_inverse=True)
    faces = unique_faces[np.array(faces).flatten()].reshape(-1, 3)

    # Create STL mesh
    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]

    # Save the STL file
    terrain_mesh.save(output_file)


def select_sub_data(data, factor_x_left, factor_x_right, factor_y_top, factor_y_bottom):
    max_width, max_height = data.shape
    y_top = int((factor_y_top/100)*max_height)
    y_bottom = int((factor_y_bottom/100)*max_height)
    x_left = int((factor_x_left/100)*max_width)
    x_right = int((factor_x_right/100)*max_width)
    return data[y_top:y_bottom, x_left:x_right]


# Z is in meters. X and Y are variable depending on the data. This should
# be a multiplier to make X and Y have the same magnitude as Z.
XY_STRETCH = 30
# Used to make mountains visible
ELEVATION_MULTIPLIER = .7
dtm = "USGS_1_n45w112_20240701.tif"

# Open the DEM with Rasterio
with rio.open(dtm) as src:
    elevation = src.read(1)  # Read the first band (elevation)
    transform = src.transform  # Get the affine transform for coordinates
    # Set masked values to np.nan
    elevation[elevation < 0] = np.nan

elevation = select_sub_data(elevation, 45, 85, 20, 60)
elevation = gaussian_filter(elevation, sigma=5)
elevation = resize(elevation, (300, 300))  # Resample to a smaller grid

# Create STL
rows, cols = elevation.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * XY_STRETCH
Y = Y * XY_STRETCH
# Scale Z (elevation) to fit the desired height
Z = elevation * ELEVATION_MULTIPLIER
Z = np.around(Z, decimals=3)  # Round to 3 decimal places to avoid precision issues


# Check and flip axes if needed
# Flip Y-axis to match input data orientation
Y = np.flipud(Y)

# Example usage
create_solid_terrain_stl(X, Y, Z, "terrain_model_fully_solid.stl", base_thickness_mm=5)
