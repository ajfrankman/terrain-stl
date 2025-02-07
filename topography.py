import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import earthpy as et
import rasterio as rio
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from stl import mesh
from datetime import datetime
import math
import trimesh
from scipy.ndimage import zoom


def create_solid_terrain_stl_old(X, Y, Z, output_folder, base_thickness_mm=None):
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

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Example usage to create a file name
    output_file = f'{output_folder}_{timestamp}.stl'
    terrain_mesh.save(output_file)


def create_solid_terrain_stl(elevation_data, output_folder, size=200, base_thickness_mm=None, z_height=None):
    """
    Create a watertight STL file for the terrain with a flat base, given only Z dimension data.

    Parameters:
    - elevation_data: 2D array of elevation data (shape: [rows, cols]).
    - output_folder: Path to save the STL file.
    - size: Desired size for the model (uniform scaling).
    - base_thickness_mm: Optional, thickness of the base in millimeters (STL units). If provided,
      removes the current base below the minimum elevation and adds the specified thickness.
    - z_height: Optional, specific Z height for scaling. If not provided, defaults to 1/3 of the size.
    """

    # Downsize the elevation data to 

    # Resize the elevation data
    Z = zoom(elevation_data, (500 / elevation_data.shape[0], 500 / elevation_data.shape[1]))

    # Determine the grid dimensions
    rows, cols = Z.shape

    # Create X and Y coordinates based on indices
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Uniform scaling factor for X and Y
    xy_scale_factor = size / max(X.max() - X.min(), Y.max() - Y.min())
    Z_scale_factor = (z_height if z_height is not None else size / 5) / (Z.max() - Z.min())

    # Scale X, Y, and Z
    X = X * xy_scale_factor
    Y = Y * xy_scale_factor
    Z = Z * Z_scale_factor

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

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Side walls connecting the top surface to the base
    offset = rows * cols
    for i in range(rows - 1):
        for j in range(cols - 1):
            top_v0 = i * cols + j
            top_v1 = top_v0 + 1
            top_v2 = top_v0 + cols
            top_v3 = top_v2 + 1

            base_v0 = offset + top_v0
            base_v1 = offset + top_v1
            base_v2 = offset + top_v2
            base_v3 = offset + top_v3

            faces.extend([
                [top_v0, base_v0, top_v1], [top_v1, base_v0, base_v1],
                [top_v1, base_v1, top_v3], [top_v3, base_v1, base_v3],
                [top_v3, base_v3, top_v2], [top_v2, base_v3, base_v2],
                [top_v2, base_v2, top_v0], [top_v0, base_v2, base_v0]
            ])

    # Base faces
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = offset + i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1

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

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save STL file
    output_file = f'{output_folder}/{timestamp}.stl'
    terrain_mesh.save(output_file)


def select_sub_data_old(data, factor_x_left, factor_x_right, factor_y_top, factor_y_bottom):
    # Each factor_*_**** variable is a number from 0-100.
    # The number is used to make a line at a fraction of the max x or max y respectively. 
    # x variables can be thought of longitudinal lines.
    # y variables can be thought of latitudinal lines.
    max_width, max_height = data.shape
    y_top = int((factor_y_top/100)*max_height)
    y_bottom = int((factor_y_bottom/100)*max_height)
    x_left = int((factor_x_left/100)*max_width)
    x_right = int((factor_x_right/100)*max_width)
    return data[y_top:y_bottom, x_left:x_right]


def select_sub_data(data: np.ndarray, coordinates: tuple[float, float], area_size: int):
    # Each factor_*_**** variable is a number from 0-100.
    # The number is used to make a line at a fraction of the max x or max y respectively. 
    # x variables can be thought of longitudinal lines.
    # y variables can be thought of latitudinal lines.
    file_maxes = get_file_max_min_lat_lon(coordinates, area_size)
    
    top_bound, bottom_bound, left_bound, right_bound = get_area_bounds(coordinates, area_size)
    full_area_width = file_maxes['file_max_lon'] - file_maxes['file_min_lon'] + 1
    full_area_height = file_maxes['file_max_lat'] - file_maxes['file_min_lat'] + 1

    left_factor = abs(left_bound - file_maxes['file_min_lon'])/full_area_width
    right_factor = abs(right_bound - file_maxes['file_min_lon'])/full_area_width
    # Because the file_min_lat is at the top of the file, we need to subtract 1 
    # from the file_min_lat to represent the actualy "bottom" of the larger area.
    bottom_factor = abs(bottom_bound - (file_maxes['file_min_lat']-1))/full_area_height
    top_factor = abs(top_bound - (file_maxes['file_min_lat']-1))/full_area_height

    max_width, max_height = data.shape
    y_top = int(top_factor*max_height)
    y_bottom = int(bottom_factor*max_height)
    x_left = int(left_factor*max_width)
    x_right = int(right_factor*max_width)

    return data[y_bottom:y_top, x_left:x_right]


def preview_terrain(X, Y, Z):
    """
    Preview the terrain model in 3D.
    Parameters:
    - X, Y, Z: 2D arrays of coordinates and elevation data.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none')
    ax.set_title("3D Terrain Preview")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation (Z)")
    plt.show()


def flip_data(data):
    """
    Flip over x axis. Because elevation data origin is bottom
    left and libraries like np use top left, we need to flip
    the data.
    
    Parameters:
    - data: 2D array of elevation data
    
    Returns:
    - flipped_data: 2D array of flipped elevation data
    """
    flipped_data = np.flipud(data)
    return flipped_data


def elevation_heat_plot(elevation_data: np.ndarray):

    # Visualize the array
    plt.contourf(elevation_data, cmap='terrain', levels=300)
    plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Contour Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def get_dem_elevation_data(path: str):
    # Open the DEM with Rasterio
    with rio.open(path) as src:
        # Read the first band (elevation)
        elevation = src.read(1)
        # Set masked values to np.nan
        # elevation[elevation < 0] = np.nan
    return elevation


def get_coor_from_file_name(file_name: str) -> tuple[float]:
    coordinate_str = file_name.split('_')[2]
    pattern = r"^(?P<ns>[nsNS])(?P<ns_coor>\d{1,3})(?P<ew>[weWE])(?P<ew_coor>\d{1,3})$"
    regex = re.compile(pattern)

    # Match the pattern against the example string
    match = regex.match(coordinate_str)

    # Extract the values if there's a match
    if match:
        coor_dict = match.groupdict()  # Get a dictionary of named groups and their values
        ns = coor_dict['ns']
        ns_coor = float(coor_dict['ns_coor'])
        ew = coor_dict['ew']
        ew_coor = float(coor_dict['ew_coor'])
        if ns.lower() == 's':
            ns_coor = ns_coor * -1
        if ew.lower() == 'w':
            ew_coor = ew_coor * -1
    else:
        print("No match found.")
        return 1000, 1000
    return ns_coor, ew_coor


def area_size_to_degrees(area_size: int, latitude: float) -> tuple:
    """
    Longitude shrinks east/west distance. Get a good estimate of the 
    longitude.

    Parameters:
    - area_size: meters wide the area is. Half the area size is the 
        distance from coordinates to the furthest edge of the area.
    - latitude: Used to help adjust the longitude.

    Returns:
    - Get the magnitude of the degrees for latitude and longitude.
    """
    area_degrees_lat = area_size / 111320  # Convert meters to degreeQs for latitude
    area_degrees_lon = area_size / (111320 * math.cos(math.radians(latitude)))

    return area_degrees_lat, area_degrees_lon


def generate_file_pattern(lat, lon):
    """Generate a file name pattern based on latitude and longitude, excluding the date."""
    ns = 'n' if lat >= 0 else 's'
    ew = 'e' if lon >= 0 else 'w'
    return f"USGS_13_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_"


def get_latest_file(directory, file_pattern):
    """Find the latest dated file matching the pattern."""
    files = [f for f in os.listdir(directory) if f.startswith(file_pattern)]
    if not files:
        return None
    # Extract dates and find the latest
    date_pattern = re.compile(r'_(\d{8})$')
    latest_file = max(files, key=lambda f: int(date_pattern.search(f).group(1)) if date_pattern.search(f) else 0)
    return latest_file


def get_area_bounds(coordinates: tuple[float, float], area_size: int):
    """
    Get the lat, lon lines that bound the area.
    
    Parameters: 
    - coordinates: Tuple of floats (latitude, longitude) representing the center point.
    - area_size: Integer, size of the area in meters (height and width).

    Return:
    - top_bound: top longitude bound (degrees)
    - bottom_bound: bottom longitude bound (degrees)
    - left_bound: left longitude bound (degrees)
    - right_bound: right longitude bound (degrees)
    """
    lat, lon = coordinates
    area_degrees_lat, area_degrees_lon = area_size_to_degrees(area_size, lat)
    top_bound = lat + area_degrees_lat / 2
    bottom_bound = lat - area_degrees_lat / 2
    left_bound = lon - area_degrees_lon / 2
    right_bound = lon + area_degrees_lon / 2
    return top_bound, bottom_bound, left_bound, right_bound


def get_data_lat_lon_bounds(coordinates: tuple[float, float], area_size: int):
    """
    Get the lat, lon lines that bound the area.
    
    Parameters: 
    - coordinates: Tuple of floats (latitude, longitude) representing the center point.
    - area_size: Integer, size of the area in meters (height and width).

    Return (as dictionary):
    - top_bound: top longitude bound (degrees)
    - bottom_bound: bottom longitude bound (degrees)
    - left_bound: left longitude bound (degrees)
    - right_bound: right longitude bound (degrees)
    """
    top_bound, bottom_bound, left_bound, right_bound = get_area_bounds(coordinates, area_size)
    min_lat = int(math.floor(bottom_bound))
    max_lat = int(math.floor(top_bound)) + 1
    min_lon = int(math.floor(left_bound))
    max_lon = int(math.floor(right_bound)) - 1
    # Return as dictionary
    bounds = {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon}
    return bounds


def get_file_max_min_lat_lon(coordinates: tuple[float, float], area_size: int):
    """
    Get the lat, lon max values for the files.
    
    Parameters: 
    - coordinates: Tuple of floats (latitude, longitude) representing the center point.
    - area_size: Integer, size of the area in meters (height and width).

    Return (as dictionary):
    - top_bound: top longitude bound (degrees)
    - bottom_bound: bottom longitude bound (degrees)
    - left_bound: left longitude bound (degrees)
    - right_bound: right longitude bound (degrees)
    """
    top_bound, bottom_bound, left_bound, right_bound = get_area_bounds(coordinates, area_size)
    file_min_lat = int(math.floor(bottom_bound))+1
    file_max_lat = int(math.floor(top_bound))+1
    file_min_lon = int(math.floor(left_bound))
    file_max_lon = int(math.floor(right_bound))
    # Return as dictionary
    bounds = {
        'file_min_lat': file_min_lat,
        'file_max_lat': file_max_lat,
        'file_min_lon': file_min_lon,
        'file_max_lon': file_max_lon
        }
    return bounds


def get_files_in_area(directory, coordinates, area_size):
    """
    Get file names that intersect with the specified area.

    Parameters:
    - directory: Path to the directory containing the files.
    - coordinates: Tuple of floats (latitude, longitude) representing the center point.
    - area_size: Integer, size of the area in meters (height and width).

    Returns:
    - List of file names that intersect with the specified area.
    - Name of the single file containing coordinates
    """

    lat, lon = coordinates
    area_degrees_lat, area_degrees_lon = area_size_to_degrees(area_size, lat)

    # Calculate bounding lat/lon for the file names.
    file_bounds = get_file_max_min_lat_lon(coordinates, area_size)
    file_min_lat = file_bounds['file_min_lat']
    file_max_lat = file_bounds['file_max_lat']
    file_min_lon = file_bounds['file_min_lon']
    file_max_lon = file_bounds['file_max_lon']

    # Generate a list of lat/lon tuples.
    needed_lat_lon = []
    for needed_lat in range(file_min_lat, file_max_lat + 1):
        for needed_lon in range(file_min_lon, file_max_lon + 1):
            needed_lat_lon.append((needed_lat, needed_lon))

    # Generate file names and check if they exist
    matching_files = []
    coordinates_file = ''
    for needed_lat, needed_lon in needed_lat_lon:
            file_pattern = generate_file_pattern(needed_lat, needed_lon)
            file_name = get_latest_file(directory, file_pattern)
            if file_name:
                matching_files.append(file_name)
                # Check if this tile contains the exact coordinate
                if needed_lat <= lat < needed_lat + 1 and needed_lon <= lon < needed_lon + 1:
                    coordinates_file = file_name
    # Get the perctage of the area of all combined files that the area_size covers
    total_area = area_degrees_lat * area_degrees_lon
    area_covered = len(matching_files) * area_degrees_lat * area_degrees_lon
    coverage_percentage = area_covered / total_area
    return matching_files, coordinates_file, file_bounds


def get_area_size_percentage(area_size, lat, file_bounds):
    area_degrees_lat, area_degrees_lon = area_size_to_degrees(area_size, lat)
    num_lon_degrees = abs(file_bounds['file_min_lon']) - abs(file_bounds['file_max_lon']) + 1
    num_lat_degrees = abs(file_bounds['file_max_lat']) - abs(file_bounds['file_min_lat']) + 1
    percentage_of_area = (area_degrees_lat * area_degrees_lat) / (num_lon_degrees * num_lat_degrees)
    return percentage_of_area


def get_full_elevation_matrix(directory: str, coordinates: tuple[float, float], 
                              area_size: int):
    """
    Retrieve the full elevation matrix for a specified area.

    Parameters:
    - directory: String, the path to the directory containing elevation data files.
    - coordinates: Tuple of floats (latitude, longitude) indicating the center point of the area.
    - area_size: Integer, the size of the area in meters (height and width).

    Returns:
    - A full elevation matrix (numpy ndarray) for the specified area, adjusted to the desired resolution.
    """
    file_names, coordinate_file, file_bounds = get_files_in_area(directory, coordinates, area_size)
    num_vertical_files = abs(file_bounds['file_max_lat']) - abs(file_bounds['file_min_lat']) + 1
    
    # Merge all the data
    file_names.sort()
    horizontal_files_list = np.array_split(file_names, num_vertical_files)
    h_stacks = []
    for horizontal_files in horizontal_files_list:
        temp_elevation_matrices = []
        for file_name in horizontal_files[::-1]:
            print(f'Processing: {file_name}')
            temp_elevation_matrix = get_dem_elevation_data(f'{directory}/{file_name}')
            temp_elevation_matrices.append(temp_elevation_matrix)
        h_stacks.append(np.hstack(tuple(temp_elevation_matrices)))

    elevation_matrix = np.vstack(tuple(h_stacks[::-1]))
    elevation_matrix = flip_data(elevation_matrix)    
    return elevation_matrix


def generate_stl_from_elevation_data(elevation_data, max_height_mm, output_size_mm, base_height_mm=0):
    # Ensure elevation_data is a 2D array (shape: rows x cols)
    rows, cols = elevation_data.shape
    
    # Normalize the elevation data to fit within the max height, starting at base_height_mm
    elevation_scaled = (elevation_data - np.min(elevation_data)) / (np.max(elevation_data) - np.min(elevation_data))
    elevation_scaled = elevation_scaled * (max_height_mm - base_height_mm) + base_height_mm
    
    # Scale the data to fit the desired output size (longitude/latitude to 3D space)
    x = np.linspace(0, output_size_mm, cols)  # Scale longitude to output size
    y = np.linspace(0, output_size_mm, rows)  # Scale latitude to output size
    
    # Create meshgrid for the coordinates
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create the base (flat layer at base_height_mm)
    base_vertices = np.vstack((x_grid.flatten(), y_grid.flatten(), np.full_like(x_grid.flatten(), base_height_mm))).T
    
    # Create a 3D surface (x, y, z) for the elevation data
    surface_vertices = np.vstack((x_grid.flatten(), y_grid.flatten(), elevation_scaled.flatten())).T
    
    # Combine base and surface vertices into a single array
    vertices = np.vstack([base_vertices, surface_vertices])
    
    # Create faces (triangular mesh) for the elevation surface
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Indices of vertices for the surface
            p1 = i * cols + j
            p2 = i * cols + j + 1
            p3 = (i + 1) * cols + j
            p4 = (i + 1) * cols + j + 1
            
            # Generate two triangles for each grid square (for surface)
            faces.append([p1, p2, p3])
            faces.append([p2, p3, p4])
    
    # Create vertical faces connecting the base to the surface
    vertical_faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Indices of vertices on the base layer
            p1 = i * cols + j
            p2 = i * cols + j + 1
            p3 = (i + 1) * cols + j
            p4 = (i + 1) * cols + j + 1

            # Create vertical walls (connecting the base layer to the terrain)
            vertical_faces.append([p1, p2, p1 + len(base_vertices)])  # Left side wall
            vertical_faces.append([p2, p1 + len(base_vertices), p2 + len(base_vertices)])  # Right side wall

            vertical_faces.append([p3, p4, p3 + len(base_vertices)])  # Bottom side wall
            vertical_faces.append([p4, p3 + len(base_vertices), p4 + len(base_vertices)])  # Top side wall

            vertical_faces.append([p1, p3, p1 + len(base_vertices)])  # Front side wall
            vertical_faces.append([p3, p1 + len(base_vertices), p3 + len(base_vertices)])  # Back side wall

            vertical_faces.append([p2, p4, p2 + len(base_vertices)])  # Back side wall
            vertical_faces.append([p4, p2 + len(base_vertices), p4 + len(base_vertices)])  # Front side wall
    
    # Combine all faces
    faces.extend(vertical_faces)
    
    # Create a trimesh object from the vertices and faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export the mesh to an STL file
    mesh.export('elevation_model_with_base_and_walls.stl')


def create_topo_stl(coordinates: tuple[float, float], area_size: int, output_size: int, shape: str = 'square') -> bool:
    """
    Create a topo based on the given parameters.

    Parameters:
    - coordinates: Tuple of floats (latitude, longitude) representing the center point.
    - area_size: Integer, size of the area in meters (height and width).
    - output_size: Integer, size of the output stl topo in mm (height and width if square).
    - shape: String, available shapes include circle, hexagon, octagon, and square.

    Returns:
    - True if the topo was created successfully, False otherwise.
    """
    # pixel_spacing = .1
    data_directory = "./terrain_data_13"
    # Given the coordinates and area_size, open, merge, and downsample the data
        # Find the right files and merge the data
    elevation_matrix = get_full_elevation_matrix(data_directory, coordinates, area_size)
    # Filter the data (filter before selecting the sub area so there isn't weird smoothing on the edges.)
    # elevation_matrix = gaussian_filter(elevation_matrix, sigma=2)    
    # # Trim the data to shape
    elevation_matrix = select_sub_data(elevation_matrix, coordinates, area_size)
    elevation_matrix = resize(elevation_matrix, (500, 500))

    # generate_stl_from_elevation_data(elevation_matrix, 50, 200, 3)
    # create_solid_terrain_stl(elevation_data, output_folder, size=200, base_thickness_mm=None, z_height=None):
    create_solid_terrain_stl(elevation_matrix, './STLS', size=100, base_thickness_mm=5)


coordinates = (44.63898065419271, -111.40156966219595)  # Example latitude and longitude
area_size =20000 # 5000 meters (5 km)
create_topo_stl(coordinates, area_size, 100)

# import rasterio

# with rasterio.open("terrain_data_13/USGS_13_n45w112_20241127.tif") as src:
#     bounds = src.bounds
#     print(f"Bounding Box: {bounds}")

# # temp_elevation_matrix = get_dem_elevation_data('terrain_data_13/USGS_13_n44w111_20240325.tif')
# # temp_elevation_matrix = get_dem_elevation_data('terrain_data_13/USGS_13_n44w112_20240701.tif')
# # # temp_elevation_matrix = get_dem_elevation_data('terrain_data_13/USGS_13_n45w112_20241127.tif')
# # # temp_elevation_matrix = get_dem_elevation_data('terrain_data_13/USGS_13_n45w111_20240325.tif')
# # temp_elevation_matrix = resize(temp_elevation_matrix, (500, 500))
# # temp_elevation_matrix = flip_data(temp_elevation_matrix)    

# elevation_heat_plot(temp_elevation_matrix)

# get_area_bounds(coordinates, area_size)

# # Z is in meters. X and Y are variable depending on the data. This should
# # be a multiplier to make X and Y have the same magnitude as Z.
# XY_STRETCH = 30
# # Used to make mountains visible
# ELEVATION_MULTIPLIER = 1.5
# dtm = "marsh_valley.tif"


# elevation = select_sub_data(elevation, 61, 82, 34, 55)

# elevation = resize(elevation, (500, 500))  # Resample to a smaller grid

# # Create STL
# rows, cols = elevation.shape
# X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
# X = X * XY_STRETCH
# Y = Y * XY_STRETCH
# # Scale Z (elevation) to fit the desired height
# Z = elevation * ELEVATION_MULTIPLIER
# Z = np.around(Z, decimals=3)  # Round to 3 decimal places to avoid precision issues


# # Check and flip axes if needed
# # Flip Y-axis to match input data orientation
# Y = np.flipud(Y)


# create_solid_terrain_stl(X, Y, Z, "terrain_model_fully_solid.stl", base_thickness_mm=5)
# # preview_terrain(X,Y,Z)
