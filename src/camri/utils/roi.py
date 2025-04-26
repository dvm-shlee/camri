import matplotlib.colors as mcolors
import numpy as np


def create_discrete_colormap(color_list):
    """
    Creates a discrete colormap that maps integer indices to colors using the given color_list.
    The first color in the color_list must be 'black'.
    
    Parameters:
        color_list (list of str): A list of color names or hex codes. Example: ["black", "red", "blue", ...]
    
    Returns:
        matplotlib.colors.ListedColormap: The generated discrete colormap
    """
    if color_list[0].lower() != "black":
        raise ValueError("The first color in color_list must be 'black'.")
    return mcolors.ListedColormap(color_list)


def find_closest_coordinate(meshgrid, target_coord):
    """
    Finds the index of the closest coordinate in a 3D meshgrid to a given target coordinate.

    Parameters:
        meshgrid (numpy.ndarray): A 4D array representing the 3D meshgrid. The last dimension
                                  should contain the coordinate components (e.g., x, y, z).
        target_coord (numpy.ndarray): A 1D array representing the target coordinate (e.g., [x, y, z]).

    Returns:
        tuple: The index of the closest coordinate in the meshgrid as a tuple of integers.
    """
    diff = meshgrid - target_coord
    distance = np.linalg.norm(diff, axis=3)
    closest_index = np.unravel_index(np.argmin(distance), distance.shape)
    return closest_index

def find_voxels_within_diameter(meshgrid, target_coord, diameter, tol=0.15, plane=None, slice_tol=0.15):
    """
    Given a world coordinate meshgrid, target_coord, and the diameter of a spherical or circular ROI,
    this function returns the indices of voxels within a distance of (diameter/2) from the target_coord.

    3D mode: Calculates a spherical ROI (radius = diameter/2) across the entire meshgrid.
    2D mode: If the plane parameter is specified, it fixes the slice within the given tolerance (slice_tol)
                and calculates a circular ROI (radius = diameter/2) in the remaining two axes.

    The returned indices are in the form of a tuple obtained via np.where(mask), which can be used
    directly for indexing 3D or 4D images.

    Parameters:
        meshgrid (np.ndarray): World coordinate meshgrid, shape = (X, Y, Z, 3)
        target_coord (array-like): Reference world coordinate, shape = (3,)
        diameter (float): Diameter of the ROI (in world units)
        tol (float, optional): Tolerance for distance comparison (default 0.15)
        plane (int, optional): Fixed axis for creating a 2D ROI (0, 1, or 2). If not specified, 3D mode is used.
        slice_tol (float, optional): Tolerance for slice selection in 2D mode (default 0.15)

    Returns:
        tuple: Indices of voxels satisfying the conditions (i, j, k)
    """
    radius = diameter / 2.0
    target_coord = np.array(target_coord)
    
    if plane is None:
        # 3D mode: Calculate distance across all three axes
        diff = meshgrid - target_coord  # shape: (X, Y, Z, 3)
        distance = np.linalg.norm(diff, axis=3)
        mask = distance <= (radius + tol)
    else:
        # 2D mode: Fix the slice along the specified plane and calculate distance in the other two axes
        # 1. Select voxels within slice_tol along the specified plane
        slice_mask = np.abs(meshgrid[..., plane] - target_coord[plane]) <= slice_tol
        # 2. Calculate distance in the remaining two axes
        #    np.delete: Removes the specified plane dimension from axis=3 → (X, Y, Z, 2)
        diff_2d = np.delete(meshgrid - target_coord, plane, axis=3)
        distance_2d = np.linalg.norm(diff_2d, axis=3)
        circle_mask = distance_2d <= (radius + tol)
        # Combine both conditions
        mask = slice_mask & circle_mask
        
    return np.where(mask)


def find_voxels_in_cube(meshgrid, target_coord, diameter, tol=0.15, plane=None, slice_tol=0.15):
    """
    Given a world coordinate meshgrid, target_coord, and the side length (diameter) of a cubic or square ROI,
    this function returns the indices of voxels within the specified region centered at target_coord.

    3D mode: Calculates a cubic ROI (side length = diameter) across the entire 3D space.
    2D mode: If the plane parameter is specified, it fixes the slice within the given tolerance (slice_tol)
             along the specified axis and calculates a square ROI (side length = diameter) in the remaining two axes.

    The returned indices are in the form of a tuple obtained via np.where(mask), which can be used
    directly for indexing 3D or 4D images.

    Parameters:
        meshgrid (np.ndarray): World coordinate meshgrid, shape = (X, Y, Z, 3)
        target_coord (array-like): Reference world coordinate, shape = (3,)
        diameter (float): Side length of the ROI (in world units)
        tol (float, optional): Tolerance for distance comparison (default 0.15)
        plane (int, optional): Fixed axis for creating a 2D ROI (0, 1, or 2). If not specified, 3D mode is used.
        slice_tol (float, optional): Tolerance for slice selection in 2D mode (default 0.15)

    Returns:
        tuple: Indices of voxels satisfying the conditions (i, j, k)
    """
    half_side = diameter / 2.0
    target_coord = np.array(target_coord)
    
    if plane is None:
        # 3D mode: Apply cube ROI conditions across all axes
        diff = np.abs(meshgrid - target_coord)
        mask = (diff[..., 0] <= (half_side + tol)) & \
               (diff[..., 1] <= (half_side + tol)) & \
               (diff[..., 2] <= (half_side + tol))
    else:
        # 2D mode: First, fix the slice along the specified plane
        slice_mask = np.abs(meshgrid[..., plane] - target_coord[plane]) <= slice_tol
        # Apply square conditions to the remaining two axes
        # np.delete: Removes the specified plane dimension from axis=3 → (X, Y, Z, 2)
        diff_2d = np.abs(np.delete(meshgrid, plane, axis=3) - np.delete(target_coord, plane))
        square_mask = (diff_2d[..., 0] <= (half_side + tol)) & \
                      (diff_2d[..., 1] <= (half_side + tol))
        mask = slice_mask & square_mask
        
    return np.where(mask)