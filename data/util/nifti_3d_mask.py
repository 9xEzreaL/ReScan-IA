"""
3D mask generation utilities for NIfTI images
"""
import numpy as np
import torch

def bbox3d_to_mask(img_shape, center_x, center_y, center_z, size_x, size_y, size_z, dtype='uint8'):
    """
    Generate 3D mask from bounding box center and size.
    
    Args:
        img_shape: tuple of (width, height, depth) or (X, Y, Z) - NIfTI format
        center_x: center x coordinate (float)
        center_y: center y coordinate (float)
        center_z: center z coordinate (slice, float)
        size_x: size in x direction (int)
        size_y: size in y direction (int)
        size_z: size in z direction (int)
        dtype: data type of mask
    
    Returns:
        numpy.ndarray: Mask in the shape of (X, Y, Z) with 1 for mask region, 0 otherwise
    """
    # NIfTI shape is typically (width, height, depth) = (X, Y, Z)
    width, height, depth = img_shape
    
    # Calculate bounding box coordinates
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    half_z = size_z / 2.0
    
    x_min = max(0, int(np.floor(center_x - half_x)))
    x_max = min(width, int(np.ceil(center_x + half_x)))
    y_min = max(0, int(np.floor(center_y - half_y)))
    y_max = min(height, int(np.ceil(center_y + half_y)))
    z_min = max(0, int(np.floor(center_z - half_z)))
    z_max = min(depth, int(np.ceil(center_z + half_z)))
    
    # Create mask
    # Note: mask shape is (width, height, depth) = (X, Y, Z)
    # So indexing should be mask[x, y, z]
    mask = np.zeros((width, height, depth), dtype=dtype)
    mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1
    
    return mask


def sphere3d_to_mask(img_shape, center_x, center_y, center_z, radius, dtype='uint8'):
    """
    Generate 3D spherical mask from center and radius.
    
    Args:
        img_shape: tuple of (width, height, depth) or (X, Y, Z) - NIfTI format
        center_x: center x coordinate (float)
        center_y: center y coordinate (float)
        center_z: center z coordinate (slice, float)
        radius: radius of the sphere (float)
        dtype: data type of mask
    
    Returns:
        numpy.ndarray: Mask in the shape of (X, Y, Z) with 1 for mask region, 0 otherwise
    """
    # NIfTI shape is typically (width, height, depth) = (X, Y, Z)
    width, height, depth = img_shape
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    z = np.arange(depth)
    
    # Create meshgrids
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # indexing='ij' gives (X, Y, Z) order
    
    # Calculate distance from center for each voxel
    dist_squared = (X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2
    
    # Create mask: voxels within radius
    mask = (dist_squared <= radius**2).astype(dtype)
    
    return mask

