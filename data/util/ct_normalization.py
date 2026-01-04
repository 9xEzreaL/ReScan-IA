"""
CT/MRA image normalization utilities
"""
import numpy as np
import torch

def ct_normalize(image, window_center=0, window_width=1000, hu_min=-1000, hu_max=1000):
    """
    Normalize CT image using windowing and HU value clipping.
    
    Args:
        image: numpy array or torch tensor, CT image in HU values
        window_center: center of the window (default: 0 for soft tissue)
        window_width: width of the window (default: 1000)
        hu_min: minimum HU value to clip (default: -1000)
        hu_max: maximum HU value to clip (default: 1000)
    
    Returns:
        Normalized image in range [-1, 1] (for diffusion models)
    """
    # Convert to numpy if torch tensor
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        is_torch = True
    else:
        image_np = image.copy()
        is_torch = False
    
    # Clip HU values
    image_np = np.clip(image_np, hu_min, hu_max)
    
    # Apply windowing
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    image_np = np.clip(image_np, window_min, window_max)
    
    # Normalize to [-1, 1] range (for diffusion models)
    if window_max > window_min:
        image_np = 2.0 * (image_np - window_min) / (window_max - window_min) - 1.0
    else:
        image_np = np.zeros_like(image_np)
    
    # Convert back to torch if input was torch
    if is_torch:
        return torch.from_numpy(image_np).to(image.device if hasattr(image, 'device') else 'cpu')
    
    return image_np

def ct_normalize_simple(image, hu_min=-1000, hu_max=1000):
    """
    Simple CT normalization: clip and normalize to [-1, 1]
    
    Args:
        image: numpy array or torch tensor, CT image in HU values
        hu_min: minimum HU value to clip (default: -1000)
        hu_max: maximum HU value to clip (default: 1000)
    
    Returns:
        Normalized image in range [-1, 1]
    """
    # Convert to numpy if torch tensor
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        is_torch = True
    else:
        image_np = image.copy()
        is_torch = False
    
    # Clip HU values
    image_np = np.clip(image_np, hu_min, hu_max)
    
    # Normalize to [-1, 1] range
    if hu_max > hu_min:
        image_np = 2.0 * (image_np - hu_min) / (hu_max - hu_min) - 1.0
    else:
        image_np = np.zeros_like(image_np)
    
    # Convert back to torch if input was torch
    if is_torch:
        return torch.from_numpy(image_np).to(image.device if hasattr(image, 'device') else 'cpu')
    
    return image_np

def ct_normalize_nnunet(image, foreground_percentiles=(0.5, 99.5), global_mean=None, global_std=None, use_nonzero_only=True):
    """
    nnUNet-style CT normalization: foreground clipping + Z-score normalization
    
    Args:
        image: numpy array or torch tensor, CT image in HU values
        foreground_percentiles: tuple of (lower_percentile, upper_percentile) for clipping (default: (0.5, 99.5))
        global_mean: global mean for Z-score normalization (if None, use image mean)
        global_std: global std for Z-score normalization (if None, use image std)
        use_nonzero_only: if True, only use non-zero voxels for statistics (default: True)
    
    Returns:
        Normalized image (Z-score normalized, typically range around [-3, 3])
    """
    # Convert to numpy if torch tensor
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
        is_torch = True
    else:
        image_np = image.copy()
        is_torch = False

    # Step 1: Identify foreground region (non-zero or non-background voxels)
    if use_nonzero_only:
        # Use non-zero voxels as foreground
        foreground_mask = image_np != 0
    else:
        # Use all voxels
        foreground_mask = np.ones_like(image_np, dtype=bool)
    
    # Step 2: Calculate foreground percentiles and clip
    if np.any(foreground_mask):
        foreground_values = image_np[foreground_mask]
        lower_percentile, upper_percentile = foreground_percentiles
        lower_bound = np.percentile(foreground_values, lower_percentile)
        upper_bound = np.percentile(foreground_values, upper_percentile)
        
        # Clip to percentile range
        image_np = np.clip(image_np, lower_bound, upper_bound)
    else:
        # If no foreground, use default clipping
        image_np = np.clip(image_np, -1000, 1000)

    # Step 3: Z-score normalization
    if use_nonzero_only and np.any(foreground_mask):
        # Update foreground mask after clipping
        foreground_mask = image_np != 0
        if np.any(foreground_mask):
            foreground_values = image_np[foreground_mask]
            
            # Use global stats if provided, otherwise use image stats
            if global_mean is not None and global_std is not None:
                mean = global_mean
                std = global_std
            else:
                mean = np.mean(foreground_values)
                std = np.std(foreground_values)
            
            # Avoid division by zero
            if std > 1e-8:
                image_np = (image_np - mean) / std
            else:
                image_np = image_np - mean
                
    else:
        # Use all voxels for statistics
        if global_mean is not None and global_std is not None:
            mean = global_mean
            std = global_std
        else:
            mean = np.mean(image_np)
            std = np.std(image_np)
        
        if std > 1e-8:
            image_np = (image_np - mean) / std
        else:
            image_np = image_np - mean
    # Convert back to torch if input was torch
    if is_torch:
        return torch.from_numpy(image_np).to(image.device if hasattr(image, 'device') else 'cpu')
    
    return image_np

def ct_normalize_for_diffusion(image, foreground_percentiles=(0.5, 99.5), global_mean=None, global_std=None, use_nonzero_only=True, clip_z=2.5):

    # Step 1: nnUNet-style normalization (your original function)
    image = ct_normalize_nnunet(
        image,
        foreground_percentiles=foreground_percentiles,
        global_mean=global_mean,
        global_std=global_std,
        use_nonzero_only=use_nonzero_only
    )

    # Step 2: squash to [-1, 1] (diffusion-friendly)
    if isinstance(image, torch.Tensor):
        image = torch.clamp(image, -clip_z, clip_z)
        image = image / clip_z
    else:
        image = np.clip(image, -clip_z, clip_z)
        image = image / clip_z

    return image
    