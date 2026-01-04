import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import nibabel as nib

from .util.ct_normalization import ct_normalize_simple, ct_normalize_nnunet, ct_normalize_for_diffusion
from .util.nifti_3d_mask import bbox3d_to_mask, sphere3d_to_mask

NIFTI_EXTENSIONS = ['.nii.gz', '.nii', '.NII.GZ', '.NII']

def is_nifti_file(filename):
    """Check if file is a NIfTI file"""
    return any(filename.endswith(ext) for ext in NIFTI_EXTENSIONS)

def make_nifti_dataset(data_root):
    """
    Create dataset list from directory containing NIfTI files.
    
    Args:
        data_root: root directory containing NIfTI files
    
    Returns:
        list: list of file paths
    """
    if os.path.isfile(data_root):
        # If it's a file, assume it's a list of file paths
        with open(data_root, 'r') as f:
            files = [line.strip() for line in f.readlines()]
        return files
    
    assert os.path.isdir(data_root), f'{data_root} is not a valid directory'
    files = []
    for root, _, fnames in sorted(os.walk(data_root)):
        for fname in sorted(fnames):
            if is_nifti_file(fname):
                path = os.path.join(root, fname)
                files.append(path)
    
    return files

def extract_series_uid(filename):
    """
    Extract SeriesInstanceUID from filename.
    
    Handles two formats:
    1. 1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317_iso0p5.nii.gz
    2. 1.2.826.0.1.3680043.8.498.10134365079002163886508836892471866754_iso0p5_t00.nii.gz
    
    Returns the UID part before the first underscore.
    """
    basename = os.path.basename(filename)
    # Remove extension
    for ext in NIFTI_EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    
    # Extract UID (part before first underscore)
    if '_' in basename:
        uid = basename.split('_')[0]
    else:
        uid = basename
    
    return uid

class Inpaint3DDataset(data.Dataset):
    """
    3D Inpainting Dataset for medical images with bounding box annotations.
    
    Args:
        data_root: root directory containing NIfTI files
        csv_path: path to CSV file containing bounding box annotations
        mask_size_range: tuple of (min_size, max_size) for random mask size (default: (10, 30))
        data_len: limit number of samples (-1 for all)
        hu_min: minimum HU value for simple normalization (default: -1000)
        hu_max: maximum HU value for simple normalization (default: 1000)
        normalization: normalization method, 'simple' or 'nnunet' (default: 'nnunet')
        global_mean: global mean for nnunet normalization (if None, use per-image mean)
        global_std: global std for nnunet normalization (if None, use per-image std)
    """
    
    def __init__(self, data_root, csv_path, mask_size_range=(10, 30), 
                 data_len=-1, hu_min=-1000, hu_max=1000,
                 normalization='nnunet', foreground_percentiles=(0.5, 99.5),
                 global_mean=None, global_std=None, image_size=None):
        super(Inpaint3DDataset, self).__init__()
        
        # Load NIfTI files
        nifti_files = make_nifti_dataset(data_root)
        
        # Load CSV with bounding box annotations
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['SeriesInstanceUID', 'new_coord_x', 'new_coord_y', 'new_coord_z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Build mapping from UID to bounding boxes
        uid_to_bboxes = {}
        for _, row in df.iterrows():
            uid = str(row['SeriesInstanceUID'])
            if uid not in uid_to_bboxes:
                uid_to_bboxes[uid] = []
            uid_to_bboxes[uid].append({
                'x': float(row['new_coord_x']),
                'y': float(row['new_coord_y']),
                'z': float(row['new_coord_z'])
            })
        
        # Create dataset: each NIfTI file is one sample, with all bounding boxes for that UID
        self.samples = []
        for nifti_file in nifti_files:
            uid = extract_series_uid(nifti_file)
            
            # Skip if no bounding boxes found
            if uid not in uid_to_bboxes or len(uid_to_bboxes[uid]) == 0:
                continue
            
            # Create one sample per NIfTI file with all bounding boxes
            self.samples.append({
                'file_path': nifti_file,
                'uid': uid,
                'bboxes': uid_to_bboxes[uid]  # All bounding boxes for this UID
            })
        
        if data_len > 0:
            self.samples = self.samples[:int(data_len)]
        
        self.mask_size_range = mask_size_range
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.normalization = normalization  # 'simple' or 'nnunet'
        self.foreground_percentiles = foreground_percentiles
        self.global_mean = global_mean
        self.global_std = global_std
        
        # Image size for random crop: [D, H, W] or None (no crop)
        if image_size is not None:
            if isinstance(image_size, (list, tuple)) and len(image_size) == 3:
                # Convert to (D, H, W) = (depth, height, width) = (Z, Y, X)
                self.image_size = tuple(image_size)  # (D, H, W)
            else:
                raise ValueError(f"image_size must be a 3-element list/tuple [D, H, W], got: {image_size}")
        else:
            self.image_size = None
        
        print(f"Loaded {len(self.samples)} samples from {len(nifti_files)} NIfTI files")
        print(f"Using normalization: {normalization}")
        if self.image_size is not None:
            print(f"Using random crop with size: {self.image_size} (D, H, W)")
    
    def __len__(self):
        return len(self.samples)
    
    def _random_crop_with_mask(self, image, mask):
        """
        Random crop that ensures the crop region contains at least part of the mask.
        
        Args:
            image: numpy array of shape (X, Y, Z) = (width, height, depth)
            mask: numpy array of shape (X, Y, Z) = (width, height, depth)
        
        Returns:
            cropped_image: numpy array of shape (X, Y, Z) = (crop_width, crop_height, crop_depth)
            cropped_mask: numpy array of shape (X, Y, Z) = (crop_width, crop_height, crop_depth)
        """
        width, height, depth = image.shape  # (X, Y, Z)
        crop_d, crop_h, crop_w = self.image_size  # (D, H, W) = (Z, Y, X)
        
        # If image is smaller than crop size, pad or use entire image
        if depth < crop_d or height < crop_h or width < crop_w:
            # Option 1: Pad to crop size
            pad_d = max(0, crop_d - depth)
            pad_h = max(0, crop_h - height)
            pad_w = max(0, crop_w - width)
            
            pad_d_before = pad_d // 2
            pad_d_after = pad_d - pad_d_before
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
            
            image = np.pad(image, ((pad_w_before, pad_w_after), (pad_h_before, pad_h_after), (pad_d_before, pad_d_after)), 
                          mode='constant', constant_values=0)
            mask = np.pad(mask, ((pad_w_before, pad_w_after), (pad_h_before, pad_h_after), (pad_d_before, pad_d_after)), 
                         mode='constant', constant_values=0)
            width, height, depth = image.shape
        
        # Find all mask positions (non-zero voxels)
        mask_positions = np.where(mask > 0)
        
        if len(mask_positions[0]) == 0:
            # No mask found, do regular random crop
            x_start = np.random.randint(0, max(1, width - crop_w + 1))
            y_start = np.random.randint(0, max(1, height - crop_h + 1))
            z_start = np.random.randint(0, max(1, depth - crop_d + 1))
        else:
            # Ensure crop contains at least part of the mask
            # Strategy: Randomly select a mask point, then ensure it's within the crop region
            mask_idx = np.random.randint(0, len(mask_positions[0]))
            mask_x = mask_positions[0][mask_idx]
            mask_y = mask_positions[1][mask_idx]
            mask_z = mask_positions[2][mask_idx]
            
            # Calculate valid crop start positions that include this mask point
            # The mask point should be within [start, start + crop_size)
            x_min = max(0, mask_x - crop_w + 1)
            x_max = min(width - crop_w + 1, mask_x + 1)
            y_min = max(0, mask_y - crop_h + 1)
            y_max = min(height - crop_h + 1, mask_y + 1)
            z_min = max(0, mask_z - crop_d + 1)
            z_max = min(depth - crop_d + 1, mask_z + 1)
            
            # Ensure valid range
            x_start = np.random.randint(x_min, max(x_min + 1, x_max))
            y_start = np.random.randint(y_min, max(y_min + 1, y_max))
            z_start = np.random.randint(z_min, max(z_min + 1, z_max))
        
        # Perform crop: (X, Y, Z) format
        cropped_image = image[x_start:x_start + crop_w, 
                             y_start:y_start + crop_h, 
                             z_start:z_start + crop_d]
        cropped_mask = mask[x_start:x_start + crop_w, 
                           y_start:y_start + crop_h, 
                           z_start:z_start + crop_d]
        
        return cropped_image, cropped_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]
        file_path = sample['file_path']
        bboxes = sample['bboxes']  # List of all bounding boxes for this UID
        
        # Load NIfTI image
        nii_img = nib.load(file_path)
        image_data = nii_img.get_fdata()
        
        # Handle different data types and orientations
        # Convert to numpy array if needed
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image_data)
        
        # Get image shape - NIfTI format from nibabel is typically (width, height, depth) = (X, Y, Z)
        # But we need to check the actual shape
        if len(image_data.shape) == 3:
            width, height, depth = image_data.shape  # (X, Y, Z)
        else:
            raise ValueError(f"Expected 3D image, got shape: {image_data.shape}")
        
        # Normalize CT image
        if self.normalization == 'nnunet':
            image_normalized = ct_normalize_nnunet(
                image_data, 
                foreground_percentiles=self.foreground_percentiles,
                global_mean=self.global_mean,
                global_std=self.global_std,
                use_nonzero_only=True
            )
        elif self.normalization == 'nn_diffusion':
            image_normalized = ct_normalize_for_diffusion(
                image_data, 
                foreground_percentiles=self.foreground_percentiles,
                global_mean=self.global_mean,
                global_std=self.global_std,
                use_nonzero_only=True
            )
        else:
            # Use simple normalization
            image_normalized = ct_normalize_simple(image_data, self.hu_min, self.hu_max)
        
        # Generate combined mask from all bounding boxes
        # Initialize mask as zeros - shape should match image_data: (width, height, depth) = (X, Y, Z)
        mask = np.zeros((width, height, depth), dtype=np.uint8)
        
        # Add mask for each bounding box
        for bbox in bboxes:
            # Randomly select mask size for each dimension
            size_x = np.random.randint(self.mask_size_range[0], self.mask_size_range[1] + 1)
            size_y = np.random.randint(self.mask_size_range[0], self.mask_size_range[1] + 1)
            size_z = np.random.randint(self.mask_size_range[0], self.mask_size_range[1] + 1)
            
            # 50% probability: use cuboid (current method), 50%: use sphere
            use_sphere = np.random.rand() < 0.5
            
            if use_sphere:
                # Use spherical mask (for aneurysm-like shapes)
                # Use average of sizes as radius, or use minimum for more conservative sphere
                radius = np.mean([size_x, size_y, size_z]) / 2.0
                # Alternatively, use max for larger sphere: radius = max(size_x, size_y, size_z) / 2.0
                
                bbox_mask = sphere3d_to_mask(
                    img_shape=(width, height, depth),
                    center_x=bbox['x'],
                    center_y=bbox['y'],
                    center_z=bbox['z'],
                    radius=radius
                )
            else:
                # Use cuboid mask (original method)
                bbox_mask = bbox3d_to_mask(
                    img_shape=(width, height, depth),
                    center_x=bbox['x'],
                    center_y=bbox['y'],
                    center_z=bbox['z'],
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z
                )
            
            # Combine masks (union of all bounding box masks)
            mask = np.maximum(mask, bbox_mask)
        
        # Apply random crop if image_size is specified
        if self.image_size is not None:
            image_normalized, mask = self._random_crop_with_mask(image_normalized, mask)
            # Update dimensions after crop
            width, height, depth = image_normalized.shape
        
        # Convert to torch tensor: shape is (X, Y, Z) = (width, height, depth)
        image_tensor = torch.from_numpy(image_normalized).float()
        
        # Permute to (Z, Y, X) = (depth, height, width) for 3D network
        # 3D UNet expects (C, D, H, W) format where D is depth/slice dimension
        image_tensor = image_tensor.permute(2, 1, 0)  # (X, Y, Z) -> (Z, Y, X) = (D, H, W)
        
        # Add channel dimension: (1, D, H, W) = (1, Z, Y, X)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Convert mask to torch tensor: shape is (X, Y, Z) = (width, height, depth)
        mask_tensor = torch.from_numpy(mask).float()
        
        # Permute to (Z, Y, X) = (depth, height, width) to match image_tensor
        mask_tensor = mask_tensor.permute(2, 1, 0)  # (X, Y, Z) -> (Z, Y, X) = (D, H, W)
        
        # Add channel dimension: (1, D, H, W) = (1, Z, Y, X)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # Create condition image (masked image with noise in masked region)
        cond_image = image_tensor * (1. - mask_tensor) + mask_tensor * torch.randn_like(image_tensor)
        
        # Create mask image for visualization
        mask_image = image_tensor * (1. - mask_tensor) + mask_tensor
        
        # Prepare return dictionary
        ret = {
            'gt_image': image_tensor,  # (1, D, H, W)
            'cond_image': cond_image,  # (1, D, H, W)
            'mask_image': mask_image,  # (1, D, H, W)
            'mask': mask_tensor,  # (1, D, H, W)
            'path': os.path.basename(file_path),
            'uid': sample['uid']
        }
        
        return ret

