"""
3D CTA datasets for ReScan-IA training and inference.

NIfTI volumes are stored in (X, Y, Z) order and converted to network tensors
in (1, D, H, W) = (1, Z, Y, X) order before being returned.
"""
import os

import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import center_of_mass, label

from .util.ct_normalization import (
    ct_normalize_for_diffusion,
    ct_normalize_nnunet,
    ct_normalize_simple,
)
from .util.nifti_3d_mask import sphere3d_to_mask

NIFTI_EXTENSIONS = ('.nii.gz', '.nii', '.NII.GZ', '.NII')


def is_nifti_file(filename):
    return any(filename.endswith(ext) for ext in NIFTI_EXTENSIONS)


def make_nifti_dataset(data_root):
    """List NIfTI files under a directory, or read paths from a text file."""
    if os.path.isfile(data_root):
        with open(data_root, 'r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'{data_root} is not a valid directory')

    files = []
    for root, _, fnames in sorted(os.walk(data_root)):
        for fname in sorted(fnames):
            if is_nifti_file(fname):
                files.append(os.path.join(root, fname))
    return files


def extract_series_uid(filename):
    """Return the SeriesInstanceUID prefix before the first underscore."""
    basename = os.path.basename(filename)
    for ext in NIFTI_EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    return basename.split('_')[0] if '_' in basename else basename


def build_uid_index(files):
    """Map SeriesInstanceUID to the first matching file path."""
    uid_to_path = {}
    for path in files:
        uid = extract_series_uid(path)
        uid_to_path.setdefault(uid, path)
    return uid_to_path


def build_paired_samples(data_root, aneurysm_root, vessel_root=None):
    """
    Pair CTA volumes with aneurysm masks (required) and vessel masks (optional)
    using SeriesInstanceUID.
    """
    uid_to_image = build_uid_index(make_nifti_dataset(data_root))
    uid_to_mask = build_uid_index(make_nifti_dataset(aneurysm_root))
    uid_to_vessel = (
        build_uid_index(make_nifti_dataset(vessel_root)) if vessel_root else {}
    )

    samples = []
    for uid, image_path in uid_to_image.items():
        if uid not in uid_to_mask:
            continue
        sample = {
            'file_path': image_path,
            'mask_path': uid_to_mask[uid],
            'uid': uid,
        }
        if uid in uid_to_vessel:
            sample['vessel_path'] = uid_to_vessel[uid]
        samples.append(sample)
    return samples


def normalize_ct_volume(image, normalization, hu_min, hu_max,
                        foreground_percentiles, global_mean, global_std):
    """Normalize a CT volume to the range expected by the diffusion model."""
    if normalization == 'nnunet':
        return ct_normalize_nnunet(
            image,
            foreground_percentiles=foreground_percentiles,
            global_mean=global_mean,
            global_std=global_std,
            use_nonzero_only=True,
        )
    if normalization == 'nn_diffusion':
        return ct_normalize_for_diffusion(
            image,
            foreground_percentiles=foreground_percentiles,
            global_mean=global_mean,
            global_std=global_std,
            use_nonzero_only=True,
        )
    return ct_normalize_simple(image, hu_min, hu_max)


def pad_volumes_to_min_shape(volumes, min_shape):
    """
    Pad (X, Y, Z) volumes to at least min_shape = (crop_w, crop_h, crop_d).
    Returns padded volumes and the updated spatial shape.
    """
    width, height, depth = volumes[0].shape
    crop_w, crop_h, crop_d = min_shape

    pad_w = max(0, crop_w - width)
    pad_h = max(0, crop_h - height)
    pad_d = max(0, crop_d - depth)
    if pad_w == 0 and pad_h == 0 and pad_d == 0:
        return volumes, (width, height, depth)

    pad_spec = (
        (pad_w // 2, pad_w - pad_w // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_d // 2, pad_d - pad_d // 2),
    )
    padded = [np.pad(volume, pad_spec, mode='constant', constant_values=0) for volume in volumes]
    return padded, padded[0].shape


def crop_volumes(volumes, origin, crop_size):
    """Crop aligned (X, Y, Z) volumes using origin = (x, y, z) and crop_size = (w, h, d)."""
    x0, y0, z0 = origin
    crop_w, crop_h, crop_d = crop_size
    return [
        volume[x0:x0 + crop_w, y0:y0 + crop_h, z0:z0 + crop_d]
        for volume in volumes
    ]


def xyz_to_tensor(volume):
    """Convert an (X, Y, Z) numpy volume to a (1, D, H, W) torch tensor."""
    return torch.from_numpy(volume).float().permute(2, 1, 0).unsqueeze(0)


def pack_inpaint_batch(image, mask, seg, vessel=None, file_path='', uid=''):
    """Build the standard batch dictionary consumed by the ReScanIA model."""
    image_t = xyz_to_tensor(image)
    mask_t = xyz_to_tensor(mask)
    seg_t = xyz_to_tensor(seg)

    batch = {
        'gt_image': image_t,
        'cond_image': image_t * (1.0 - mask_t),
        'mask_image': image_t * (1.0 - mask_t) + mask_t,
        'mask': mask_t,
        'seg': seg_t,
        'path': os.path.basename(file_path),
        'uid': uid,
    }
    if vessel is not None:
        batch['vessel_seg'] = xyz_to_tensor(vessel)
    return batch


def generate_training_inpaint_mask(seg_mask, mask_size_range):
    """
    Sample a spherical inpainting mask centered on a random voxel inside the
    aneurysm segmentation mask.
    """
    width, height, depth = seg_mask.shape
    mask = np.zeros((width, height, depth), dtype=np.uint8)

    seg_positions = np.where(seg_mask > 0)
    if len(seg_positions[0]) == 0:
        return mask

    pos_idx = np.random.randint(0, len(seg_positions[0]))
    center_x = int(seg_positions[0][pos_idx])
    center_y = int(seg_positions[1][pos_idx])
    center_z = int(seg_positions[2][pos_idx])

    size_x = np.random.randint(mask_size_range[0], mask_size_range[1] + 1)
    size_y = np.random.randint(mask_size_range[0], mask_size_range[1] + 1)
    size_z = np.random.randint(mask_size_range[0], mask_size_range[1] + 1)
    radius = float(np.mean([size_x, size_y, size_z])) / 2.0
    return sphere3d_to_mask(
        img_shape=(width, height, depth),
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        radius=radius,
    )


def generate_inference_inpaint_mask(aneurysm_mask, seg_mask, rng):
    """
    Build a spherical inpainting mask that fully covers the transplanted aneurysm.
    """
    shape = aneurysm_mask.shape
    mask = np.zeros(shape, dtype=np.uint8)

    aneurysm_points = np.argwhere(aneurysm_mask > 0)
    if len(aneurysm_points) == 0:
        return mask

    com = center_of_mass(aneurysm_mask)
    center = np.array([int(round(c)) for c in com], dtype=np.float32)
    min_radius = float(np.linalg.norm(aneurysm_points - center[None, :], axis=1).max())
    margin = 1.0
    radius = min_radius + margin + float(rng.uniform(0.0, 1.0))

    mask = sphere3d_to_mask(
        img_shape=shape,
        center_x=int(center[0]),
        center_y=int(center[1]),
        center_z=int(center[2]),
        radius=radius,
    )

    if np.any((seg_mask > 0) & (mask == 0)):
        mask = sphere3d_to_mask(
            img_shape=shape,
            center_x=int(center[0]),
            center_y=int(center[1]),
            center_z=int(center[2]),
            radius=min_radius + margin + 3.0,
        )
    return mask


def transplant_aneurysm_cc(source_mask, target_shape, target_coord, rng):
    """Translate one connected aneurysm component to a target coordinate."""
    labeled, num_components = label(source_mask > 0)
    if num_components == 0:
        raise ValueError('No aneurysm connected component found in source mask')

    component_id = 1 if num_components == 1 else int(rng.integers(1, num_components + 1))
    component_points = np.argwhere(labeled == component_id)
    shift = np.round(np.asarray(target_coord) - component_points.mean(axis=0)).astype(int)

    shifted_points = component_points + shift
    valid = np.all((shifted_points >= 0) & (shifted_points < np.asarray(target_shape)), axis=1)
    shifted_points = shifted_points[valid]

    transplanted = np.zeros(target_shape, dtype=np.uint8)
    transplanted[shifted_points[:, 0], shifted_points[:, 1], shifted_points[:, 2]] = 1
    return transplanted.astype(np.float32)


class ReScanIATrainDataset(data.Dataset):
    """
    Training dataset for ReScan-IA.

    Loads paired CTA volumes, aneurysm masks, and vessel masks; normalizes the
    CT volume; samples a random inpainting mask inside the aneurysm region; and
    returns a random 96^3 patch when image_size is set.
    """

    def __init__(
        self,
        data_root,
        aneurysm_root,
        vessel_root=None,
        mask_size_range=(10, 30),
        data_len=-1,
        hu_min=-50,
        hu_max=450,
        normalization='ct_normalize_simple',
        foreground_percentiles=(0.5, 99.5),
        global_mean=None,
        global_std=None,
        image_size=None,
    ):
        super().__init__()

        self.samples = build_paired_samples(data_root, aneurysm_root, vessel_root)
        if data_len > 0:
            self.samples = self.samples[:int(data_len)]

        self.mask_size_range = mask_size_range
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.normalization = normalization
        self.foreground_percentiles = foreground_percentiles
        self.global_mean = global_mean
        self.global_std = global_std

        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 3:
            self.image_size = tuple(image_size)  # (D, H, W) = (Z, Y, X)
        else:
            raise ValueError(f'image_size must be [D, H, W], got: {image_size}')

        print(f'[Train] Loaded {len(self.samples)} paired samples')
        if vessel_root is not None:
            vessel_count = sum(1 for sample in self.samples if 'vessel_path' in sample)
            print(f'[Train] {vessel_count} samples include vessel masks')

    def __len__(self):
        return len(self.samples)

    def _crop_around_seg(self, image, mask, seg_mask, vessel_mask=None):
        """Random crop that keeps part of the aneurysm segmentation in view."""
        crop_d, crop_h, crop_w = self.image_size
        crop_size_xyz = (crop_w, crop_h, crop_d)

        volumes = [image, mask, seg_mask] if vessel_mask is None else [image, mask, seg_mask, vessel_mask]
        volumes, (width, height, depth) = pad_volumes_to_min_shape(volumes, crop_size_xyz)
        image, mask, seg_mask = volumes[:3]
        vessel_mask = volumes[3] if vessel_mask is not None else None

        seg_positions = np.where(seg_mask > 0) if np.any(seg_mask > 0) else np.where(mask > 0)
        if len(seg_positions[0]) == 0:
            x0 = np.random.randint(0, max(1, width - crop_w + 1))
            y0 = np.random.randint(0, max(1, height - crop_h + 1))
            z0 = np.random.randint(0, max(1, depth - crop_d + 1))
        else:
            anchor_idx = np.random.randint(0, len(seg_positions[0]))
            anchor = (
                seg_positions[0][anchor_idx],
                seg_positions[1][anchor_idx],
                seg_positions[2][anchor_idx],
            )

            def crop_start(anchor_pos, crop_size, full_size):
                low = 1.0 / 3.0
                high = 2.0 / 3.0
                min_start = max(0, int(anchor_pos - crop_size * high))
                max_start = min(full_size - crop_size, int(anchor_pos - crop_size * low))
                if min_start > max_start:
                    centered = max(0, min(full_size - crop_size, int(anchor_pos - crop_size // 2)))
                    return centered
                return np.random.randint(min_start, max_start + 1)

            x0 = crop_start(anchor[0], crop_w, width)
            y0 = crop_start(anchor[1], crop_h, height)
            z0 = crop_start(anchor[2], crop_d, depth)

        origin = (x0, y0, z0)
        cropped = crop_volumes([image, mask, seg_mask], origin, crop_size_xyz)
        if vessel_mask is not None:
            cropped.append(crop_volumes([vessel_mask], origin, crop_size_xyz)[0])
        return cropped

    def __getitem__(self, index):
        sample = self.samples[index]

        image = np.asarray(nib.load(sample['file_path']).get_fdata(), dtype=np.float32)
        seg_mask = (nib.load(sample['mask_path']).get_fdata() > 0).astype(np.float32)
        vessel_mask = None
        if 'vessel_path' in sample:
            vessel_mask = (nib.load(sample['vessel_path']).get_fdata() > 0).astype(np.float32)

        if image.ndim != 3:
            raise ValueError(f'Expected a 3D CTA volume, got shape {image.shape}')
        if seg_mask.shape != image.shape:
            raise ValueError(
                f'Aneurysm mask shape {seg_mask.shape} does not match image shape {image.shape}'
            )
        if vessel_mask is not None and vessel_mask.shape != image.shape:
            raise ValueError(
                f'Vessel mask shape {vessel_mask.shape} does not match image shape {image.shape}'
            )

        image = normalize_ct_volume(
            image,
            self.normalization,
            self.hu_min,
            self.hu_max,
            self.foreground_percentiles,
            self.global_mean,
            self.global_std,
        )
        inpaint_mask = generate_training_inpaint_mask(seg_mask, self.mask_size_range)

        if self.image_size is not None:
            image, inpaint_mask, seg_mask, *rest = self._crop_around_seg(
                image, inpaint_mask, seg_mask, vessel_mask
            )
            vessel_mask = rest[0] if rest else None

        return pack_inpaint_batch(
            image,
            inpaint_mask,
            seg_mask,
            vessel=vessel_mask,
            file_path=sample['file_path'],
            uid=sample['uid'],
        )


class ReScanIAInferenceDataset(data.Dataset):
    """
    Inference dataset for ReScan-IA.

    Transplants an aneurysm mask from a donor pool into each CTA volume at a
    user-specified coordinate, builds a spherical inpainting mask around the
    transplant, and returns a larger patch (typically 128^3) for synthesis.
    """

    def __init__(
        self,
        data_root,
        aneurysm_root,
        vessel_root=None,
        mask_size_range=(33, 34),
        data_len=-1,
        hu_min=-50,
        hu_max=450,
        normalization='ct_normalize_simple',
        foreground_percentiles=(0.5, 99.5),
        global_mean=None,
        global_std=None,
        image_size=None,
        testset_aneurysm_root=None,
        target_coord=None,
    ):
        super().__init__()

        self.samples = build_paired_samples(data_root, aneurysm_root, vessel_root)
        if data_len > 0:
            self.samples = self.samples[:int(data_len)]

        if not testset_aneurysm_root:
            raise ValueError('testset_aneurysm_root must be provided for inference')
        if target_coord is None:
            raise ValueError('target_coord must be provided for inference')

        self.test_uid_to_aneurysm = {
            extract_series_uid(path): path
            for path in make_nifti_dataset(testset_aneurysm_root)
        }
        if not self.test_uid_to_aneurysm:
            raise ValueError(f'No aneurysm masks found in {testset_aneurysm_root}')

        self._donor_uid_pool = list(self.test_uid_to_aneurysm.keys())
        self.target_coord = list(target_coord)
        self.mask_size_range = mask_size_range
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.normalization = normalization
        self.foreground_percentiles = foreground_percentiles
        self.global_mean = global_mean
        self.global_std = global_std
        self.image_size = tuple(image_size) if image_size is not None else None
        self.rng = np.random.default_rng()

        print(f'[Inference] Loaded {len(self.samples)} CTA volumes')
        print(f'[Inference] Donor aneurysm pool size: {len(self._donor_uid_pool)}')

    def __len__(self):
        return len(self.samples)

    def _sample_donor_mask(self):
        if not self._donor_uid_pool:
            raise RuntimeError('All donor aneurysm masks have been used')
        pool_idx = int(self.rng.integers(len(self._donor_uid_pool)))
        donor_uid = self._donor_uid_pool.pop(pool_idx)
        donor_mask = (nib.load(self.test_uid_to_aneurysm[donor_uid]).get_fdata() > 0).astype(np.uint8)
        return donor_mask, donor_uid

    def _sample_target_coord(self):
        base_coord = np.array(self.rng.choice(self.target_coord), dtype=int)
        offset = self.rng.integers(-1, 2, size=3)
        return list(base_coord + offset)

    def _crop_around_transplant(self, image, mask, seg_mask, transplant_mask, vessel_mask=None):
        """Center a crop on the transplanted aneurysm."""
        crop_d, crop_h, crop_w = self.image_size
        crop_size_xyz = (crop_w, crop_h, crop_d)

        volumes = [image, mask, seg_mask]
        volumes, (width, height, depth) = pad_volumes_to_min_shape(volumes, crop_size_xyz)
        image, mask, seg_mask = volumes
        if vessel_mask is not None:
            vessel_mask, _ = pad_volumes_to_min_shape([vessel_mask], crop_size_xyz)

        transplant_points = np.argwhere(transplant_mask > 0)
        if len(transplant_points) == 0:
            x0 = int(self.rng.integers(0, max(1, width - crop_w + 1)))
            y0 = int(self.rng.integers(0, max(1, height - crop_h + 1)))
            z0 = int(self.rng.integers(0, max(1, depth - crop_d + 1)))
        else:
            center = transplant_points.mean(axis=0)
            jitter = self.rng.uniform(-0.1, 0.1, size=3) * np.array([crop_w, crop_h, crop_d])

            def crop_start(center_coord, crop_size, full_size):
                start = int(round(center_coord - crop_size / 2))
                return max(0, min(full_size - crop_size, start))

            jittered_center = center + jitter
            x0 = crop_start(jittered_center[0], crop_w, width)
            y0 = crop_start(jittered_center[1], crop_h, height)
            z0 = crop_start(jittered_center[2], crop_d, depth)

        origin = (x0, y0, z0)
        image, mask, seg_mask = crop_volumes([image, mask, seg_mask], origin, crop_size_xyz)
        vessel_cropped = crop_volumes(vessel_mask, origin, crop_size_xyz)[0] if vessel_mask is not None else None
        return image, mask, seg_mask, vessel_cropped, origin

    def __getitem__(self, index):
        sample = self.samples[index]

        image = np.asarray(nib.load(sample['file_path']).get_fdata(), dtype=np.float32)
        original_aneurysm = (nib.load(sample['mask_path']).get_fdata() > 0).astype(np.float32)
        vessel_mask = None
        if 'vessel_path' in sample:
            vessel_mask = (nib.load(sample['vessel_path']).get_fdata() > 0).astype(np.float32)

        if image.ndim != 3:
            raise ValueError(f'Expected a 3D CTA volume, got shape {image.shape}')

        donor_mask, donor_uid = self._sample_donor_mask()
        target_coord = self._sample_target_coord()
        transplant_mask = transplant_aneurysm_cc(
            donor_mask,
            image.shape,
            target_coord,
            self.rng,
        )

        if vessel_mask is not None:
            vessel_mask = np.maximum(vessel_mask, transplant_mask)

        seg_mask = np.clip(original_aneurysm + transplant_mask, 0.0, 1.0)
        image_full = normalize_ct_volume(
            image,
            self.normalization,
            self.hu_min,
            self.hu_max,
            self.foreground_percentiles,
            self.global_mean,
            self.global_std,
        )
        inpaint_mask = generate_inference_inpaint_mask(transplant_mask, seg_mask, self.rng)

        crop_origin = (0, 0, 0)
        image_crop = image_full
        if self.image_size is not None:
            image_crop, inpaint_mask, seg_mask, vessel_mask, crop_origin = self._crop_around_transplant(
                image_full,
                inpaint_mask,
                seg_mask,
                transplant_mask,
                vessel_mask,
            )

        batch = pack_inpaint_batch(
            image_crop,
            inpaint_mask,
            seg_mask,
            vessel=vessel_mask,
            file_path=sample['file_path'],
            uid=sample['uid'],
        )
        batch['ori_image'] = xyz_to_tensor(image_full)
        batch['donor_uid'] = donor_uid
        batch['target_coord'] = target_coord
        batch['patch_origin_xyz'] = crop_origin
        return batch

