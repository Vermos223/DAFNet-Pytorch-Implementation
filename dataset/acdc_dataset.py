import json
import os
import logging
from typing import Sequence, Tuple, Union, List, Dict, Any, Callable

import h5py
import numpy as np
import cv2
import albumentations as A

import pandas as pd
import scipy.interpolate
import torch
from torch.utils import data as Data

import numpy as np
import numba

class NormalDistribution(object):
    def __init__(self):
        self.mu = 0
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples
    
norm = NormalDistribution()


# =============================================================================
# Suppress the redundant annoying logging information from library
# =============================================================================
# create a logger filter for the stupid albumentations warning
_logger = logging.getLogger('py.warnings')
_logger.addFilter(
    lambda record: False if 'albumentations/core/transforms_interface.py' in record.args[0] else True
)

# =============================================================================
# Dataset PATH
# =============================================================================
class _CINESliceBundleHDF5Images(object):
    def __init__(self, slice_dataset_path: str):
        self.slice_dataset_path = slice_dataset_path

    def load_h5_slice(self, h5filepath: str):
        slice_dict = {}
        with h5py.File(h5filepath, 'r') as f:
            if 'cine' in f:
                f_meta_data = json.loads(f.attrs['metadata'])
                slice_dict['cine_metadata'] = json.loads(f['/cine'].attrs['header_dict'])
                slice_dict['mask'] = f['/cine/mask'][:].astype(np.float32, copy=False)  # [H, W]
                slice_dict['image'] = f['/cine/array'][:].astype(np.float32, copy=False)

        return slice_dict

    def __getitem__(self, dataset_index: int):

        filename = f"{dataset_index:05d}.h5"
        h5filepath = os.path.join(self.slice_dataset_path, filename)
        slice_dict = self.load_h5_slice(h5filepath)
        slice_dict = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for (k, v) in slice_dict.items()}
        return slice_dict


class CINEBundleSliceDataset(Data.Dataset):
    def __init__(self, *, dataset_dirpath,
                 is_bbox_labeled=True,
                 partition_filepath='partition2patients.json'):
        
        dataset_index2slice = _CINESliceBundleHDF5Images(slice_dataset_path=os.path.join(dataset_dirpath, 'slice_dataset'))

        df_metadata = pd.read_csv(os.path.join(dataset_dirpath, 'infos/metadata.csv'), index_col='dataset_index')
        N_all = len(df_metadata)

        # partition indices according to patient names in different groups
        with open(os.path.join(dataset_dirpath, 'infos', partition_filepath)) as f:
            parition2patients = json.load(f)
        group2mask: Dict[int, Sequence[int]] = {}
        for group, patient_names in parition2patients.items():
            indices_mask = np.zeros_like(df_metadata.index, dtype=bool)
            for item in patient_names:
                patient_name = item[0]
                indices_mask = indices_mask | \
                               (df_metadata['patient_name'] == patient_name)
            group2mask[group] = indices_mask

        if is_bbox_labeled:
            df_info_bbox = pd.read_csv(
                os.path.join(dataset_dirpath, 'infos/heart_localization.csv'),
                index_col='dataset_index'
            )
        else:
            df_info_bbox=None

        self.dataset_index2slice = dataset_index2slice
        self.df_metadata = df_metadata
        self.is_bbox_labeled = is_bbox_labeled
        self.df_info_bbox = df_info_bbox

        self.group2mask = group2mask
        self.N_all = N_all

    def get_valid_indices(self, 
                          group_keys: Sequence[str],
                          ) -> List[int]:
        condition = np.ones_like(self.df_metadata.index, dtype=np.bool_)
        group_condition = np.stack([self.group2mask[key] for key in group_keys]).sum(axis=0) != 0
        condition &= group_condition
        indices = self.df_metadata.index[condition].tolist()
        indices.sort()
        return indices

    def __getitem__(self, dataset_index: int) -> dict:

        # use loc instead of iloc, becuase dataset_index does not a continuous integers from 0 to len(dataset)
        meta_info = self.df_metadata.loc[dataset_index, :]
        pair_dict = self.dataset_index2slice[dataset_index]

        extra_info = {
            'dataset_index': dataset_index,
            'sample_identifier': f"I{dataset_index}_"
                                 f"{meta_info['patient_name']}_{meta_info['slice_index']}_{meta_info['cardiac_phase']}",
        }

        if self.is_bbox_labeled and dataset_index in self.df_info_bbox.index:
            row = self.df_info_bbox.loc[dataset_index, :]
            bbox = [row.xmin, row.ymin, row.xmax, row.ymax]  # List[int*4]
            extra_info['bbox'] = bbox

        pair_dict.update(extra_info)
        return pair_dict

    def __len__(self):
        return self.N_all


def normalize_array(x, mode='std', eps=1e-16):
    """
    :param x: [...]
    """
    assert mode in {'minmax_zero_centered', 'minmax_zero2one', 'std'}
    if mode == 'minmax_zero2one':
        # range in [0, 1]
        vmin = x.min()
        vmax = x.max()
        x = (x - vmin) / (vmax - vmin + eps)
    elif mode == 'minmax_zero_centered':
        # range in [-1, 1]
        vmin = x.min()
        vmax = x.max()
        x = (x - vmin) / (vmax - vmin + eps)
        x = x * 2 - 1
    elif mode == 'std':
        x = (x - x.mean()) / (x.std() + eps)
    else:
        raise RuntimeError(f"Unexpected normalization mode {mode}")
    return x


def _albumentataion_optional_apply(transform: Callable, **optional_params: Dict[str, Any]):
    input_dict = {k: v for (k, v) in optional_params.items() if v is not None}
    x = transform(**input_dict)
    return x


class AffineAugmentation(object):
    def __init__(self, mode='train'):
        if mode == 'train':
            Tlist = [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    rotate=(0, 360),
                    shear=(0, 10),
                    scale=(0.8, 1.0),
                    translate_px=(0, 4),
                    p=1.0
                )
            ]
        elif mode == 'val':
            Tlist = []
        else:
            raise NotImplemented

        self.T = A.Compose(
            Tlist,
            bbox_params=A.BboxParams(format='albumentations', label_fields=[])
        )

    def __call__(self, images, mask=None, bboxes=None, 
                 ):
        """
        :param images: ndarray[N, H, W] float32 (value in 0-1)
        :param mask: ndarray[N, H, W] float32 (value in 0-1)
        :param bboxes: List[[float] * 4] (value in 0-1)
        :param keypoints: List[[float] * 3]
        """
        N = len(images)
        assert N >= 1

        images = images.transpose(1, 2, 0)
        if mask is not None:
            mask = mask.transpose(1, 2, 0)

        x = _albumentataion_optional_apply(self.T,
                                           image=images, 
                                           mask=mask,
                                           bboxes=bboxes, 
                                           )
        retvals = []
        retvals.append(x['image'].transpose(2, 0, 1))

        if mask is not None:
            retvals.append(x['mask'].transpose(2, 0, 1))
        if bboxes is not None:
            retvals.append(x['bboxes'])
        return tuple(retvals)


class CINESlices4Segmentation(Data.Dataset):
    def __init__(self, 
                 index2slice_dataset: Data.Dataset,
                 *,
                 is_augmentation: bool = False,
                 is_roi_resample=True,
                 roi_resample_size=(128, 128),  # [width (x), height (y)]
                 roi_fixed_pixel_spacing: Tuple[float, float] = (0.89, 0.89),  # in x (column) and y (row) order
                 modality_vec_dim: int = 8
                 ):
        self.index2slice_dataset = index2slice_dataset
        self.num_slices = len(index2slice_dataset)
        self.is_roi_resample = is_roi_resample
        self.roi_resample_size = roi_resample_size
        self.roi_fixed_pixel_spacing = roi_fixed_pixel_spacing
        self.modality_vec_dim = modality_vec_dim
        self.T = AffineAugmentation(mode='train' if is_augmentation else 'val')

    def crop_and_resize_to_roi(self, images: np.ndarray, bbox):
        """ Resample input images to ROI using a bounding box
        :param images: [N, H, W]
        :param bbox: [4 (xmin, ymin, xmax, ymax), ] in pixels
        :return: [N, Hroi, Wroi]
        """
        N, H, W = images.shape
        xmin, ymin, xmax, ymax = bbox

        images_roi = images[:, max(ymin, 0):min(ymax, H), max(xmin, 0):min(xmax, W)]
        images_roi = np.pad(images_roi,
                            ((0, 0),
                             (abs(min(ymin, 0)), abs(min(H - ymax, 0))),
                             (abs(min(xmin, 0)), abs(min(W - xmax, 0)))),
                            mode='constant')

        Wroi, Hroi = self.roi_resample_size
        images_roi = cv2.resize(images_roi.transpose([1, 2, 0]), (Wroi, Hroi), interpolation=cv2.INTER_LINEAR)
        if N == 1:
            # workaround for cv2.resize will consume the single value axis, we always keep dims
            images_roi = images_roi[:, :, None]
        images_roi = images_roi.transpose([2, 0, 1])
        return images_roi

    def nearest_interp1d(self, xs, ys, new_xs):
        """
        :param ys: [N,]
        :param xs: [N,]
        :param new_xs: [N,]
        """
        assert len(ys) == len(xs) == len(new_xs)
        f = scipy.interpolate.interp1d(xs, ys, kind='nearest', fill_value='extrapolate')
        new_ys = f(new_xs)
        return new_ys

    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        # ===== Data Extraction
        pair_dict: dict = self.index2slice_dataset[index]
        
        if 'bbox' in pair_dict:
            bbox = pair_dict['bbox'].copy()
            # normalize bbox
            bbox = np.clip(bbox, 0, 1).tolist()
            roi_enabled = True
        else:
            bbox = [0.0, 0.0, 1.0, 1.0]
            roi_enabled = False
            
        t_images = []
        t_masks = []
        
        mask = pair_dict['mask'].astype(np.float32, copy=True)  # [H, W]
        t_masks.extend([mask])

        image = normalize_array(pair_dict['image'].astype(np.float32, copy=True), mode='minmax_zero2one')  # [H, W]
        t_images.extend([image])
            
        sample_identifier = pair_dict['sample_identifier']
        
        # ===== prepare data
        t_images = np.stack(t_images)  # [N, H, W]
        H, W = t_images.shape[-2:]
        t_masks = t_masks + [np.ones([H, W])]  # Add mask for image location
        t_masks = np.stack(t_masks)
        
        original_bbox = bbox.copy()
        original_images = t_images.copy()
        original_masks = t_masks.copy()
        # ===== Combined Augmentation
        t_images, t_masks, t_bboxes= self.T(
            t_images,
            mask=t_masks,
            bboxes=[bbox],
        )
        if len(t_bboxes) > 0:
            bbox = t_bboxes[0]
        else:
            t_images = original_images
            t_masks = original_masks
            bbox = original_bbox
        
        # ===== resample to ROI using bbox (after augmentation)
        if self.is_roi_resample and roi_enabled and bbox is not None:
            # compute roi bbox
            xmin, ymin, xmax, ymax = bbox            
            if self.roi_fixed_pixel_spacing is not None:
                # dicom store Pixel Spacing in [row spacing, col spacing] order
                
                cine_pixel_spacing = np.array(pair_dict['cine_metadata']['zooms'][:2]).astype(np.float32, copy=True)
                spacing_x_src, spacing_y_src = cine_pixel_spacing[1], cine_pixel_spacing[0]
                spacing_x_dst, spacing_y_dst = self.roi_fixed_pixel_spacing
                width = self.roi_resample_size[0] * (spacing_x_dst / spacing_x_src)
                height = self.roi_resample_size[1] * (spacing_y_dst / spacing_y_src)
            else:
                raise RuntimeError(f"`roi_fixed_pixel_spacing` must be specified for ROI computing.")
            center_x, center_y = (xmax + xmin) * W / 2, (ymax + ymin) * H / 2
            xmin = int(center_x - 0.5 * width)
            ymin = int(center_y - 0.5 * height)
            xmax = int(center_x + 0.5 * width)
            ymax = int(center_y + 0.5 * height)
            bbox_pixel = [xmin, ymin, xmax, ymax]
            # resample to bounding box
            t_images = self.crop_and_resize_to_roi(t_images, bbox_pixel)
            t_masks = self.crop_and_resize_to_roi(t_masks, bbox_pixel)
        
        image = normalize_array(t_images[0], mode='minmax_zero2one')
        
        mask = t_masks[0]
                 
        # image_location_mask = t_masks[-1]
        
        # background
        # mask_bg = mask.copy()
        # mask_bg[mask != 0] = 0
        # mask_bg[mask == 0] = 1

        # right ventricle
        # mask_rv = mask.copy()
        # mask_rv[mask != 1] = 0
        # mask_rv[mask == 1] = 1

        # myocardium
        mask_myo = mask.copy()
        mask_myo[mask != 2] = 0
        mask_myo[mask == 2] = 1

        # left ventricle
        mask_lv = mask.copy()
        mask_lv[mask != 3] = 0
        mask_lv[mask == 3] = 1
        
        # mask_concat = np.stack([mask_lv, mask_myo, mask_rv], axis=0)
        mask_concat = np.stack([mask_lv, mask_myo], axis=0)
        
        z_input = norm.sample(self.modality_vec_dim)
        sample = {
            'sample_identifier': sample_identifier,
            'cine': image[None, :, :].astype(np.float32, copy=False),  # [C=1, H, W]
            # 'image_location_mask': image_location_mask.astype(bool, copy=False),  # [H, W]
            # 'cine_mask_original': mask[None, ...].astype(np.float32, copy=False),  # [H, W]
            'cine_mask': mask_concat.astype(np.float32, copy=False),  # [C=3, H, W]
            'cine_z_input': z_input.astype(np.float32, copy=False),  # [modality_vec_dim,]
            
            # modality 2 test
            # 'psir': image[None, :, :].astype(np.float32, copy=False),  # [C=1, H, W]
            # # 'image_location_mask': image_location_mask.astype(bool, copy=False),  # [H, W]
            # 'psir_mask_original': mask.astype(np.float32, copy=False),  # [H, W]
            # 'psir_mask': mask_concat.astype(np.float32, copy=False),  # [C=4, H, W]
            # 'psir_z_input': z_input.astype(np.float32, copy=False),  # [modality_vec_dim,]
        }
        return sample