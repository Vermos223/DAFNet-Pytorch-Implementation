import enum
import json
import os
import logging
import re
import numba
import itertools
import threading
from typing import Sequence, Tuple, Union, List, Dict, Any, Callable, Iterable

import h5py
import numpy as np
import cv2
import albumentations as A
import pandas as pd
import scipy.interpolate
import scipy.ndimage
import torch

from torch.nn import functional as nnf
from torch.utils import data as Data
from .utils import *
from einops import rearrange
from sklearn.metrics import mutual_info_score

# =============================================================================
# Suppress the redundant annoying logging information from library
# =============================================================================
# create a logger filter for the stupid albumentations warning
""" goes like: 
py.warnings - WARNING: /home/qianpf/miniconda3/envs/py39torch/lib/python3.9/site-packages/albumentations/core/transforms_interface.py:113: UserWarning: Affine could work incorrectly in ReplayMode for other input data because its' params depend on targets.
"""
_logger = logging.getLogger('py.warnings')
_logger.addFilter(
    lambda record: False if 'albumentations/core/transforms_interface.py' in record.args[0] else True
)

# =============================================================================
# Dataset PATH
# =============================================================================

DATASET_PATH = {
    'HCMVNE': '/public_bme2/bme-qihk/liaohx/PROJECTS/Cine_Generated_Enhancement/Data/HCM_20250519',
    'ZCMU_DCM': '/public_bme2/bme-qihk/liaohx/PROJECTS/Cine_Generated_Enhancement/Data/ZCMU_DCM/processed_batch1',
}

norm = NormalDistribution()

class _VNESliceBundleHDF5Images(object):
    def __init__(self,
                 slice_dataset_path: str,
                 protocols: Sequence[str] = ('cine', 'psir', 't1native'),
                 psir_type: str = 'original'
                 ):
        assert all(name in {'cine', 'psir', 't1native'} for name in protocols)
        assert psir_type in {'original', 'registered_to_cine_closest_tt', 'registered_to_cine_max_zncc'}

        self.slice_dataset_path = slice_dataset_path
        self.protocols = protocols
        self.psir_type = psir_type

    def load_h5_slice(self, h5filepath: str):
        slice_dict = {}
        with h5py.File(h5filepath, 'r') as f:
            if 'cine' in self.protocols and 'cine' in f:
                if 'psir' in self.protocols:
                    if self.psir_type == 'registered_to_cine_max_zncc':
                        slice_dict['cine_idx'] = int(f['/psir/array_registered_to_cine_max_zncc'].attrs['cine_idx'])
                    else:
                        slice_dict['cine_idx'] = int(f['/psir/array_registered_to_cine_closest_tt'].attrs['cine_idx'])
                else:
                    slice_dict['cine_idx'] = 0
                slice_dict['cine'] = f['/cine/array'][slice_dict['cine_idx']].astype(np.float32, copy=False)  # [H, W]
                slice_dict['cine_mask_128'] = f['/cine/mask_128'][:].astype(np.float32, copy=False)  # [128, 128]
                slice_dict['cine_mask_192'] = f['/cine/mask_192'][:].astype(np.float32, copy=False)  # [192, 192]
                slice_dict['cine_dicoms_metadata'] = json.loads(f['/cine'].attrs['jsonstr_dicom_metadata_dicts'])
                
            if 'psir' in self.protocols and 'psir' in f:
                if self.psir_type == 'original':
                    psir_array = f['/psir/array'][:][0, :, :].astype(np.float32, copy=False)  # [H, W]
                elif self.psir_type == 'registered_to_cine_closest_tt':
                    # if there is no registered psir, we step back to original psir
                    if '/psir/array_registered_to_cine_closest_tt' in f:
                        psir_array = f['/psir/array_registered_to_cine_closest_tt'][:].astype(np.float32, copy=False)  # [H, W]
                    else:
                        psir_array = f['/psir/array'][:][0, :, :].astype(np.float32, copy=False)  # [H, W]
                elif self.psir_type == 'registered_to_cine_max_zncc':
                    if '/psir/array_registered_to_cine_max_zncc' in f:
                        psir_array = f['/psir/array_registered_to_cine_max_zncc'][:].astype(np.float32, copy=False)  # [H, W]
                    else:
                        psir_array = f['/psir/array'][:][0, :, :].astype(np.float32, copy=False)  # [H, W]
                else:
                    raise NotImplementedError(f"PSIR type {self.psir_type} is not implemented.")
                
                slice_dict['psir'] = psir_array  # [H, W]
                slice_dict['psir_dicom_metadata'] = json.loads(f['/psir'].attrs['jsonstr_dicom_metadata_dicts'])[0]
                slice_dict['psir_pixel_spacing'] = np.array(slice_dict['psir_dicom_metadata']['00280030']['Value'])  # (2,) [row spacing, column spacing]

        return slice_dict

    def __getitem__(self, dataset_index: int):
        filename = f"{dataset_index:05d}.h5"
        h5filepath = os.path.join(self.slice_dataset_path, filename)
        slice_dict = self.load_h5_slice(h5filepath)
        return slice_dict


class VNEBundleSliceDataset_HCM(Data.Dataset):
    """
    pair_dict structure:
        cine_images: <class 'numpy.ndarray'> [T, H, W]
        psir_image: <class 'numpy.ndarray'> [H, W]
        t1native_images: <class 'numpy.ndarray'> [N_TI, H, W]
    """
    def __init__(self, *,
                 dataset_dirpath=DATASET_PATH['HCMVNE'],
                 patient_partition_file: str = 'partition2patients.json',
                 protocols: Sequence[str] = ('psir', 'cine', 't1native'),  # protocols need to be loaded
                 psir_type: str = 'original'):
        dataset_index2slice = _VNESliceBundleHDF5Images(
            slice_dataset_path=os.path.join(dataset_dirpath, 'slice_dataset'),
            protocols=protocols,
            psir_type=psir_type
        )
        
        df_metadata = pd.read_csv(os.path.join(dataset_dirpath, 'infos/metadata.csv'), index_col='dataset_index')
        N_all = len(df_metadata)

        df_info_tags = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/manually_quality_control.csv'),
            index_col='dataset_index'
        ).astype(np.bool_)

        df_info_blood_pool = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/blood_pool_classification.csv'),
            index_col='dataset_index'
        )
        assert len(df_info_blood_pool) == len(df_metadata), \
            f"Blood Pool labels have missing, please check the correctness of files."

        # partition indices according to patient names in different groups
        with open(os.path.join(dataset_dirpath, f'infos/{patient_partition_file}')) as f:
            parition2patients = json.load(f)
        group2mask: Dict[int, Sequence[int]] = {}
        for group, patient_names in parition2patients.items():
            indices_mask = np.zeros_like(df_metadata.index, dtype=bool)
            for (institution_name, patient_name) in patient_names:
                indices_mask = indices_mask | \
                               ((df_metadata['InstitutionName'] == institution_name) &
                                (df_metadata['PatientName'] == patient_name))
            group2mask[group] = indices_mask

        df_info_bbox = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/heart_localization.csv'),
            index_col='dataset_index'
        )

        self.dataset_index2slice = dataset_index2slice
        self.df_metadata = df_metadata
        self.df_info_tags = df_info_tags
        self.df_info_bbox = df_info_bbox
        self.df_info_blood_pool = df_info_blood_pool
        self.group2mask = group2mask
        self.N_all = N_all

    def get_valid_indices(self, 
                          group_keys: Sequence[str],
                          mandatory_protocols: Sequence[str],
                          requirements: Sequence[str] = ()
                          ) -> List[int]:
        """
        Classify the dataset into slices level partition otherwise train/val dataset may have patients interleaved.

        There are two types of reasons to excluding samples in this dataset:
            1. There is no suitable bbox label.
            2. Match certain predefined reasons in exclusion matrix file.
        """
        # only_apex, base_position,
        condition = np.ones_like(self.df_metadata.index, dtype=np.bool_)

        for protocol in mandatory_protocols:
            key_name = f"{protocol}__SeriesNumber"
            condition &= ~np.isnan(self.df_metadata[key_name])

        for item in requirements:
            if item == 'basic':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'arbitary_exclude']
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_uninterpretable']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.loc[:, 'psir_uninterpretable']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.contrast_class_id.isna()
            elif item == 'high_quality_cine':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_artifact']
            elif item == 'high_quality_t1native':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_artifact']
            elif item == 'high_quality_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_poor'] 
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_acceptable']
            elif item == 'acceptable_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_poor']
            elif item == 'cine_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_mismatch']
            elif item == 'cine_strictly_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_mismatch']
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_slightly_mismatch']
            elif item == 't1native_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_mismatch']
            elif item == 't1native_strictly_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_mismatch']
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_slightly_mismatch']
            elif item == 'without_apex_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'apex_position']
            elif item == 'without_base_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'base_position']
            elif item == 'possible_enhancement':
                condition[self.df_info_tags.index] &= self.df_info_tags.loc[:, 'possible_enhancement']
            else:
                raise ValueError(f"Unknown requirement {item}.")

        group_condition = np.stack([self.group2mask[key] for key in group_keys]).sum(axis=0) != 0
        condition &= group_condition
        indices = self.df_metadata.index[condition].tolist()
        patients_count = len(set(self.df_metadata.loc[indices, 'PatientName']))
        indices.sort()
        return indices, patients_count

    def partition_slices_to_volumes(self, indices: Sequence[int]) -> Dict[str, List[int]]:
        volume_mapping = {}
        indices_set = set(indices)
        for group, rows in self.df_metadata.loc[indices].groupby(
                ['InstitutionName', 'PatientName', 'StudyDate']):
            sample_identifier = f"I{rows.index[0]}_{group[0]}_{group[1]}_{group[2]}"
            sample_indices = list(set(rows.index.to_list()) & indices_set)
            if len(sample_indices) > 0:
                volume_mapping[sample_identifier] = sample_indices
        return volume_mapping

    def indices2classes(self, indices: np.ndarray):
        """
        :param indices: [N,] Integer
        """
        class_ids = self.df_info_blood_pool.loc[indices, 'contrast_class_id'].values  # [N,]
        return class_ids

    def __getitem__(self, dataset_index: int) -> dict:
        """
        Some indices are excluded in dataset, so dataset_index is not necessarily corresponding to the length of this dataset object, use the subset of this dataset (indices from get_valid_indices) object instead.

        Contents in the returned pair_dict:
            - cine_images: [T, H, W]
            - psir_image: [H, W]
            - t1native_images: [N_TI, H, W]
            - cine_dicoms_metadata: dict
            - psir_dicom_metadata: dict
            - psir_pixel_spacing: [row spacing, column spacing]
        """
        # Use loc instead of iloc, because dataset_index does not a continuous integers from 0 to len(dataset).
        meta_info = self.df_metadata.loc[dataset_index, :]
        pair_dict = self.dataset_index2slice[dataset_index]

        extra_info = {
            'dataset_index':     dataset_index,
            'sample_identifier': f"I{dataset_index}_"
                                 f"{meta_info['InstitutionName']}_{meta_info['PatientName']}_{meta_info['StudyDate']}_"
                                 f"Proj{meta_info['position']:.2f}"
        }

        if dataset_index in self.df_info_bbox.index:
            row = self.df_info_bbox.loc[dataset_index, :]
            bbox = [row.xmin, row.ymin, row.xmax, row.ymax]  # List[int*4]
            extra_info['bbox'] = bbox

        if dataset_index in self.df_info_blood_pool.index:
            extra_info['blood_pool_contrast_class'] = \
                self.df_info_blood_pool.loc[dataset_index, 'contrast_class_id'].astype(np.int64).item()

        pair_dict.update(extra_info)
        return pair_dict

    def __len__(self):
        return self.N_all


class VNEBundleSliceDataset_DCM(Data.Dataset):
    """
    pair_dict structure:
        cine_images: <class 'numpy.ndarray'> [T, H, W]
        psir_image: <class 'numpy.ndarray'> [H, W]
        t1native_images: <class 'numpy.ndarray'> [N_TI, H, W]
    """
    def __init__(self, *,
                 dataset_dirpath=None,
                 patient_partition_file: str = 'partition2patients.json',
                 protocols: Sequence[str] = ('psir', 'cine', 't1native'),  # protocols need to be loaded
                 psir_type: str = 'original'):
        dataset_index2slice = _VNESliceBundleHDF5Images(
            slice_dataset_path=os.path.join(dataset_dirpath, 'slice_dataset'),
            protocols=protocols,
            psir_type=psir_type
        )
        
        df_metadata = pd.read_csv(os.path.join(dataset_dirpath, 'infos/metadata.csv'), index_col='dataset_index')
        N_all = len(df_metadata)

        df_info_tags = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/manually_quality_control.csv'),
            index_col='dataset_index'
        ).astype(np.bool_)

        df_info_blood_pool = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/blood_pool_classification.csv'),
            index_col='dataset_index'
        )
        assert len(df_info_blood_pool) == len(df_metadata), \
            f"Blood Pool labels have missing, please check the correctness of files."

        # partition indices according to patient names in different groups
        with open(os.path.join(dataset_dirpath, f'infos/{patient_partition_file}')) as f:
            parition2patients = json.load(f)
        group2mask: Dict[int, Sequence[int]] = {}
        for group, patient_names in parition2patients.items():
            indices_mask = np.zeros_like(df_metadata.index, dtype=bool)
            for (institution_name, patient_name) in patient_names:
                indices_mask = indices_mask | \
                               ((df_metadata['InstitutionName'] == institution_name) &
                                (df_metadata['PatientName'] == patient_name))
            group2mask[group] = indices_mask

        df_info_bbox = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/heart_localization.csv'),
            index_col='dataset_index'
        )

        self.dataset_index2slice = dataset_index2slice
        self.df_metadata = df_metadata
        self.df_info_tags = df_info_tags
        self.df_info_bbox = df_info_bbox
        self.df_info_blood_pool = df_info_blood_pool
        # self.df_MI_sectors = df_MI_sectors
        self.group2mask = group2mask
        self.N_all = N_all

    def get_valid_indices(self, 
                          group_keys: Sequence[str],
                          mandatory_protocols: Sequence[str],
                          requirements: Sequence[str] = ()
                          ) -> List[int]:
        """
        Classify the dataset into slices level partition otherwise train/val dataset may have patients interleaved.

        There are two types of reasons to excluding samples in this dataset:
            1. There is no suitable bbox label.
            2. Match certain predefined reasons in exclusion matrix file.
        """
        # only_apex, base_position,
        condition = np.ones_like(self.df_metadata.index, dtype=np.bool_)

        for protocol in mandatory_protocols:
            key_name = f"{protocol}__SeriesNumber"
            condition &= ~np.isnan(self.df_metadata[key_name])

        for item in requirements:
            if item == 'basic':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'arbitary_exclude']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.loc[:, 'psir_uninterpretable']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.contrast_class_id.isna()
            elif item == 'high_quality_cine':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_artifact']
            elif item == 'high_quality_t1native':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_artifact']
            elif item == 'recognizable_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact']
            elif item == 'high_quality_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact_but_usable']
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact']
            elif item == 'cine_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_mismatch']
            elif item == 't1native_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_mismatch']
            elif item == 'without_apex_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'apex_position']
            elif item == 'without_base_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'base_position']
            elif item == 'possible_enhancement':
                condition[self.df_info_tags.index] &= self.df_info_tags.loc[:, 'possible_enhancement']
            else:
                raise ValueError(f"Unknown requirement {item}.")

        group_condition = np.stack([self.group2mask[key] for key in group_keys]).sum(axis=0) != 0
        condition &= group_condition
        indices = self.df_metadata.index[condition].tolist()
        indices.sort()
        patients_count = len(set(self.df_metadata.loc[indices, 'PatientName']))
        
        return indices, patients_count

    def partition_slices_to_volumes(self, indices: Sequence[int]) -> Dict[str, List[int]]:
        volume_mapping = {}
        indices_set = set(indices)
        for group, rows in self.df_metadata.loc[indices].groupby(
                ['InstitutionName', 'PatientName', 'StudyDate']):
            sample_identifier = f"I{rows.index[0]}_{group[0]}_{group[1]}_{group[2]}"
            sample_indices = list(set(rows.index.to_list()) & indices_set)
            if len(sample_indices) > 0:
                volume_mapping[sample_identifier] = sample_indices
        return volume_mapping

    def indices2classes(self, indices: np.ndarray):
        """
        :param indices: [N,] Integer
        """
        class_ids = self.df_info_blood_pool.loc[indices, 'contrast_class_id'].values  # [N,]
        return class_ids

    def __getitem__(self, dataset_index: int) -> dict:
        """
        Some indices are excluded in dataset, so dataset_index is not necessarily corresponding to the length of this dataset object, use the subset of this dataset (indices from get_valid_indices) object instead.

        Contents in the returned pair_dict:
            - cine_images: [T, H, W]
            - psir_image: [H, W]
            - t1native_images: [N_TI, H, W]
            - cine_dicoms_metadata: dict
            - psir_dicom_metadata: dict
            - psir_pixel_spacing: [row spacing, column spacing]
        """
        # Use loc instead of iloc, because dataset_index does not a continuous integers from 0 to len(dataset).
        meta_info = self.df_metadata.loc[dataset_index, :]
        pair_dict = self.dataset_index2slice[dataset_index]

        extra_info = {
            'dataset_index':     dataset_index,
            'sample_identifier': f"I{dataset_index}_"
                                 f"{meta_info['InstitutionName']}_{meta_info['PatientName']}_{meta_info['StudyDate']}_"
                                 f"Proj{meta_info['position']:.2f}"
        }
        # if dataset_index in self.df_MI_sectors.index:
        #     extra_info['sector_MI_label'] = np.array(
        #         self.df_MI_sectors.loc[dataset_index, 'deg-180':].astype(np.float32))  # [72,]

        if dataset_index in self.df_info_bbox.index:
            row = self.df_info_bbox.loc[dataset_index, :]
            bbox = [row.xmin, row.ymin, row.xmax, row.ymax]  # List[int*4]
            extra_info['bbox'] = bbox

        if dataset_index in self.df_info_blood_pool.index:
            extra_info['blood_pool_contrast_class'] = \
                self.df_info_blood_pool.loc[dataset_index, 'contrast_class_id'].astype(np.int64).item()

        pair_dict.update(extra_info)
        return pair_dict

    def __len__(self):
        return self.N_all


class VNEBundleSliceDataset_AMI(Data.Dataset):
    """
    pair_dict structure:
        cine_images: <class 'numpy.ndarray'> [T, H, W]
        psir_image: <class 'numpy.ndarray'> [H, W]
        t1native_images: <class 'numpy.ndarray'> [N_TI, H, W]
    """
    def __init__(self, *,
                 dataset_dirpath=DATASET_PATH['HCMVNE'],
                 patient_partition_file: str = 'partition2patients.json',
                 protocols: Sequence[str] = ('psir', 'cine', 't1native'),  # protocols need to be loaded
                 psir_type: str = 'original'):
        dataset_index2slice = _VNESliceBundleHDF5Images(
            slice_dataset_path=os.path.join(dataset_dirpath, 'slice_dataset'),
            protocols=protocols,
            psir_type=psir_type
        )
        
        df_metadata = pd.read_csv(os.path.join(dataset_dirpath, 'infos/metadata.csv'), index_col='dataset_index')
        N_all = len(df_metadata)

        df_info_tags = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/manually_quality_control.csv'),
            index_col='dataset_index'
        ).astype(np.bool_)

        df_info_blood_pool = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/blood_pool_classification.csv'),
            index_col='dataset_index'
        )
        assert len(df_info_blood_pool) == len(df_metadata), \
            f"Blood Pool labels have missing, please check the correctness of files."

        # df_MI_sectors = pd.read_csv(
        #     os.path.join(dataset_dirpath, 'infos/MI_sectors.csv'),
        #     index_col='dataset_index'
        # )
        # assert len(df_MI_sectors) == len(df_metadata), \
        #     f"MI sectors have missing, please check the correctness of files."

        # partition indices according to patient names in different groups
        with open(os.path.join(dataset_dirpath, f'infos/{patient_partition_file}')) as f:
            parition2patients = json.load(f)
        group2mask: Dict[int, Sequence[int]] = {}
        for group, patient_names in parition2patients.items():
            indices_mask = np.zeros_like(df_metadata.index, dtype=bool)
            for (institution_name, patient_name) in patient_names:
                indices_mask = indices_mask | \
                               ((df_metadata['InstitutionName'] == institution_name) &
                                (df_metadata['PatientName'] == patient_name))
            group2mask[group] = indices_mask

        df_info_bbox = pd.read_csv(
            os.path.join(dataset_dirpath, 'infos/heart_localization.csv'),
            index_col='dataset_index'
        )

        self.dataset_index2slice = dataset_index2slice
        self.df_metadata = df_metadata
        self.df_info_tags = df_info_tags
        self.df_info_bbox = df_info_bbox
        self.df_info_blood_pool = df_info_blood_pool
        # self.df_MI_sectors = df_MI_sectors
        self.group2mask = group2mask
        self.N_all = N_all

    def get_valid_indices(self, 
                          group_keys: Sequence[str],
                          mandatory_protocols: Sequence[str],
                          requirements: Sequence[str] = ()
                          ) -> List[int]:
        """
        Classify the dataset into slices level partition otherwise train/val dataset may have patients interleaved.

        There are two types of reasons to excluding samples in this dataset:
            1. There is no suitable bbox label.
            2. Match certain predefined reasons in exclusion matrix file.
        """
        # only_apex, base_position,
        condition = np.ones_like(self.df_metadata.index, dtype=np.bool_)

        for protocol in mandatory_protocols:
            key_name = f"{protocol}__SeriesNumber"
            condition &= ~np.isnan(self.df_metadata[key_name])

        for item in requirements:
            if item == 'basic':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'arbitary_exclude']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.loc[:, 'psir_uninterpretable']
                condition[self.df_info_blood_pool.index] &= ~self.df_info_blood_pool.contrast_class_id.isna()
            elif item == 'high_quality_cine':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_artifact']
            elif item == 'high_quality_t1native':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_artifact']
            elif item == 'recognizable_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact']
            elif item == 'high_quality_psir':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact_but_usable']
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'psir_artifact']
            elif item == 'cine_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'cine_mismatch']
            elif item == 't1native_aligned':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 't1native_mismatch']
            elif item == 'without_apex_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'apex_position']
            elif item == 'without_base_position':
                condition[self.df_info_tags.index] &= ~self.df_info_tags.loc[:, 'base_position']
            elif item == 'possible_enhancement':
                condition[self.df_info_tags.index] &= self.df_info_tags.loc[:, 'possible_enhancement']
            else:
                raise ValueError(f"Unknown requirement {item}.")

        group_condition = np.stack([self.group2mask[key] for key in group_keys]).sum(axis=0) != 0
        condition &= group_condition
        indices = self.df_metadata.index[condition].tolist()
        patients_count = len(set(self.df_metadata.loc[indices, 'PatientName']))
        indices.sort()
        return indices, patients_count
    
    def partition_slices_to_volumes(self, indices: Sequence[int]) -> Dict[str, List[int]]:
        volume_mapping = {}
        indices_set = set(indices)
        for group, rows in self.df_metadata.loc[indices].groupby(
                ['InstitutionName', 'PatientName', 'StudyDate']):
            sample_identifier = f"I{rows.index[0]}_{group[0]}_{group[1]}_{group[2]}"
            sample_indices = list(set(rows.index.to_list()) & indices_set)
            if len(sample_indices) > 0:
                volume_mapping[sample_identifier] = sample_indices
        return volume_mapping

    def indices2classes(self, indices: np.ndarray):
        """
        :param indices: [N,] Integer
        """
        class_ids = self.df_info_blood_pool.loc[indices, 'contrast_class_id'].values  # [N,]
        return class_ids

    def __getitem__(self, dataset_index: int) -> dict:
        """
        Some indices are excluded in dataset, so dataset_index is not necessarily corresponding to the length of this dataset object, use the subset of this dataset (indices from get_valid_indices) object instead.

        Contents in the returned pair_dict:
            - cine_images: [T, H, W]
            - psir_image: [H, W]
            - t1native_images: [N_TI, H, W]
            - cine_dicoms_metadata: dict
            - psir_dicom_metadata: dict
            - psir_pixel_spacing: [row spacing, column spacing]
        """
        # Use loc instead of iloc, because dataset_index does not a continuous integers from 0 to len(dataset).
        meta_info = self.df_metadata.loc[dataset_index, :]
        pair_dict = self.dataset_index2slice[dataset_index]

        extra_info = {
            'dataset_index':     dataset_index,
            'sample_identifier': f"I{dataset_index}_"
                                 f"{meta_info['InstitutionName']}_{meta_info['PatientName']}_{meta_info['StudyDate']}_"
                                 f"Proj{meta_info['position']:.2f}"
        }

        if dataset_index in self.df_info_bbox.index:
            row = self.df_info_bbox.loc[dataset_index, :]
            bbox = [row.xmin, row.ymin, row.xmax, row.ymax]  # List[int*4]
            extra_info['bbox'] = bbox

        if dataset_index in self.df_info_blood_pool.index:
            extra_info['blood_pool_contrast_class'] = \
                self.df_info_blood_pool.loc[dataset_index, 'contrast_class_id'].astype(np.int64).item()

        pair_dict.update(extra_info)
        return pair_dict

    def __len__(self):
        return self.N_all

class Triplets4VNE(Data.Dataset):
    """
    Dataset of short_axis_MRI_slice(cine, T1 native, LGE) of patients, 
    which is organized for VNE-like project.
    """
    # SECTOR_KEYPOINTS, SECTOR_ANGLES = make_sector_keypoints(angle_step=5)
    def __init__(self,
                 index2slice_dataset: Data.Dataset,
                 *,
                #  num_fixed_cine_frames: int = None,
                 augmentation_mode: str = None,
                 is_roi_resample=True,
                 roi_resample_size=(128, 128),  # [width (x), height (y)]
                 roi_fixed_pixel_spacing: Tuple[float, float] = (0.89, 0.89),  # in x (column) and y (row) order
                 modality_vec_dim: int = 8,
                 without_rightventricular: bool = False
                 ):
        self.index2slice_dataset = index2slice_dataset
        self.num_slices = len(index2slice_dataset)  # define the length of the dataset

        # self.num_fixed_cine_frames = num_fixed_cine_frames  # cine frames to resample 
        self.is_roi_resample = is_roi_resample  # resample the ROI to fixed resolution to let the cine and psir matched
        self.roi_resample_size = roi_resample_size  # the size of the resampled ROI
        self.roi_fixed_pixel_spacing = roi_fixed_pixel_spacing  # the fixed pixel spacing of the resampled ROI, which should be the same as the main protocol(psir)
        # self.max_t1native_frames = max_t1native_frames  # t1native frames to pad
        self.modality_vec_dim = modality_vec_dim
        self.without_rightventricular = without_rightventricular
        # Data transform
        # self.T_structural = StructuralAugmentation(mode=mode)
        # self.T_in_roi = InROIAugmentation(mode=mode)
        self.T = AffineAugmentation(mode=augmentation_mode)

    def crop_and_resize_to_roi(self, images: np.ndarray, bbox):
        """
        Resample input images to ROI using a bounding box.

        Params:
            images: [N, H, W]
            bbox: [4 (xmin, ymin, xmax, ymax), ] in pixels
        Returns:
            [N, Hroi, Wroi]
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
        # data extraction
        pair_dict: dict = self.index2slice_dataset[index]
        sample_identifier = pair_dict['sample_identifier']
        
        if 'bbox' in pair_dict:
            bbox = pair_dict['bbox'].copy()
            # normalize bbox
            bbox = np.clip(bbox, 0, 1).tolist()
            roi_enabled = True
        else:
            bbox = [0.0, 0.0, 1.0, 1.0]
            roi_enabled = False

        if 'blood_pool_contrast_class' in pair_dict:
            blood_pool_contrast_class = pair_dict['blood_pool_contrast_class']
        else:
            blood_pool_contrast_class = -1

        # temporal image list for vectorized cropping & augmentation
        # stack semantics: first in, last out
        t_images = []
        t_masks = []

        psir_image = pair_dict['psir'].astype(np.float32, copy=True)  # [H, W]
        H, W = psir_image.shape
        psir_pixel_spacing = pair_dict['psir_pixel_spacing'].astype(np.float32, copy=True)
        
        psir_image = normalize_array(psir_image, mode='minmax_zero2one')
        t_images.extend([psir_image])
        
        if 'cine' in pair_dict:
            cine_image = pair_dict['cine'].astype(np.float32, copy=True)  # [H, W]
            
            assert (H, W) == tuple(cine_image.shape[-2:])
            # cine_closest_index = pair_dict['cine_idx']
            # T_raw =  cine_images.shape[0]

            # cine resample to fixed frames
            # if self.num_fixed_cine_frames is not None:
            #     # 3d interpolation will cost ~55ms, while 1d interpolation only cost ~0.2ms
            #     cine_images = rearrange(cine_images, 'T_raw H W -> 1 (H W) T_raw')
            #     cine_images = nnf.interpolate(torch.from_numpy(cine_images), 
            #                                   self.num_fixed_cine_frames,
            #                                   mode='linear', 
            #                                   align_corners=True).numpy()
            #     cine_images = rearrange(cine_images, '1 (H W) T -> T H W', H=H, W=W)  # [T', H, W]

            # normalization to [0, 1] for augmentation pipeline compatibility
            cine_image = normalize_array(cine_image, mode='minmax_zero2one')
            H, W = cine_image.shape
            t_images.extend([cine_image])
            
            cine_mask = pair_dict['cine_mask_192']
            h_mask, w_mask = cine_mask.shape[-2:]
            t_masks.extend([cine_mask])

        t_images = np.stack(t_images)  # [N, H, W]
        H, W = t_images.shape[-2:]
        
        # t_masks = t_masks + [np.ones([h_mask, w_mask])]
        t_masks = np.stack(t_masks)
        h_mask, w_mask = t_masks.shape[-2:] 
        # resample and crop the images and the masks
        if self.is_roi_resample and roi_enabled and bbox is not None:
            # compute roi bbox
            xmin, ymin, xmax, ymax = bbox
            if self.roi_fixed_pixel_spacing is not None:
                # dicom store Pixel Spacing in [row spacing, col spacing] order
                spacing_x_src, spacing_y_src = psir_pixel_spacing[1], psir_pixel_spacing[0]
                spacing_x_dst, spacing_y_dst = self.roi_fixed_pixel_spacing
                width = self.roi_resample_size[0] * (spacing_x_dst / spacing_x_src)
                height = self.roi_resample_size[1] * (spacing_y_dst / spacing_y_src)
                
                width_mask = self.roi_resample_size[0] * (spacing_x_dst / 0.89)
                height_mask = self.roi_resample_size[1] * (spacing_y_dst / 0.89)
            else:
                raise RuntimeError(f"`roi_fixed_pixel_spacing` must be specified for ROI computing.")
            
            # images processing
            center_x, center_y = (xmax + xmin) * W / 2, (ymax + ymin) * H / 2
            xmin = int(center_x - 0.5 * width)
            ymin = int(center_y - 0.5 * height)
            xmax = int(center_x + 0.5 * width)
            ymax = int(center_y + 0.5 * height)
            bbox_pixel = [xmin, ymin, xmax, ymax]

            # resample to bounding box
            t_images = self.crop_and_resize_to_roi(t_images, bbox_pixel)
            
            # mask processing                    
            center_x_mask, center_y_mask = (1 + 0) * w_mask / 2, (1 + 0) * h_mask / 2
            xmin_mask = int(center_x_mask - 0.5 * width_mask)
            ymin_mask = int(center_y_mask - 0.5 * height_mask)
            xmax_mask = int(center_x_mask + 0.5 * width_mask)
            ymax_mask = int(center_y_mask + 0.5 * height_mask)
            bbox_pixel_mask = [xmin_mask, ymin_mask, xmax_mask, ymax_mask]
            t_masks = self.crop_and_resize_to_roi(t_masks, bbox_pixel_mask)            
             
        # combined augmentation
        t_images, t_masks, t_bboxes= self.T(t_images, mask=t_masks, bboxes=[bbox])
        bbox = t_bboxes[0]

        # extract augmentation and ROI cropping results
        # if 'cine' in pair_dict:
        #     cine_images = t_images[1:(T+1), :, :]
        #     # normalize because intensity range is changed after ROI cropping
        #     cine_images = normalize_array(cine_images, mode='minmax_zero2one')
        #     # extract the closest image of cine to psir
        #     if self.num_fixed_cine_frames is not None:
        #         relative_position = cine_closest_index / (T_raw - 1) if T_raw > 1 else 0
        #         cine_closest_index = int(relative_position * (self.num_fixed_cine_frames -1) + 0.5) if self.num_fixed_cine_frames > 1 else 0
        #     closest_cine_image = cine_images[cine_closest_index, :, :]
        cine_image = t_images[1]
        cine_image = normalize_array(cine_image, mode='minmax_zero2one')
        psir_image = t_images[0]
        psir_image = normalize_array(psir_image, mode='minmax_zero2one')
        mask = t_masks[0]
        # right ventricle
        mask_rv = mask.copy()
        mask_rv[mask != 1] = 0
        mask_rv[mask == 1] = 1

        # myocardium
        mask_myo = mask.copy()
        mask_myo[mask != 2] = 0
        mask_myo[mask == 2] = 1

        # left ventricle
        mask_lv = mask.copy()
        mask_lv[mask != 3] = 0
        mask_lv[mask == 3] = 1
        
        # mask_concat = np.stack([mask_lv, mask_myo, mask_rv], axis=0)
        if self.without_rightventricular:
            mask_concat = np.stack([mask_lv, mask_myo], axis=0)
        else:
            mask_concat = np.stack([mask_lv, mask_myo, mask_rv], axis=0)
        
        cine_z_input = norm.sample(self.modality_vec_dim)
        psir_z_input = norm.sample(self.modality_vec_dim)
        
        sample = {
            'sample_identifier':         sample_identifier,  # str
            'psir':                      psir_image[None, :, :].astype(np.float32, copy=False),  # [C=1, H, W]
            'psir_z_input': psir_z_input.astype(np.float32, copy=False),  # [modality_vec_dim,]
            # 'image_location_mask':       image_location_mask.astype(bool, copy=False),  # [H, W]
            # 'blood_pool_contrast_class': int(blood_pool_contrast_class),  # Integer
        }

        if 'cine' in pair_dict:
            sample.update({
                # 'cine_images':        cine_images[None, ...].astype(np.float32, copy=False),  # [C=1, T, H, W]
                # 'cine_closest_index': int(cine_closest_index),  # int
                'cine': cine_image[None, :, :].astype(np.float32, copy=False),  # [C=1, H, W]
                'cine_mask': mask_concat.astype(np.float32, copy=False),  # [C=3, H, W]
                'cine_z_input': cine_z_input.astype(np.float32, copy=False),  # [modality_vec_dim,]
            })

        return sample


# =============================================================================
# StarGANv2 related
# StarGANv2 need extra reference images (same_class_im1, same_class_im2, class_label) for training
# =============================================================================
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def build_random_reference_generator(indices, classes):
    """
    :param indices: [N,] Integer
    :param classes: [N,] Integer
    """
    assert len(indices) == len(classes)
    N = len(indices)

    def generator():
        nonlocal indices, classes, N
        classes = np.array(classes).copy()
        image_indices = np.array(indices).copy()
        image2_indices = np.array(indices).copy()
        while True:
            perm = np.random.permutation(np.arange(N))
            intraclass_perm = np.arange(N)  # only permute intra-class
            for class_id in np.unique(classes):
                class_mask = (classes == class_id)
                intraclass_perm[class_mask] = np.random.permutation(intraclass_perm[class_mask])

            classes = classes[perm]
            image_indices = image_indices[perm]
            image2_indices = image2_indices[intraclass_perm]

            for i in range(N):
                yield (int(image_indices[i]), int(image2_indices[i]), int(classes[i]))

    it = iter(generator())
    it = iter(LockedIterator(it))
    return it
