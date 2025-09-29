import torch
import torch.utils.data
from torch.utils.data import DataLoader
from typing import Dict, Optional
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from .acdc_dataset import *

class ACDCSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_dirpath,
                 batch_size,
                 num_workers,
                 without_rightventricular,
                 is_roi_resample=True,
                 roi_fixed_pixel_spacing=(0.89, 0.89),
                 roi_resample_size=(128, 128),
                 modality_vec_dim=8,
                 training_groups=None,
                 testing_groups=None,
                 train=None,
                 validation=None,
                 test=None,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # data preprocess
        self.dataset_dirpath = dataset_dirpath
        self.without_rightventricular = without_rightventricular
        self.roi_fixed_pixel_spacing = roi_fixed_pixel_spacing
        self.roi_resample_size = roi_resample_size
        self.is_roi_resample = is_roi_resample
        self.training_groups = training_groups
        self.testing_groups = testing_groups
        self.dataset_configs = dict()
        self.modality_vec_dim = modality_vec_dim
        
        if train is not None:
            self.dataset_configs['train'] = train
        
        if validation is not None:
            self.dataset_configs['validation'] = validation
        
        if test is not None:
            self.dataset_configs['test'] = test
            
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up data splits for training, validation, and testing.
        Called once per process in distributed training.
        """
        self.raw_pairs_dataset = CINEBundleSliceDataset(
            dataset_dirpath=self.dataset_dirpath,
            is_bbox_labeled=True,
            partition_filepath='partition2patients.json'
        )
        self.prefetch_to_gpu = True
        
        training_groups = self.training_groups
        testing_groups = self.testing_groups
        
        
        self.train_indices = self.raw_pairs_dataset.get_valid_indices(
            training_groups,
        )
        
        total_validation_indices = self.raw_pairs_dataset.get_valid_indices(
            testing_groups,
        )
        
        _, self.val_indices = train_test_split(
            total_validation_indices, test_size=0.2, random_state=42
        )
        self.test_indices = total_validation_indices

        # Initialize datasets based on stage
        if stage == 'fit' or stage is None:
            if 'train' in self.dataset_configs:
                self.train_dataset = self._build_dataset(
                    indices=self.train_indices, 
                    augmentation_mode='hvflip_affine'
                )
                
            if 'validation' in self.dataset_configs:
                self.val_dataset = self._build_dataset(
                    indices=self.val_indices, 
                    augmentation_mode=None
                )
        
        if stage == 'test' or stage is None:
            if 'test' in self.dataset_configs:
                self.test_dataset = self._build_dataset(
                    indices=self.test_indices,
                    augmentation_mode=None
                )
    
    def _build_dataset(self, indices, augmentation_mode=None):
        subset = torch.utils.data.Subset(self.raw_pairs_dataset, indices)
        dataset = CINESlices4Segmentation(subset,
                                  is_augmentation = augmentation_mode,
                                  is_roi_resample = self.is_roi_resample,
                                  roi_resample_size = self.roi_resample_size,
                                  roi_fixed_pixel_spacing = self.roi_fixed_pixel_spacing,
                                  modality_vec_dim = self.modality_vec_dim,
                                  without_rightventricular=self.without_rightventricular
                                  )
        return dataset

    def _build_loader(self, dataset, is_train=False, **kwargs4loader):
        """
        Build data loader with appropriate parameters
        """
        if dataset is None:
            return None
            
        kwargs = dict(batch_size=self.batch_size, 
                      shuffle=is_train,
                      num_workers=self.num_workers,
                      pin_memory=True,
                      persistent_workers=False)
        kwargs.update(kwargs4loader)
        loader = torch.utils.data.DataLoader(
            dataset,
            **kwargs
        )
        return loader
    
    def train_dataloader(self):
        """Return the train dataloader"""
        if self.train_dataset is None:
            return None
            
        train_params = self.dataset_configs.get('train', {}).get('params', {})
        return self._build_loader(self.train_dataset, is_train=True, **train_params)
    
    def val_dataloader(self):
        """Return the validation dataloader"""
        if self.val_dataset is None:
            return None
            
        val_params = self.dataset_configs.get('validation', {}).get('params', {})
        return self._build_loader(self.val_dataset, is_train=False, **val_params)
    
    def test_dataloader(self):
        """Return the test dataloader"""
        if self.test_dataset is None:
            return None
            
        test_params = self.dataset_configs.get('test', {}).get('params', {})
        return self._build_loader(self.test_dataset, is_train=False, **test_params)