import torch
import torch.utils.data
from torch.utils.data import DataLoader
from typing import Dict, Optional
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from .inhouse_dataset import *

class HCMDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_type,
                 dataset_dirpath,
                 batch_size,
                 num_workers,
                 modality_vec_dim,
                 without_rightventricular,
                 roi_fixed_pixel_spacing,
                 roi_resample_size,
                 mandatory_protocols,
                 train_criteria,
                 test_criteria, 
                 training_groups,
                 testing_groups,
                 train=None,
                 validation=None,
                 test=None,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # data preprocess
        self.dataset_type = dataset_type
        self.dataset_dirpath = dataset_dirpath
        self.roi_fixed_pixel_spacing = roi_fixed_pixel_spacing
        self.roi_resample_size = roi_resample_size
        # self.num_fixed_cine_frames = num_fixed_cine_frames
        self.modality_vec_dim = modality_vec_dim
        self.without_rightventricular = without_rightventricular
        self.mandatory_protocols = mandatory_protocols
        self.training_groups = training_groups
        self.testing_groups = testing_groups
        self.dataset_configs = dict()
        
        self.train_criteria = train_criteria
        self.test_criteria = test_criteria
        
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
        if self.dataset_type.upper() == 'HCM':
            self.raw_pairs_dataset = VNEBundleSliceDataset_HCM(
                psir_type='registered_to_cine_closest_tt',
                dataset_dirpath=self.dataset_dirpath,
                protocols=self.mandatory_protocols,
            )
        elif self.dataset_type.upper() == 'DCM':
            self.raw_pairs_dataset = VNEBundleSliceDataset_DCM(
                psir_type='registered_to_cine_closest_tt',
                dataset_dirpath=self.dataset_dirpath,
                protocols=self.mandatory_protocols,
            )
        elif self.dataset_type.upper() == 'AMI':
            self.raw_pairs_dataset = VNEBundleSliceDataset_AMI(
                psir_type='registered_to_cine_closest_tt',
                dataset_dirpath=self.dataset_dirpath,
                protocols=self.mandatory_protocols,
            )
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")
        
        self.prefetch_to_gpu = True
        
        train_criteria = self.train_criteria
        test_criteria = self.test_criteria
        
        training_groups = self.training_groups
        testing_groups = self.testing_groups
        
        self.train_indices, _ = self.raw_pairs_dataset.get_valid_indices(
            training_groups,
            self.mandatory_protocols,
            self.train_criteria
        )
        
        total_validation_indices, _= self.raw_pairs_dataset.get_valid_indices(
            testing_groups,
            self.mandatory_protocols,
            self.test_criteria
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
        dataset = Triplets4VNE(subset,
                               augmentation_mode = augmentation_mode,
                               roi_resample_size = self.roi_resample_size,
                               roi_fixed_pixel_spacing = self.roi_fixed_pixel_spacing,
                               modality_vec_dim = self.modality_vec_dim,
                               without_rightventricular = self.without_rightventricular
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