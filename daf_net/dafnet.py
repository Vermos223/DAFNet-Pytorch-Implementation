import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from abc import abstractmethod

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models import create_discriminator
from functions.loss import (
    DiceLoss, CombinedDiceBCELoss, KLDivergenceLoss, 
    MAELoss, IdentityLoss, 
)
from functions.utils_code import instantiate_from_config


def disabled_train(self, mode=True):
    """
    Overwrite mode.train with this function to make sure train/eval mode
    does not change anymore.
    """
    return self


class BaseMultimodal(pl.LightningModule):
    def __init__(self, base_learning_rate, loss_config, scheduler_config=None):
        super(BaseMultimodal, self).__init__()
        self.save_hyperparameters()
        self._initialize_losses(**loss_config)
        
        self.automatic_optimization = False
        self.scheduler_config = scheduler_config
        self.base_learning_rate = float(base_learning_rate)
           
    def _initialize_losses(self, w_sup_M, w_adv_M, w_rec_X, w_adv_X, w_kl, w_rec_Z, restrict_channels=4):    
        self.register_buffer('w_sup_M', torch.tensor(w_sup_M))
        self.register_buffer('w_adv_M', torch.tensor(w_adv_M))
        self.register_buffer('w_rec_X', torch.tensor(w_rec_X))
        self.register_buffer('w_adv_X', torch.tensor(w_adv_X))
        self.register_buffer('w_kl', torch.tensor(w_kl))
        self.register_buffer('w_rec_Z', torch.tensor(w_rec_Z))
        
        self.dice_loss = DiceLoss(restrict_channels)
        self.combined_dice_bce = CombinedDiceBCELoss(num_classes=restrict_channels)
        self.kl_loss = KLDivergenceLoss()
        self.mae_loss = MAELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.identity_loss = IdentityLoss()
     
    @abstractmethod
    def forward(self, x):
        """Forward pass - to be implemented by subclasses"""
        pass

    def configure_optimizers(self):
        """Configure optimizers for generators and discriminators"""
        # Generator optimizer
        lr = float(self.base_learning_rate)
        g_params = list(self.anatomy_encoders.parameters()) + \
            list(self.modality_encoder.parameters()) + \
            list(self.anatomy_fuser.parameters()) + \
            list(self.segmentor.parameters()) + \
            list(self.decoder.parameters()) + \
            (list(self.balancer.parameters()) if hasattr(self, 'balancer') else [])
        g_optimizer = torch.optim.AdamW(g_params, lr=lr)
        
        # Discriminator optimizers
        d_optimizers = []
        betas = (0.5, 0.999)
        d_mask_optimizer = torch.optim.AdamW(self.d_mask.parameters(), lr=lr, betas=betas)
        d_optimizers.append(d_mask_optimizer)
        
        d_image1_optimizer = torch.optim.AdamW(self.d_image1.parameters(), lr=lr, betas=betas)
        d_optimizers.append(d_image1_optimizer)
        
        if hasattr(self, 'd_image2'):
            d_image2_optimizer = torch.optim.AdamW(self.d_image2.parameters(), lr=lr, betas=betas)
            d_optimizers.append(d_image2_optimizer)
        
        optimizers = [g_optimizer] + d_optimizers
        
        if self.scheduler_config is not None:
            assert 'target' in self.scheduler_config
            print("Setting up LambdaLR scheduler...")
            
            scheduler_class = instantiate_from_config(self.scheduler_config)
            
            self.schedulers_list = []
            for opt in optimizers:
                lr_scheduler = LambdaLR(opt, lr_lambda=scheduler_class.schedule)
                self.schedulers_list.append(lr_scheduler)
        
        return optimizers
    
    def training_step(self, batch, batch_idx):
        """Training step with adversarial training"""
 
        optimizers = self.optimizers()
        
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        g_opt = optimizers[0]  # Generator optimizer
        d_opts = optimizers[1:] if len(optimizers) > 1 else []  # Discriminator optimizers
        
        # prepare the data
        x1 = batch.get(self.modalities[0], None)
        x2 = batch.get(self.modalities[1], None) if len(self.modalities) > 1 else None
        mask1 = batch.get(self.modalities[0] + '_mask')
        mask2 = batch.get(self.modalities[1] + '_mask') if len(self.modalities) > 1 else None
        
        z1_input = batch.get(f'{self.modalities[0]}_z_input', None)
        z2_input = batch.get(f'{self.modalities[1]}_z_input', None) if len(self.modalities) > 1 else None
        
        # ===============================
        # Train Discriminators
        # ===============================
        for d_opt in d_opts:
            d_opt.zero_grad()
        
        d_loss = self._discriminator_step(x1, mask1, x2, mask2, z1_input, z2_input)
        self.manual_backward(d_loss)
        
        for d_opt in d_opts:
            d_opt.step()
        
        # ===============================
        # Train Generator
        # ===============================
        g_opt.zero_grad()
        g_loss, g_losses = self._generator_step(x1, mask1, x2, mask2, z1_input, z2_input)
        self.manual_backward(g_loss)
        g_opt.step()
        
        # ===============================
        # Step Schedulers (Manual Mode)
        # ===============================
        if hasattr(self, 'schedulers_list') and self.schedulers_list:
            for scheduler in self.schedulers_list:
                scheduler.step()
        
        # Log losses
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True)
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True)
        
        for loss_name, loss_value in g_losses.items():
            self.log(f'train/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return g_loss
    
    @abstractmethod
    def _discriminator_step(self, x1, mask1=None, x2=None, mask2=None, z1_input=None, z2_input=None):
        """Discriminator training step - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _generator_step(self, x1, mask1, x2=None, mask2=None, z1_input=None, z2_input=None):
        """Generator training step - to be implemented by subclasses"""
        pass
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):    
        # prepare the data
        x1 = batch.get(self.modalities[0], None)
        x2 = batch.get(self.modalities[1], None) if len(self.modalities) > 1 else None
        mask1 = batch.get(self.modalities[0] + '_mask')
        mask2 = batch.get(self.modalities[1] + '_mask') if len(self.modalities) > 1 else None
        
        z1_input = batch.get('z1_input', None)
        z2_input = batch.get('z2_input', None)
        
        d_loss = self._discriminator_step(x1, mask1, x2, mask2, z1_input, z2_input)
        g_loss, g_losses = self._generator_step(x1, mask1, x2, mask2, z1_input, z2_input)

        # Log losses
        self.log('val/d_loss', d_loss, on_step=True, on_epoch=True)
        self.log('val/g_loss', g_loss, on_step=True, on_epoch=True)
        
        for loss_name, loss_value in g_losses.items():
            self.log(f'val/{loss_name}', loss_value, on_step=True, on_epoch=True)
        
        return g_loss 

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x1 = batch.get(self.modalities[0], None)
        x2 = batch.get(self.modalities[1], None) if len(self.modalities) > 1 else None    
        z1_input = batch.get('z1_input', None)
        z2_input = batch.get('z2_input', None) if len(self.modalities) > 1 else None
        
        return self(x1, x2, z1_input, z2_input)
    
    @torch.no_grad()
    def log_images(self, batch, split="train", **kwargs):
        x1 = batch.get(self.modalities[0], None)
        x2 = batch.get(self.modalities[1], None) if len(self.modalities) > 1 else None
        mask1 = batch.get(self.modalities[0] + '_mask')
        mask2 = batch.get(self.modalities[1] + '_mask') if len(self.modalities) > 1 else None
        
        z1_input = batch.get(f'{self.modalities[0]}_z_input', None)
        z2_input = batch.get(f'{self.modalities[1]}_z_input', None) if len(self.modalities) > 1 else None
        
        results = self(x1, x2, z1_input, z2_input)
        
        log_dict = {}
        
        if x1 is not None:
            log_dict[f'input_{self.modalities[0]}'] = x1
        if x2 is not None:
            log_dict[f'input_{self.modalities[1]}'] = x2
            
        mask1_original = batch.get(self.modalities[0] + '_mask_original', None)
        mask2_original = batch.get(self.modalities[1] + '_mask_original', None) if len(self.modalities) > 1 else None
        
        if mask1_original is not None:
            if mask1_original.dim() == 3:  # [B, H, W]
                mask1_original = mask1_original.unsqueeze(1)  # [B, 1, H, W]
            log_dict[f'gt_mask_{self.modalities[0]}'] = mask_converting_show(mask1_original)
        elif mask1 is not None:
            mask1_rgb = mask_converting_show(mask1)
            log_dict[f'gt_mask_{self.modalities[0]}'] = mask1_rgb
                
        if mask2_original is not None:
            # mask2_original 是 [B, H, W]，需要添加通道维度变成 [B, 1, H, W]
            if mask2_original.dim() == 3:  # [B, H, W]
                mask2_original = mask2_original.unsqueeze(1)  # [B, 1, H, W]
            log_dict[f'gt_mask_{self.modalities[1]}'] = mask_converting_show(mask2_original)
        elif mask2 is not None:
            mask2_rgb = mask_converting_show(mask2)
            log_dict[f'gt_mask_{self.modalities[1]}'] = mask2_rgb
        
        if 'segmentation_1' in results and results['segmentation_1'] is not None:
            #  seg1_rgb = mask_converting_show(results['segmentation_1'])
            seg1 = results['segmentation_1']
            if seg1.shape[1] == 4:
                seg1_class = torch.argmax(seg1, dim=1, keepdim=True).float()  # [B, 1, H, W]
            else:
                seg1_class = seg1
            seg1_rgb = mask_converting_show(seg1_class)
            log_dict[f'pred_mask_{self.modalities[0]}'] = seg1_rgb
                
        if 'segmentation_2' in results and results['segmentation_2'] is not None:
            #  seg2_rgb = mask_converting_show(results['segmentation_2'])
            seg2 = results['segmentation_2']
            if seg2.shape[1] == 4:
                seg2_class = torch.argmax(seg2, dim=1, keepdim=True).float()  # [B, 1, H, W]
            else:
                seg2_class = seg2
            seg2_rgb = mask_converting_show(seg2_class)
            log_dict[f'pred_mask_{self.modalities[1]}'] = seg2_rgb
        
        # Reconstruction
        if 'reconstruction_1' in results and results['reconstruction_1'] is not None:
            log_dict[f'recon_{self.modalities[0]}'] = results['reconstruction_1']
        if 'reconstruction_2' in results and results['reconstruction_2'] is not None:
            log_dict[f'recon_{self.modalities[1]}'] = results['reconstruction_2']
            
        # Cross-domain conversion image (if available)
        if 'crossdomain_image_2' in results and results['crossdomain_image_2'] is not None:
            log_dict[f'cross_{self.modalities[0]}_to_{self.modalities[1]}'] = results['crossdomain_image_2']
        if 'crossdomain_image_1' in results and results['crossdomain_image_1'] is not None:
            log_dict[f'cross_{self.modalities[1]}_to_{self.modalities[0]}'] = results['crossdomain_image_1']
        
        return log_dict
    
    def _make_trainable(self, model, trainable=True):
        """Set model parameters as trainable or not"""
        for param in model.parameters():
            param.requires_grad = trainable
    
    def _sample_batch(self, tensor, target_size):
        if tensor is None or tensor.numel() == 0:
            raise ValueError("Input tensor is None or empty")
        
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
            
        total_samples = tensor.shape[0]
        
        if total_samples == target_size:
            return tensor
        elif total_samples < target_size:
            # If we have fewer samples than needed, repeat some samples
            indices = torch.randint(0, total_samples, (target_size,), device=tensor.device)
            return tensor[indices]
        else:
            # If we have more samples than needed, randomly sample
            indices = torch.randperm(total_samples, device=tensor.device)[:target_size]
            return tensor[indices]


class DAFNetLightning(BaseMultimodal):
    def __init__(self,
                 modalities,
                 num_masks,
                 loss_config,
                 anatomy_encoder_config,
                 modality_encoder_config,
                 segmentor_params_config,
                 decoder_params_config,
                 anatomy_fuser_config,
                 d_mask_params,
                 d_image_params,
                 automated_pairing=False,
                 balancer_config=None,
                 learning_rate=1e-4,
                 scheduler_config=None,
                 **kwargs):
        super(DAFNetLightning, self).__init__(learning_rate, loss_config, scheduler_config)
        
        self.modalities = modalities
        self.num_masks = num_masks
        self.automated_pairing = automated_pairing
        self.loss_config = loss_config
        
        self.d_mask_params = d_mask_params
        self.d_image_params = d_image_params

        self.anatomy_encoders = instantiate_from_config(anatomy_encoder_config)
        self.modality_encoder = instantiate_from_config(modality_encoder_config)
        self.segmentor = instantiate_from_config(segmentor_params_config)
        self.decoder = instantiate_from_config(decoder_params_config)
        self.anatomy_fuser = instantiate_from_config(anatomy_fuser_config)
        
        if self.automated_pairing:
            assert balancer_config is not None
            self.balancer = instantiate_from_config(balancer_config)
        
        # Discriminators
        self._build_discriminators()
        
        # Z Regressor for reconstructing sampled Z
        self._build_z_regressor()
    
    def _build_discriminators(self):
        # Mask discriminator
        self.d_mask = create_discriminator(**self.d_mask_params)
        
        # Image discriminators
        if len(self.modalities) >= 1:
            self.d_image1 = create_discriminator(**self.d_image_params)

        if len(self.modalities) >= 2:
            self.d_image2 = create_discriminator(**self.d_image_params)
    
    def _build_z_regressor(self):
        self.z_regressor_available = True
    
    def forward(self, x1, x2=None, z1_input=None, z2_input=None):
        
        if x1 is None:
            raise ValueError("x1 cannot be None")
        
        results = {}
        
        # Encode anatomy for each modality
        anatomy_inputs = {f'input_{self.modalities[0]}': x1}
        if x2 is not None:
            anatomy_inputs[f'input_{self.modalities[1]}'] = x2
        
        anatomy_outputs = self.anatomy_encoders(anatomy_inputs)
        s1 = anatomy_outputs[f'anatomy_{self.modalities[0]}']
        s2 = anatomy_outputs.get(f'anatomy_{self.modalities[1]}', None) if x2 is not None else None
        
        # Encode modality information
        z1, kl1 = self.modality_encoder(s1, x1)
        if x2 is not None:
            z2, kl2 = self.modality_encoder(s2, x2)
        else:
            z2, kl2 = None, None
        
        # Segment
        m1 = self.segmentor(s1)
        m2 = self.segmentor(s2) if x2 is not None else None
        
        # Decode (reconstruct images)
        y1 = self.decoder(s1, z1)  # the reconstructed s1 and z1 should be in consistency with the input x1
        y2 = self.decoder(s2, z2) if x2 is not None else None
        
        results.update({
            'anatomy_1': s1,
            'anatomy_2': s2,
            'modality_1': z1,
            'modality_2': z2,
            'segmentation_1': m1,
            'segmentation_2': m2,
            'reconstruction_1': y1,
            'reconstruction_2': y2,
            'kl_1': kl1,
            'kl_2': kl2
        })
        
        # Cross-domain translation if both modalities available
        if s2 is not None and z2 is not None:
            # Deform and fuse anatomies
            s1_def, s1_fused = self.anatomy_fuser(s1, s2)
            s2_def, s2_fused = self.anatomy_fuser(s2, s1)
            
            # Cross-segmentation
            m1_s2_def = self.segmentor(s2_def)
            m2_s1_def = self.segmentor(s1_def)
            
            # Cross-reconstruction
            y1_s2_def = self.decoder(s2_def, z1)
            y2_s1_def = self.decoder(s1_def, z2)
            
            # CI: Cross-domain Image
            CI_2_s1_def_z1 = self.decoder(s1_def, z1)
            CI_1_s2_def_z2 = self.decoder(s2_def, z2)
            
            results.update({
                'anatomy_1_def': s1_def,
                'anatomy_2_def': s2_def,
                'anatomy_1_fused': s1_fused,
                'anatomy_2_fused': s2_fused,
                'segmentation_1_s2_def': m1_s2_def,
                'segmentation_2_s1_def': m2_s1_def,
                'reconstruction_1_s2_def': y1_s2_def,
                'reconstruction_2_s1_def': y2_s1_def,
                'crossdomain_image_2_s1_def': CI_2_s1_def_z1,
                'crossdomain_image_1_s2_def': CI_1_s2_def_z2
            })
        
        # Z Reconstruction
        if z1_input is not None or z2_input is not None:
            if hasattr(self, 'z_regressor_available') and self.z_regressor_available:
                z_reconstructions = {}
                
                if z1_input is not None and s1 is not None:
                    y1_z_rec = self.decoder(s1, z1_input)
                    z1_rec_mean, _ = self.modality_encoder(s1, y1_z_rec)
                    z_reconstructions['z1_reconstructed'] = z1_rec_mean
                
                if z2_input is not None and s2 is not None:
                    y2_z_rec = self.decoder(s2, z2_input)
                    z2_rec_mean, _ = self.modality_encoder(s2, y2_z_rec)
                    z_reconstructions['z2_reconstructed'] = z2_rec_mean
                
                results.update(z_reconstructions)
        
        return results

    def _discriminator_step(self, x1, mask1=None, x2=None, mask2=None, z1_input=None, z2_input=None):
        # Freeze generator parameters
        self._make_trainable(self.anatomy_encoders, False)
        self._make_trainable(self.modality_encoder, False)
        self._make_trainable(self.anatomy_fuser, False)
        self._make_trainable(self.segmentor, False)
        self._make_trainable(self.decoder, False)
        
        # Ensure discriminator parameters are trainable
        self._make_trainable(self.d_mask, True)
        self._make_trainable(self.d_image1, True)
        if hasattr(self, 'd_image2'):
            self._make_trainable(self.d_image2, True)
        
        # Get generator outputs (detached)
        with torch.no_grad():
            results = self(x1, x2, z1_input, z2_input)
        
        d_losses = []
        
        # Mask discriminator loss
        # Modality 1
        m1 = mask1[:, :self.num_masks]
        fake_m1_list = []
        fake_m1 = results['segmentation_1'][:, :self.num_masks].detach()
        fake_m1_list.append(fake_m1)
        if x2 is not None and 'segmentation_1_s2_def' in results:
            fake_m1_from_s2_def = results['segmentation_1_s2_def'][:, :self.num_masks].detach()
            fake_m1_list.append(fake_m1_from_s2_def)
        fake_masks1 = torch.cat(fake_m1_list, dim=0)
        fake_masks1 = self._sample_batch(fake_masks1, m1.shape[0])
        real_pred1 = self.d_mask(m1)
        fake_pred1 = self.d_mask(fake_masks1)
        d_mask_loss = torch.mean((real_pred1 - 1) ** 2) + torch.mean(fake_pred1 ** 2)
        d_losses.append(d_mask_loss)
        
        # Modality 2
        if x2 is not None:
            m2 = mask2[:, :self.num_masks] if mask2 is not None else mask1[:, :self.num_masks]
            fake_m2_list = []
            fake_m2 = results['segmentation_2'][:, :self.num_masks].detach()
            fake_m2_list.append(fake_m2)
            if 'segmentation_2_s1_def' in results:
                fake_m2_from_s1_def = results['segmentation_2_s1_def'][:, :self.num_masks].detach()
                fake_m2_list.append(fake_m2_from_s1_def)
            fake_mask2 = torch.cat(fake_m2_list, dim=0)
            fake_mask2 = self._sample_batch(fake_mask2, m2.shape[0])
            real_pred2 = self.d_mask(m2)
            fake_pred2 = self.d_mask(fake_mask2)
            d_mask_loss2 = (torch.mean((real_pred2 - 1) ** 2) + torch.mean(fake_pred2 ** 2))
            d_losses.append(d_mask_loss2)
        
        # Image discriminator losses
        image1_list = []
        y1a = results['reconstruction_1']
        image1_list.append(y1a)
        if x2 is not None:  # 
            if 'reconstruction_1_s2_def' in results:
                y1b = results['reconstruction_1_s2_def']   # s2_def + z1
                image1_list.append(y1b)
            if 'crossdomain_image_2_s1_def' in results:
                y1c = results['crossdomain_image_2_s1_def']  # s1_def + z1
                image1_list.append(y1c)

        image1_combined = torch.cat(image1_list, dim=0)
        image1_sampled = self._sample_batch(image1_combined, x1.shape[0])
        real_pred_1 = self.d_image1(x1)
        fake_pred_1 = self.d_image1(image1_sampled)
        d_image1_loss = (torch.mean((real_pred_1 - 1) ** 2) + torch.mean(fake_pred_1 ** 2))
        d_losses.append(d_image1_loss)
        
        if x2 is not None:
            image2_list = []
            y2a = results['reconstruction_2']
            image2_list.append(y2a)
            if 'reconstruction_2_s1_def' in results:
                y2b = results['reconstruction_2_s1_def']  # s1_def + z2
                image2_list.append(y2b)
            if 'crossdomain_image_1_s2_def' in results:
                y2c = results['crossdomain_image_1_s2_def']  # s2_def + z2
                image2_list.append(y2c)
            
            image2_combined = torch.cat(image2_list, dim=0)
            image2_sampled = self._sample_batch(image2_combined, x2.shape[0])
            real_pred_2 = self.d_image2(x2)
            fake_pred_2 = self.d_image2(image2_sampled)
            d_image2_loss = (torch.mean((real_pred_2 - 1) ** 2) + torch.mean(fake_pred_2 ** 2))
            d_losses.append(d_image2_loss)
        
        return sum(d_losses)
    
    def _generator_step(self, x1, mask1, x2=None, mask2=None, z1_input=None, z2_input=None):
        # Unfreeze generator parameters
        self._make_trainable(self.anatomy_encoders, True)
        self._make_trainable(self.modality_encoder, True)
        self._make_trainable(self.anatomy_fuser, True)
        self._make_trainable(self.segmentor, True)
        self._make_trainable(self.decoder, True)
        
        # Freeze discriminator parameters
        self._make_trainable(self.d_mask, False)
        self._make_trainable(self.d_image1, False)
        if hasattr(self, 'd_image2'):
            self._make_trainable(self.d_image2, False)
        
        results = self(x1, x2, z1_input, z2_input)
        
        g_losses = {}
        total_loss = 0
        
        # Segmentation loss (supervised)
        seg_loss = self.combined_dice_bce(results['segmentation_1'], mask1)
        if results['segmentation_2'] is not None and mask2 is not None:
            seg_loss += self.combined_dice_bce(results['segmentation_2'], mask2)
        
        if x2 is not None:
            # Cross-segmentation losses
            if 'segmentation_1_s2_def' in results:
                seg_loss += self.combined_dice_bce(results['segmentation_1_s2_def'], mask1)
        if mask2 is not None:
            if 'segmentation_2_s1_def' in results:
                seg_loss += self.combined_dice_bce(results['segmentation_2_s1_def'], mask2)
            
        g_losses['seg_loss'] = seg_loss
        total_loss += self.w_sup_M * seg_loss
        
        # Adversarial losses
        # Mask adversarial loss
        adv_mask_loss = 0
        if mask1 is not None and results['segmentation_1'] is not None:
            fake_mask1_pred = self.d_mask(results['segmentation_1'][:, :self.num_masks])
            adv_mask_loss += torch.mean((fake_mask1_pred - 1) ** 2)
            
        if mask2 is not None and results['segmentation_2'] is not None:
            fake_mask2_pred = self.d_mask(results['segmentation_2'][:, :self.num_masks])
            adv_mask_loss += torch.mean((fake_mask2_pred - 1) ** 2)
        
        # if x2 exists, we can execute cross-domain adversarial loss
        if x2 is not None:
            if 'segmentation_1_s2_def' in results:
                fake_mask1_s2_def_pred = self.d_mask(results['segmentation_1_s2_def'][:, :self.num_masks])
                adv_mask_loss += torch.mean((fake_mask1_s2_def_pred - 1) ** 2)
            if 'segmentation_2_s1_def' in results:
                fake_mask2_s1_def_pred = self.d_mask(results['segmentation_2_s1_def'][:, :self.num_masks])
                adv_mask_loss += torch.mean((fake_mask2_s1_def_pred - 1) ** 2)
        
            
        g_losses['adv_mask_loss'] = adv_mask_loss
        total_loss += self.w_adv_M * adv_mask_loss
        
        # Image adversarial losses
        fake_img1_pred = self.d_image1(results['reconstruction_1'])
        adv_img1_loss = torch.mean((fake_img1_pred - 1) ** 2)
        
        if x2 is not None:
            fake_img2_pred = self.d_image2(results['reconstruction_2'])
            adv_img2_loss = torch.mean((fake_img2_pred - 1) ** 2)
            
            if 'reconstruction_1_s2_def' in results:
                fake_img1_s2_def_pred = self.d_image1(results['reconstruction_1_s2_def'])
                adv_img1_loss += torch.mean((fake_img1_s2_def_pred - 1) ** 2)
            if 'reconstruction_2_s1_def' in results:
                fake_img2_s1_def_pred = self.d_image2(results['reconstruction_2_s1_def'])
                adv_img2_loss += torch.mean((fake_img2_s1_def_pred - 1) ** 2)
            
            g_losses['adv_img2_loss'] = adv_img2_loss
            total_loss += self.w_adv_X * adv_img2_loss
        
        g_losses['adv_img1_loss'] = adv_img1_loss
        total_loss += self.w_adv_X * adv_img1_loss
        
        # Reconstruction losses
        rec_loss = self.mae_loss(results['reconstruction_1'], x1)
        if x2 is not None:
            rec_loss += self.mae_loss(results['reconstruction_2'], x2)
            if 'reconstruction_1_s2_def' in results:
                rec_loss += self.mae_loss(results['reconstruction_1_s2_def'], x1)
            if 'reconstruction_2_s1_def' in results:
                rec_loss += self.mae_loss(results['reconstruction_2_s1_def'], x2)
        
        g_losses['rec_loss'] = rec_loss
        total_loss += self.w_rec_X * rec_loss
        
        # KL divergence losses
        kl_loss = results['kl_1'] if results['kl_1'] is not None else 0
        if x2 is not None and results['kl_2'] is not None:
            kl_loss += results['kl_2']
        
        g_losses['kl_loss'] = kl_loss
        total_loss += self.w_kl * kl_loss
        
        # Z reconstruction loss
        if 'z1_reconstructed' in results or 'z2_reconstructed' in results:
            z_rec_loss = 0
            if 'z1_reconstructed' in results and z1_input is not None:
                z_rec_loss += self.mae_loss(results['z1_reconstructed'], z1_input)
            if 'z2_reconstructed' in results and z2_input is not None:
                z_rec_loss += self.mae_loss(results['z2_reconstructed'], z2_input)
            
            g_losses['z_rec_loss'] = z_rec_loss
            total_loss += self.w_rec_Z * z_rec_loss
        
        return total_loss, g_losses


def mask_converting_show(mask):
    """
    Convert mask to RGB visualization
    Args:
        mask: [B, C, H, W] where C=4 (one-hot) or C=1 (class labels)
    Returns:
        mask_rgb: [B, 3, H, W] RGB visualization
    """
    if mask.dim() == 4 and mask.shape[1] == 4:  # [B, 4, H, W] - one-hot format
        mask_rgb = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3], 
                              device=mask.device, dtype=mask.dtype)
        # 0=background, 1=RV, 2=LV, 3=Myocardium
        mask_rgb[:, 0, :, :] = mask[:, 1, :, :]  # Red for RV
        mask_rgb[:, 1, :, :] = mask[:, 2, :, :]  # Green for LV  
        mask_rgb[:, 2, :, :] = mask[:, 3, :, :]  # Blue for Myocardium
        
    elif mask.dim() == 4 and mask.shape[1] == 1:  # [B, 1, H, W] - class labels
        mask_rgb = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3], 
                              device=mask.device, dtype=mask.dtype)
        # Create boolean masks for each class
        mask_squeeze = mask.squeeze(1)  # [B, H, W]
        rv_mask = (mask_squeeze == 1)   # [B, H, W]
        lv_mask = (mask_squeeze == 2)
        myo_mask = (mask_squeeze == 3)
        
        mask_rgb[:, 0, :, :] = rv_mask.float()   # Red for RV
        mask_rgb[:, 1, :, :] = lv_mask.float()  # Green for LV
        mask_rgb[:, 2, :, :] = myo_mask.float()  # Blue for Myocardium
        
        # Debug: check if we have any non-zero pixels
        total_pixels = torch.sum(rv_mask) + torch.sum(lv_mask) + torch.sum(myo_mask)
        if total_pixels == 0:
            print(f"Warning: No non-background pixels found in mask. "
                  f"Unique values: {torch.unique(mask_squeeze)}")
        
    else:
        raise ValueError(f"Mask dimension must be 4 and shape must be [B, 4, H, W] or [B, 1, H, W], but got {mask.shape}")
    
    return mask_rgb

if __name__ == '__main__':
    import yaml

    config_path = '/data2/liaohx/Projects/DAFNet/configs/dafnet.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    #  print(config['model'])
    
    from functions.utils_code import instantiate_from_config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = instantiate_from_config(config['model']).to(device)

    bs = 2
    img_shape = (1, 128, 128) 
    modality_vector = torch.randn(bs, 8).to(device)
    shape = (bs, ) + img_shape
    x1 = torch.randn(*shape).to(device)
    x2 = torch.randn(*shape).to(device)
    
    mask1 = torch.randn(bs, 4, 128, 128).to(device)
    mask2 = torch.randn(bs, 4, 128, 128).to(device)
    result = model(x1, x2, modality_vector, modality_vector)
    # print("Forward pass result keys:", result.keys())
    
    # for key, value in result.items():
    #     print(f"{key}: {value.shape}")
    
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    

    class DummyDataModule(pl.LightningDataModule):
        def __init__(self, batch_size=2, single_modality=False):
            super().__init__()
            self.batch_size = batch_size
            self.single_modality = single_modality
            
        def train_dataloader(self):
            def dummy_data():
                while True:
                    if self.single_modality:
                        #  print("single modality")
                        yield {
                                'cine': torch.randn(1, 128, 128).to(device),
                                'cine_mask': torch.randn( 4, 128, 128).to(device),
                            }
                    else:
                        yield {
                            'cine': torch.randn(1, 128, 128).to(device),
                            'psir': torch.randn( 1, 128, 128).to(device),
                            'cine_mask': torch.randn( 4, 128, 128).to(device),
                        'psir_mask': torch.randn( 4, 128, 128).to(device)
                    }
            
            from torch.utils.data import DataLoader, Dataset
            class DummyDataset(Dataset):
                def __init__(self, data_gen):
                    self.data_gen = data_gen
                    self.data = [next(data_gen) for _ in range(10)]
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = DummyDataset(dummy_data())
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        def val_dataloader(self):
            return self.train_dataloader()
    
    data_module = DummyDataModule(batch_size=2, single_modality=True)
    
    # Create trainer
    trainer = Trainer(
        max_epochs=1,
        devices=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        fast_dev_run=True,
    )
    
    print("\nStart training test...")
    try:
        # Run training
        trainer.fit(model, data_module)
        print("Training test completed!")
        
        # Run validation
        print("\nStart validation test...")
        trainer.validate(model, data_module)
        print("Validation test completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")