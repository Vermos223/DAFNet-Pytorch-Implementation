import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
import cv2


def morphological_postprocess(mask, min_area=50, closing_kernel_size=3, opening_kernel_size=2):
    """
    Apply morphological operations to improve mask connectivity
    
    Args:
        mask: numpy array of shape [H, W] with integer class labels
        min_area: minimum area for connected components
        closing_kernel_size: kernel size for closing operation
        opening_kernel_size: kernel size for opening operation
    
    Returns:
        processed_mask: cleaned mask
    """
    processed_mask = mask.copy()
    
    # Get unique classes (excluding background)
    classes = np.unique(mask)
    classes = classes[classes > 0]  # Remove background
    
    for class_id in classes:
        # Extract binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8)
        
        # Morphological closing to fill small gaps
        if closing_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (closing_kernel_size, closing_kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)
        
        # Morphological opening to remove small artifacts
        if opening_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (opening_kernel_size, opening_kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small connected components
        if min_area > 0:
            binary_mask = remove_small_objects(binary_mask.astype(bool), 
                                             min_size=min_area).astype(np.uint8)
        
        # Update the processed mask
        processed_mask[binary_mask == 1] = class_id
        processed_mask[binary_mask == 0] = np.where(processed_mask[binary_mask == 0] == class_id, 
                                                   0, processed_mask[binary_mask == 0])
    
    return processed_mask


def gaussian_smooth_postprocess(prediction, sigma=0.5):
    """
    Apply Gaussian smoothing to segmentation probabilities before argmax
    
    Args:
        prediction: torch tensor of shape [B, C, H, W] with class probabilities
        sigma: standard deviation for Gaussian kernel
        
    Returns:
        smoothed_prediction: smoothed probabilities
    """
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    
    # Apply Gaussian smoothing to each channel
    smoothed = torch.zeros_like(prediction)
    
    for b in range(prediction.shape[0]):
        for c in range(prediction.shape[1]):
            # Convert to numpy for scipy processing
            channel = prediction[b, c].cpu().numpy()
            smoothed_channel = ndimage.gaussian_filter(channel, sigma=sigma)
            smoothed[b, c] = torch.tensor(smoothed_channel)
    
    # Renormalize to ensure probabilities sum to 1
    smoothed = F.softmax(smoothed, dim=1)
    
    return smoothed


def connected_components_postprocess(mask, min_component_size=20):
    """
    Remove small connected components and keep only the largest component per class
    
    Args:
        mask: numpy array of shape [H, W] with integer class labels
        min_component_size: minimum size for connected components
        
    Returns:
        processed_mask: mask with small components removed
    """
    processed_mask = np.zeros_like(mask)
    
    # Get unique classes (excluding background)
    classes = np.unique(mask)
    classes = classes[classes > 0]
    
    for class_id in classes:
        # Extract binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        # If no components found, continue
        if num_labels <= 1:
            continue
            
        # Calculate component sizes
        component_sizes = []
        for label in range(1, num_labels):  # Skip background (label 0)
            size = np.sum(labels == label)
            component_sizes.append((label, size))
        
        # Sort by size and keep only components above threshold
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for label, size in component_sizes:
            if size >= min_component_size:
                component_mask = (labels == label)
                processed_mask[component_mask] = class_id
    
    return processed_mask


def comprehensive_postprocess(prediction, use_gaussian=True, use_morphology=True, 
                            use_connected_components=True, **kwargs):
    """
    Comprehensive postprocessing pipeline for segmentation results
    
    Args:
        prediction: torch tensor [B, C, H, W] or numpy array [B, C, H, W]
        use_gaussian: whether to apply Gaussian smoothing
        use_morphology: whether to apply morphological operations
        use_connected_components: whether to remove small components
        **kwargs: additional arguments for individual postprocessing steps
        
    Returns:
        processed_masks: processed segmentation masks
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu()
    
    # Convert to numpy if needed
    if isinstance(prediction, torch.Tensor):
        pred_np = prediction.numpy()
    else:
        pred_np = prediction
    
    processed_results = []
    
    for b in range(pred_np.shape[0]):
        # Get single sample
        sample_pred = pred_np[b]  # [C, H, W]
        
        # Apply Gaussian smoothing if requested
        if use_gaussian:
            sample_pred = torch.tensor(sample_pred)
            sample_pred = gaussian_smooth_postprocess(sample_pred.unsqueeze(0), 
                                                    sigma=kwargs.get('gaussian_sigma', 0.5))
            sample_pred = sample_pred.squeeze(0).numpy()
        
        # Convert to class labels
        mask = np.argmax(sample_pred, axis=0).astype(np.uint8)
        
        # Apply morphological operations if requested
        if use_morphology:
            mask = morphological_postprocess(mask, 
                                           min_area=kwargs.get('min_area', 50),
                                           closing_kernel_size=kwargs.get('closing_kernel_size', 3),
                                           opening_kernel_size=kwargs.get('opening_kernel_size', 2))
        
        # Apply connected components filtering if requested
        if use_connected_components:
            mask = connected_components_postprocess(mask, 
                                                  min_component_size=kwargs.get('min_component_size', 20))
        
        processed_results.append(mask)
    
    return np.array(processed_results)


# Convenience function for batch processing
def postprocess_batch(predictions, **kwargs):
    """
    Convenience function for postprocessing a batch of predictions
    
    Args:
        predictions: torch tensor [B, C, H, W] with class probabilities
        **kwargs: postprocessing parameters
        
    Returns:
        processed_masks: numpy array [B, H, W] with processed class labels
    """
    return comprehensive_postprocess(predictions, **kwargs)


# Default postprocessing configuration for cardiac segmentation
CARDIAC_POSTPROCESS_CONFIG = {
    'use_gaussian': True,
    'gaussian_sigma': 0.5,
    'use_morphology': True,
    'min_area': 100,           # Adjusted for cardiac structures
    'closing_kernel_size': 5,  # Larger kernel for cardiac regions
    'opening_kernel_size': 3,
    'use_connected_components': True,
    'min_component_size': 50
}
