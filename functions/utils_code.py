import importlib
import numpy as np
import torch 
from functions.segmentation_mask import filter_largest_components


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class NormalDistribution(object):
    def __init__(self):
        self.mu = 0
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples
    
def mask_process(mask):
    """
    :param mask: [B, H, W]
    :return: [B, H, W]
    """
    pred_seg_mask = mask.cpu().numpy()
    pred_seg_mask_postpp = []
    for i in range(pred_seg_mask.shape[0]):  # process each mask in the batch
        pred_seg_mask_i = pred_seg_mask[i, :, :]
        pred_seg_mask_postpp_i = filter_largest_components(pred_seg_mask_i, labels4masking=(1, 2, 3))
        pred_seg_mask_postpp.append(pred_seg_mask_postpp_i)
    pred_seg_mask_postpp = torch.from_numpy(np.stack(pred_seg_mask_postpp))
    return pred_seg_mask_postpp