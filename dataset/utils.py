import albumentations as A
from typing import Sequence, Tuple, Union, List, Dict, Any, Callable, Iterable
import numpy as np

def normalize_array(x, mode='std', eps=1e-16):
    """
    Normalize the input array to the specified mode.

    Args:
      x: [..., H, W]
      mode: str, 'minmax_zero_centered', 'minmax_zero2one', 'std'
      eps: float, a small number to avoid division by zero
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
    def __init__(self, mode=None):
        if mode is None:
            Tlist = []
        elif mode == 'rot90_hvflip':
            Tlist = [
                A.RandomRotate90(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5)
            ]
        elif mode == 'hvflip_affine':
            Tlist = [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(0, 360), shear=(0, 10), scale=(0.8, 1.0), translate_px=(0, 4), p=1.0)
            ]
        elif mode == 'hvflip_gamma_affinescale1.6':
            Tlist = [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.Affine(rotate=(0, 360), shear=(0, 10), scale=(0.8, 1.6), translate_px=(0, 4), p=1.0)
            ]
        else:
            raise NotImplemented

        self.T = A.Compose(
            Tlist,
            # keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True),
            bbox_params=A.BboxParams(format='albumentations', label_fields=[])
        )

    def __call__(self, images, mask=None, bboxes=None):
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

        x = _albumentataion_optional_apply(self.T, image=images, mask=mask, bboxes=bboxes)
        retvals = []
        retvals.append(x['image'].transpose(2, 0, 1))

        if mask is not None:
            retvals.append(x['mask'].transpose(2, 0, 1))
        if bboxes is not None:
            retvals.append(x['bboxes'])
        # if keypoints is not None:
        #     retvals.append(x['keypoints'])

        return tuple(retvals)


class NormalDistribution(object):
    def __init__(self):
        self.mu = 0
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples