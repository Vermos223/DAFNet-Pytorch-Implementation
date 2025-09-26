from typing import Dict, Sequence
import numpy as np
import cv2


def binary_connected_components(binary_mask: np.ndarray, component_indices: Sequence[int] = (1,)):
    """
    :param binary_mask: [H, W]
    :param component_indices: a list of integers, each integer specify N-th largest connected component in
        foreground to preserve, start from 1.
        0 is background (which maybe not a connected component)
    :return: a filtered mask
    """
    mask_out = np.zeros_like(binary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8, copy=False), connectivity=8)

    index2label = np.zeros(num_labels)
    orders = np.argsort(stats[1:, 4])  # background's label is 0, exclude it
    foreground_index2label = orders[::-1] + 1  # largest components' index to smallest, +1 offset to make it as indices
    index2label[1:] = foreground_index2label

    selectable_labels = set(range(0, num_labels))
    component_indices = set(component_indices) & selectable_labels
    for index in component_indices:
        current_component_mask = (labels == index2label[index])
        mask_out[current_component_mask] = 1

    return mask_out


def filter_largest_components(mask: np.ndarray, labels4masking: Sequence[int] = (1,), is_preserve_all_labels=True):
    """
    :param mask: [H, W] mask contains multiple labels
    :param labels4masking: a list of integers, specifying which labels need to remove redundant connected component
    :param is_preserve_all_labels: only masked labels will be contained in result if is_preserve_all_labels is False,
        otherwise labels not be specified will contains all connected components in output result.
    :return: a filtered mask
    """
    mask_out = np.zeros_like(mask)
    mask_in = mask.astype(np.int64, copy=False)
    labels_in_input = set(np.unique(mask_in).tolist())
    labels_need_masking = labels_in_input & set(labels4masking)
    for label in labels_need_masking:
        current_label_mask = binary_connected_components(mask_in == label)
        mask_out[current_label_mask] = label
    if is_preserve_all_labels:
        labels_unchanged = labels_in_input - labels_need_masking - {0}
        for label in labels_unchanged:
            mask_out[mask_in == label] = label
    return mask_out
