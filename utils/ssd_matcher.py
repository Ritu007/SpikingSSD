from typing import Dict, List
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops
# import pytorch_lightning as pl

def apply_offsets(anchor_boxes, predicted_offsets):
    """
    Apply predicted offsets to anchor boxes to obtain predicted bounding boxes.

    Args:
    - anchor_boxes: Tensor of anchor boxes, shape (num_anchors, 4), representing the coordinates of anchor boxes.
    - predicted_offsets: Tensor of predicted offsets, shape (num_anchors, 4), representing the predicted offsets for each anchor box.

    Returns:
    - predicted_boxes: Tensor of predicted bounding boxes, shape (num_anchors, 4), representing the coordinates of predicted bounding boxes.
    """
    # Calculate predicted bounding box coordinates by adding predicted offsets to anchor boxes
    predicted_boxes = anchor_boxes + predicted_offsets

    predicted_boxes = torch.clamp(predicted_boxes, 0, 1)

    return predicted_boxes


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply non-maximum suppression (NMS) to filter out redundant detections.

    Args:
    - boxes: Tensor of bounding boxes, shape (num_boxes, 4), representing the coordinates of bounding boxes.
    - scores: Tensor of confidence scores, shape (num_boxes,), representing the confidence scores for each bounding box.
    - threshold: IoU threshold for NMS.

    Returns:
    - selected_indices: Indices of selected bounding boxes after NMS.
    """
    # Sort boxes by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    selected_indices = []

    while sorted_indices.numel() > 0:
        # Select the box with the highest score
        max_index = sorted_indices[0]
        selected_indices.append(max_index.item())

        # Calculate IoU between the selected box and other boxes
        ious = calculate_iou(boxes[max_index].unsqueeze(0), boxes[sorted_indices[1:]])

        # Filter out boxes with IoU greater than the threshold
        keep_indices = (ious <= threshold)
        sorted_indices = sorted_indices[1:][keep_indices]

    return torch.tensor(selected_indices)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
    - box1: Tensor of bounding boxes, shape (N, 4), representing the coordinates of bounding boxes.
    - box2: Tensor of bounding boxes, shape (M, 4), representing the coordinates of bounding boxes.

    Returns:
    - iou: Tensor of IoU values, shape (N, M).
    """
    # Calculate intersection coordinates
    xmin = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    ymin = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    xmax = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    ymax = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    # Calculate intersection area
    intersection_area = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)

    # Calculate area of box1
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])

    # Calculate area of box2
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Calculate union area
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


