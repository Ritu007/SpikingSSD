from utils.prior_boxes import *
from utils.box_utils import *
from utils.parameters import *
import torch.nn.functional as F


import torch

# Assuming `class_scores` is a tensor of shape (N, 8732, n_classes)
# `decoded_boxes` is a tensor of shape (N, 8732, 4), representing bounding box coordinates
# `n_classes` is the number of object classes
# `confidence_threshold` is the minimum confidence for a box to be retained

def filter_by_confidence(class_scores, decoded_boxes, n_classes, confidence_threshold=0.5):
    # Apply the confidence threshold across all classes (excluding background, usually class 0)
    # mask = class_scores[:, :, 1:] > confidence_threshold  # Shape (N, 8732, n_classes - 1)
    probabilities = F.softmax(class_scores, dim=2)

    mask = (probabilities[: , :, 1:] > confidence_threshold).any(dim=2)

    print("mask",mask)



    # Create indices to map the class score tensor to the original indices and class ID
    n_classes_minus_one = n_classes - 1
    # class_indices = torch.arange(1, n_classes, dtype=torch.int64, device=mask.device).repeat(class_scores.shape[0], 8732, 1)

    # print(class_indices)

    # Use the mask to filter out low confidence scores
    filtered_scores = probabilities[mask]  # Filtered scores
    filtered_boxes = decoded_boxes[mask] # Filtered boxes using the same mask
    # filtered_classes = class_indices[mask]  # Corresponding class IDs

    # print("filtered_classes", filtered_classes)


    return filtered_boxes, filtered_scores


def decoding_boxes(locs, priors):
    cx = priors[:, 0]
    cy = priors[:, 1]
    pw = priors[:, 2]
    ph = priors[:, 3]

    print(locs.shape)

    # print("cx", cx.shape)
    # Convert offsets to absolute coordinates
    decoded_xmin = cx + locs[:, 0] * pw - (torch.exp(locs[:, 2]) * pw) / 2
    decoded_ymin = cy + locs[:, 1] * ph - (torch.exp(locs[:, 3]) * ph) / 2
    decoded_xmax = decoded_xmin + torch.exp(locs[:, 2]) * pw
    decoded_ymax = decoded_ymin + torch.exp(locs[:, 3]) * ph

    return torch.stack([decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax], dim=1)


def get_final_predictions(locs, class_scores):
    pass