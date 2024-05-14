from utils.prior_boxes import *
from utils.box_utils import *
from utils.parameters import *
import torch.nn.functional as F

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont




import torch

# Assuming `class_scores` is a tensor of shape (N, 8732, n_classes)
# `decoded_boxes` is a tensor of shape (N, 8732, 4), representing bounding box coordinates
# `n_classes` is the number of object classes
# `confidence_threshold` is the minimum confidence for a box to be retained

def filter_by_confidence(class_scores, decoded_boxes, n_classes, confidence_threshold=0.5):
    # Apply the confidence threshold across all classes (excluding background, usually class 0)
    # mask = class_scores[:, :, 1:] > confidence_threshold  # Shape (N, 8732, n_classes - 1)

    # print("class", class_scores.shape)
    # print("decoded", decoded_boxes.shape)
    probabilities = F.softmax(class_scores, dim=1)

    mask = (probabilities[: , 1:] > confidence_threshold).any(dim=1)

    # print("mask",mask)



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

    # print(locs.shape)

    # print("cx", cx.shape)
    # Convert offsets to absolute coordinates
    decoded_xmin = cx + locs[:, 0] * pw - (torch.exp(locs[:, 2]) * pw) / 2
    decoded_ymin = cy + locs[:, 1] * ph - (torch.exp(locs[:, 3]) * ph) / 2
    decoded_xmax = decoded_xmin + torch.exp(locs[:, 2]) * pw
    decoded_ymax = decoded_ymin + torch.exp(locs[:, 3]) * ph

    return torch.stack([decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax], dim=1)


def get_final_predictions(image, locs, scores):

    image_pil = Image.fromarray(image)
    image_width, image_height = image_pil.size
    locations_abs = locs.clone()
    locations_abs[:, 0] *= image_width  # xmin
    locations_abs[:, 1] *= image_height  # ymin
    locations_abs[:, 2] *= image_width  # xmax
    locations_abs[:, 3] *= image_height  # ymax


    image_np = np.array(image_pil)  # Convert back to numpy for OpenCV

    scores, class_indices = torch.max(scores, dim=1)

    boxes_clamped = torch.clone(locations_abs)
    boxes_clamped[:, 0] = torch.clamp(locations_abs[:, 0], min=0, max=image_width)  # xmin
    boxes_clamped[:, 1] = torch.clamp(locations_abs[:, 1], min=0, max=image_height)  # ymin
    boxes_clamped[:, 2] = torch.clamp(locations_abs[:, 2], min=0, max=image_width)  # xmax
    boxes_clamped[:, 3] = torch.clamp(locations_abs[:, 3], min=0, max=image_height)  # ymax

    for i, box in enumerate(boxes_clamped):
        # print("box", box)
        xmin, ymin, xmax, ymax = box.int()  # Ensure integer pixel values
        class_index = class_indices[i].item()  # Class label
        score = scores[i].item()  # Confidence score
        # print("xoord", xmin.item(), ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = xmin.item(), ymin.item(), xmax.item(), ymax.item()
        # Draw the bounding box (OpenCV uses BGR, not RGB)
        cv2.rectangle(image_np, ( xmin, ymin, xmax, ymax), (0, 0, 255), 1)  # Red box

        # Draw the label and score
        label_text = f"Class {class_index}: {score:.2f}"
        cv2.putText(image_np, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Convert back to PIL image for display
    final_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

    # Display the image
    final_image.show()  # Or use IPython.display to show in Jupyter
