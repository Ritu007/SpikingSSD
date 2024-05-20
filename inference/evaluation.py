from utils.prior_boxes import *
from utils.box_utils import *
from utils.parameters import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import re
from scipy.interpolate import interp1d





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

    print("absolute locs", locations_abs)

    image_np = np.array(image_pil)  # Convert back to numpy for OpenCV

    scores, class_indices = torch.max(scores[:, 1:], dim=1)

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


def calculate_auc():
    precision =  ['0.00', '0.50', '0.33', '0.50', '0.60', '0.67', '0.71', '0.62', '0.67', '0.70', '0.73', '0.75', '0.69',
                '0.71', '0.67', '0.69', '0.71', '0.72', '0.74', '0.75', '0.76', '0.77', '0.78', '0.75', '0.76', '0.73',
                '0.74', '0.71', '0.69', '0.70', '0.68', '0.66', '0.67', '0.68', '0.66', '0.64', '0.65', '0.63', '0.62',
                '0.60', '0.61', '0.62', '0.63', '0.61', '0.62', '0.63', '0.62', '0.60', '0.59', '0.60', '0.61', '0.60',
                '0.60', '0.59', '0.58', '0.59', '0.60', '0.59', '0.59', '0.58', '0.57', '0.56', '0.56', '0.56', '0.55',
                '0.56', '0.55', '0.56', '0.57', '0.56', '0.55', '0.56', '0.55', '0.55', '0.55', '0.54', '0.55', '0.54',
                '0.54', '0.55', '0.54', '0.54', '0.54', '0.54', '0.53', '0.52', '0.52', '0.51', '0.52', '0.52', '0.52',
                '0.52', '0.53', '0.52', '0.53', '0.52', '0.53', '0.53', '0.53', '0.53', '0.53', '0.53', '0.53', '0.54',
                '0.54', '0.55', '0.54', '0.55', '0.54', '0.55', '0.54', '0.54', '0.54', '0.54', '0.54', '0.54', '0.55',
                '0.54', '0.55', '0.54', '0.54', '0.53', '0.53', '0.53', '0.53', '0.53', '0.53', '0.52', '0.53']

    recall= ['0.00', '0.01', '0.01', '0.02', '0.02', '0.03', '0.04', '0.04', '0.05', '0.05', '0.06', '0.07', '0.07',
             '0.08', '0.08', '0.09', '0.09', '0.10', '0.11', '0.12', '0.12', '0.13', '0.14', '0.14', '0.15', '0.15',
             '0.16', '0.16', '0.16', '0.16', '0.16', '0.16', '0.17', '0.18', '0.18', '0.18', '0.19', '0.19', '0.19',
             '0.19', '0.19', '0.20', '0.21', '0.21', '0.22', '0.22', '0.22', '0.22', '0.22', '0.23', '0.24', '0.24',
             '0.25', '0.25', '0.25', '0.26', '0.26', '0.26', '0.27', '0.27', '0.27', '0.27', '0.27', '0.28', '0.28',
             '0.29', '0.29', '0.29', '0.30', '0.30', '0.30', '0.31', '0.31', '0.32', '0.32', '0.32', '0.33', '0.33',
             '0.33', '0.34', '0.34', '0.34', '0.35', '0.35', '0.35', '0.35', '0.35', '0.35', '0.36', '0.36', '0.36',
             '0.37', '0.38', '0.38', '0.39', '0.39', '0.40', '0.40', '0.40', '0.41', '0.42', '0.42', '0.43', '0.43',
             '0.44', '0.45', '0.45', '0.46', '0.46', '0.47', '0.47', '0.47', '0.47', '0.48', '0.48', '0.49', '0.50',
             '0.50', '0.50', '0.50', '0.50', '0.50', '0.50', '0.51', '0.51', '0.52', '0.52', '0.52', '0.53']

    Precision= ['1.00', '0.50', '0.33', '0.50', '0.60', '0.50', '0.43', '0.38', '0.44', '0.40', '0.36', '0.33', '0.38',
                '0.36', '0.33', '0.38', '0.41', '0.39', '0.42', '0.40', '0.38', '0.36', '0.35', '0.33', '0.36', '0.35',
                '0.37', '0.39', '0.38', '0.37', '0.35', '0.38', '0.36', '0.38', '0.37', '0.39', '0.41', '0.39', '0.41',
                '0.40', '0.39', '0.38', '0.40', '0.41', '0.40', '0.39', '0.40', '0.42', '0.41', '0.42', '0.41', '0.40',
                '0.42', '0.41', '0.40', '0.39', '0.39', '0.38', '0.37', '0.38', '0.38', '0.39', '0.40', '0.39', '0.40',
                '0.41', '0.42', '0.41', '0.41', '0.40', '0.39', '0.40', '0.40', '0.41', '0.41', '0.41', '0.40', '0.41',
                '0.41', '0.40', '0.41', '0.40', '0.40', '0.40', '0.40', '0.40', '0.40', '0.41', '0.40', '0.41', '0.41',
                '0.40', '0.40', '0.39', '0.40', '0.41', '0.40', '0.41', '0.40', '0.41', '0.42', '0.42', '0.42', '0.41',
                '0.42', '0.42', '0.41', '0.41', '0.41', '0.41', '0.41', '0.40', '0.40', '0.40', '0.40', '0.41', '0.41',
                '0.42', '0.41', '0.42', '0.41', '0.42', '0.42', '0.42', '0.42']

    print(len(Precision))

    detected_objects = {
        'aeroplane': (68, 61),
        'bicycle': (53, 72),
        'bird': (83, 123),
        'boat': (83, 123),
        'bottle': (83, 123),
        'bus': (83, 123),
        'car': (83, 123),
        'cat': (142, 42),
        'chair': (108,  255),
        'cow': (31,  64),
        'diningtable': (52,  16),
        'dog': (115,  114),
        'horse': (90,  52),
        'motorbike': (81,  74),
        'person': (1414,  1746),
        'pottedplant': (30,  174),
        'sheep': (31,  57),
        'sofa': (56,  31),
        'train': (63,  26),
        'tvmonitor': (49, 98)

        # Add other classes here...
    }

    # Step 1: Sum up TP and FP across all classes
    total_tp = sum(tp for tp, _ in detected_objects.values())
    total_fp = sum(fp for _, fp in detected_objects.values())

    total_objects = total_tp + total_fp

    # Step 2: Calculate overall precision and recall
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / total_objects

    # Step 3: Plot the precision-recall curve
    plt.figure()
    plt.plot(overall_recall, overall_precision, '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('overall_precision_recall_curve.png')
    plt.show()


# calculate_auc()


# Function to parse the text file
def parse_metrics(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Regular expressions to extract the required data
    ap_regex = r'(\d+\.\d+)% = (\w+) AP'
    precision_regex = r"Precision: \[([^\]]+)\]"
    recall_regex = r"Recall :\[([^\]]+)\]"

    # Find all matches for AP, class name, precision, and recall
    ap_matches = re.findall(ap_regex, data)
    precision_matches = re.findall(precision_regex, data)
    recall_matches = re.findall(recall_regex, data)

    # Create lists to hold the parsed data
    class_names = []
    ap_values = []
    precision_values = []
    recall_values = []

    for match in ap_matches:
        ap_values.append(float(match[0]))
        class_names.append(match[1])

    for match in precision_matches:
        precision_values.append([float(x.strip("'")) for x in match.split(', ')])

    for match in recall_matches:
        recall_values.append([float(x.strip("'")) for x in match.split(', ')])

    # Create a DataFrame from the parsed data
    df = pd.DataFrame({
        'Class': class_names,
        'AP': ap_values,
        'Precision': precision_values,
        'Recall': recall_values
    })

    return df


# def plot_pr_curve(df, smooth=True):
#     # Extracting and flattening the precision and recall lists
#     all_precisions = [item for sublist in df['Precision'] for item in sublist]
#     all_recalls = [item for sublist in df['Recall'] for item in sublist]
#
#     # Sort the recall and corresponding precision values
#     sorted_indices = np.argsort(all_recalls)
#     sorted_recalls = np.array(all_recalls)[sorted_indices]
#     sorted_precisions = np.array(all_precisions)[sorted_indices]
#
#     # Smooth the curve using cubic spline interpolation
#     if smooth:
#         f = interp1d(sorted_recalls, sorted_precisions, kind='cubic')
#         recall_smooth = np.linspace(sorted_recalls.min(), sorted_recalls.max(), 100)
#         precision_smooth = f(recall_smooth)
#     else:
#         recall_smooth = sorted_recalls
#         precision_smooth = sorted_precisions
#
#     # Plot the Precision-Recall curve
#     plt.figure()
#     plt.plot(recall_smooth, precision_smooth, label='Precision-Recall Curve (Smoothed)', color='blue')
#     plt.scatter(sorted_recalls, sorted_precisions, marker='o', color='red', label='Data Points')
#
#     # Calculate the mean of the Average Precision for the title
#     average_precision = df['Average Precision'].mean()
#     plt.title(f'Precision-Recall Curve (Combined Classes) AP={average_precision:.2f}')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.show()


def plot_pr_curve(df, smooth=True):
    # Extracting and flattening the precision and recall lists
    all_precisions = [item for sublist in df['Precision'] for item in sublist]
    all_recalls = [item for sublist in df['Recall'] for item in sublist]

    # Remove duplicate values from recall array
    unique_recalls, unique_indices = np.unique(all_recalls, return_index=True)
    unique_precisions = np.array(all_precisions)[unique_indices]

    # Sort the recall and corresponding precision values
    sorted_indices = np.argsort(unique_recalls)
    sorted_recalls = unique_recalls[sorted_indices]
    sorted_precisions = unique_precisions[sorted_indices]

    # Smooth the curve using cubic spline interpolation
    if smooth:
        f = interp1d(sorted_recalls, sorted_precisions, kind='cubic', bounds_error=True)
        recall_smooth = np.linspace(sorted_recalls.min(), sorted_recalls.max(), 100)
        precision_smooth = f(recall_smooth)
    else:
        recall_smooth = sorted_recalls
        precision_smooth = sorted_precisions

    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall_smooth, precision_smooth, label='Precision-Recall Curve (Smoothed)', color='blue')
    plt.scatter(sorted_recalls, sorted_precisions, marker='o', color='red', label='Data Points')

    # Calculate the mean of the Average Precision for the title
    average_precision = df['AP'].mean()
    plt.title(f'Precision-Recall Curve (Combined Classes) mAP={average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# def plot_pr_curve(df):
#     # Flatten the Precision and Recall lists
#     all_precisions = [item for sublist in df['Precision'] for item in sublist]
#     all_recalls = [item for sublist in df['Recall'] for item in sublist]
#
#     # Sort the recall and corresponding precision values
#     sorted_indices = np.argsort(all_recalls)
#     sorted_recalls = np.array(all_recalls)[sorted_indices]
#     sorted_precisions = np.array(all_precisions)[sorted_indices]
#
#     # Plot the Precision-Recall curve
#     plt.figure()
#     plt.step(sorted_recalls, sorted_precisions, where='post', label='Precision-Recall Curve')
#
#     # Calculate the mean of the Average Precision for the title
#     average_precision = df['AP'].mean()
#     plt.title(f'Precision-Recall Curve (Combined Classes) AP={average_precision:.2f}')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend(loc='best')
#     plt.show()

filename = "E:/Python/SpikingSSD/detections/PascalVocextend/output/PR.txt"

# Use the function to parse the text file and create a DataFrame
df = parse_metrics(filename)

# Display the DataFrame
print(df)

plot_pr_curve(df)
