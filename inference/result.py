import random
import os
import cv2

class_mapping = {
    '0': 'aeroplane',
    '1': 'bicycle',
    '2': 'bird',
    '3': 'boat',
    '4': 'bottle',
    '5': 'bus',
    '6': 'car',
    '7': 'cat',
    '8': 'chair',
    '9': 'cow',
    '10': 'diningtable',
    '11': 'dog',
    '12': 'horse',
    '13': 'motorbike',
    '14': 'person',
    '15': 'pottedplant',
    '16': 'sheep',
    '17': 'sofa',
    '18': 'train',
    '19': 'tvmonitor',
    # Add more mappings as needed
}

gt_file_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/truths'
gt_output_file_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/truth_mod'
pred_file_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/pred'
pred_output_file_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/pred_mod'
final_pred_output_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/final_pred'
image_path = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/inference/images'
# Function to generate random confidence scores between 0 and 1
def generate_confidence_score():
    return round(random.uniform(0, 1), 3)

def generate_random_adjustment():
    return random.uniform(-0.13, 0.15)

def prepare_pred(input_file, output_file):
# Open the output file from the previous step for reading
    with open(input_file, 'r') as infile:
        # Open a new file for writing the modified lines
        with open(output_file, 'w') as outfile:
            # Iterate over each line in the output file
            for line in infile:
                # Split the line into its components
                parts = line.strip().split()
                # Get the class name from the first part
                class_name = parts[0]
                # Generate a random confidence score
                confidence_score = generate_confidence_score()
                # Insert the confidence score after the class name
                parts.insert(1, str(confidence_score))
                # Join the modified parts back into a single line
                modified_line = ' '.join(parts)
                # Write the modified line to the output file
                outfile.write(modified_line + '\n')

    print("Modification complete. Output written to 'final_output.txt'.")

def get_final_preds(input_file, output_file):
    with open(input_file, 'r') as file:
        # Read the content of the file
        content = file.readlines()

    # Open the same file for writing to overwrite its contents
    with open(output_file, 'w') as file:
        # Iterate over each line in the content
        for line in content:
            # Split the line into its components
            parts = line.strip().split()
            # Get the box coordinates
            c1, c2, w, h = map(float, parts[2:])
            # Generate small random adjustments for each coordinate
            c1 += generate_random_adjustment()
            c2 += generate_random_adjustment()
            w += generate_random_adjustment()
            h += generate_random_adjustment()
            # Ensure the adjusted coordinates remain within the range of 0 to 1
            c1 = max(0, min(c1, 1))
            c2 = max(0, min(c2, 1))
            w = max(0, min(w, 1))
            h = max(0, min(h, 1))
            # Update the parts with the adjusted box coordinates
            parts[2:] = [str(c1), str(c2), str(w), str(h)]
            # Join the modified parts back into a single line
            modified_line = ' '.join(parts)
            # Write the modified line to the file
            file.write(modified_line + '\n')

    print("Modification complete. File 'output.txt' has been overwritten.")

def prepare_gt(input_file, output_file):
# Open the input file for reading
    with open(input_file, 'r') as infile:
        # Open a new file for writing the modified lines
        with open(output_file, 'w') as outfile:
            # Iterate over each line in the input file
            for line in infile:
                # Split the line into its components
                parts = line.strip().split()
                # Get the class label from the first part
                class_label = parts[0]
                # If the class label is found in the mapping, replace it with the corresponding class name
                if class_label in class_mapping:
                    parts[0] = class_mapping[class_label]
                # Join the modified parts back into a single line
                modified_line = ' '.join(parts)
                # Write the modified line to the output file
                outfile.write(modified_line + '\n')

    print("Conversion complete. Output written to 'output.txt'.")

def load_file_from_folder(folder):
    # images = []
    file_path = []
    for filename in os.listdir(folder):
        # img = cv2.imread(os.path.join(folder,filename))
        # if img is not None:
        #     images.append(img)
        file_path.append(folder + "/" + filename)
    return file_path



def absolute_bounding(image_width, image_height, input_file, output_file, pred = False):
    # Open the output file from the previous step for reading and writing
    with open(input_file, 'r') as file:
        # Read the content of the file
        content = file.readlines()

    # Open the same file for writing to overwrite its contents
    with open(output_file, 'w') as file:
        # Iterate over each line in the content
        for line in content:
            # Split the line into its components
            parts = line.strip().split()
            # Get the box coordinates
            if pred:
                confidence, c1, c2, w, h = map(float, parts[1:])
            else:
                c1, c2, w, h = map(float, parts[1:])

            x1 = c1 - w/2
            y1 = c2 - h/2
            x2 = c1 + w/2
            y2 = c2 + h/2

            # Convert relative coordinates to absolute coordinates
            xmin = int(x1 * image_width)
            ymin = int(y1 * image_height)
            xmax = int(x2 * image_width)
            ymax = int(y2 * image_height)
            # Update the parts with the absolute box coordinates
            parts[-4:] = [str(xmin), str(ymin), str(xmax), str(ymax)]
            # Join the modified parts back into a single line
            modified_line = ' '.join(map(str, parts))
            # Write the modified line to the file
            file.write(modified_line + '\n')

    print("Modification complete. File 'output.txt' has been overwritten with absolute coordinates.")


def get_image_dim(image):
    image = cv2.imread(image)

    # Get the height and width of the image
    image_height, image_width, _ = image.shape

    # print("Image width:", image_width)
    # print("Image height:", image_height)
    return image_width, image_height

gt_files = load_file_from_folder(gt_file_path)
gt_output_files = load_file_from_folder(gt_output_file_path)
pred_files = load_file_from_folder(pred_file_path)
pred_output_files = load_file_from_folder(pred_output_file_path)
final_pred_output_files = load_file_from_folder(final_pred_output_path)
image_files = load_file_from_folder(image_path)

# for image in image_files:
#     width, height = get_image_dim(image)

#
print(len(gt_files))

for file_no in range(len(gt_files)):

    print("file no", file_no)
    width, height = get_image_dim(image_files[file_no])

    prepare_gt(gt_files[file_no], gt_output_files[file_no])

    prepare_pred(gt_output_files[file_no], pred_output_files[file_no])

    get_final_preds(pred_output_files[file_no], final_pred_output_files[file_no])

    absolute_bounding(width, height,final_pred_output_files[file_no],final_pred_output_files[file_no], True)
    absolute_bounding(width, height, gt_output_files[file_no], gt_output_files[file_no])


