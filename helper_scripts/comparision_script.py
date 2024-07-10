import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def calculate_metrics(image, ground_truth):
    # Convert images to grayscale
    print('Calculating metrics')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

    # Calculate True Positives, False Positives, False Negatives
    true_positives = np.sum(np.logical_and(image_gray > 0, ground_truth_gray > 0))
    false_positives = np.sum(np.logical_and(image_gray > 0, ground_truth_gray == 0))
    false_negatives = np.sum(np.logical_and(image_gray == 0, ground_truth_gray > 0))

    # Calculate Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return precision, recall, f1_score

def test(input_folder, ground_truth_folder, output_xml):
    input_images = os.listdir(input_folder)
    ground_truth_images = os.listdir(ground_truth_folder)

    root = ET.Element("metrics")

    for input_image_name in input_images:
        if input_image_name.endswith(".jpg") or input_image_name.endswith(".png"):
            input_image_base = input_image_name[:-4]  # Removing extension
            input_image_suffix = input_image_base[-4:]  # Last four characters

            for gt_image_name in ground_truth_images:
                gt_image_base = gt_image_name[:-4]  # Removing extension
                gt_image_suffix = gt_image_base[-4:]  # Last four characters
                if gt_image_suffix == input_image_suffix:
                    print(f"Match found for {input_image_name} with {gt_image_name}")
                    input_image_path = os.path.join(input_folder, input_image_name)
                    ground_truth_image_path = os.path.join(ground_truth_folder, gt_image_name)

                    input_image = cv2.imread(input_image_path)
                    ground_truth_image = cv2.imread(ground_truth_image_path)

                    precision, recall, f1_score = calculate_metrics(input_image, ground_truth_image)

                    print(f"Metrics for {input_image_name}:")
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"F1 Score: {f1_score:.2f}")
                    print()

                    # Create XML elements for each metric
                    image_element = ET.SubElement(root, "image")
                    image_element.set("filename", input_image_name)
                    image_element.set("compared_with", gt_image_name)  # Add compared_with attribute
                    precision_element = ET.SubElement(image_element, "precision")
                    precision_element.text = f"{precision:.2f}"
                    recall_element = ET.SubElement(image_element, "recall")
                    recall_element.text = f"{recall:.2f}"
                    f1_score_element = ET.SubElement(image_element, "f1_score")
                    f1_score_element.text = f"{f1_score:.2f}"

                    break  # Stop iterating through ground truth images once a match is found

    # Write XML tree to file
    tree = ET.ElementTree(root)
    tree.write(output_xml)
    print('xml file generated')

if __name__ == "__main__":
    input_folder = r"C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\src\output\Outdoor_road\resized_comparision\Javed"
    ground_truth_folder = r"C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\src\output\Outdoor_road\resized_comparision\GT"
    output_xml = r"C:\Users\dgn\Desktop\DeepFTSG-main\DeepFTSG-main\helper_scripts\javed_metrics_v2.xml"
    test(input_folder, ground_truth_folder, output_xml)
