import cv2
import os
import pickle


def get_single_image(file_path):
    """
    Get a single image with OpenCV
    """
    return cv2.imread(file_path)


def get_museum_dataset(file_path):
    """
    Get the images in the entire museum_dataset 
    """
    dataset_folder = os.path.join(file_path, "BBDD")
    return [cv2.imread(os.path.join(dataset_folder, img)) for img in os.listdir(dataset_folder) if img.endswith(".jpg")]


def get_query_set_images(file_path, dataset_name="qsd1"):
    """
    Get the images in the query sets
    Dataset name argument should be "qsd1" or "qsd2"
    """
    dataset_folder = os.path.join(file_path, dataset_name + "_w1")  
    return [cv2.imread(os.path.join(dataset_folder, img)) for img in os.listdir(dataset_folder) if img.endswith(".jpg")]


def get_query_set_labels(file_path, dataset_name="qsd1"):
    """
    Get the ground truth for the museum dataset
    Dataset name argument should be "qsd1" or "qsd2"
    """
    lbls = open(os.path.join(file_path, dataset_name + "_w1", "gt_corresps.pkl"), 'rb')
    return pickle.load(lbls)


def get_qsd2_masks(file_path):
    """
    Get the ground truth for masks in query set 2
    """
    dataset_folder = os.path.join(file_path, "qsd2_w1")  
    return [cv2.imread(os.path.join(dataset_folder, img)) for img in os.listdir(dataset_folder) if img.endswith(".png")]
