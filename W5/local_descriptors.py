from skimage import feature
import numpy as np
from utils import lowe_filter
from get_images_and_labels import *
from evaluation import mapk
import cv2
import os
import pickle


def difference_of_gaussian(image):

    blobs_dog = feature.blob_dog(image, max_sigma=30, threshold=.1)
    points2f = np.array(blobs_dog[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_dog[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
        
    return keypoints

def ORB(img, keypoint_size):

    orb = cv2.ORB_create(keypoint_size)
    return orb.detectAndCompute(img, None)

def SIFT(img, keypoint_size):
    
    sift = cv2.SIFT_create(keypoint_size)
    return sift.detectAndCompute(img, None)

def harris_laplacian(img, keypoint_size):

    hl = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(keypoint_size)
    kp = hl.detect(img, mask=None)
    orb = cv2.ORB_create(keypoint_size)
    
    return orb.compute(img, kp)


local_descs = {
    "SIFT": SIFT,
    "ORB": ORB,
    "harris_laplacian": harris_laplacian,
    "DOG": difference_of_gaussian
}


def get_local_descriptors(method, img, keypoint_size):
    return local_descs[method](img, keypoint_size)


def match_algorithm(desc1, desc2, method="BF", k=2):
    
    if method == "flann":
        
        index = dict(algorithm=0, trees=5)
        search = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index, search)
        matches = flann.knnMatch(desc1, desc2, k=k)
        
    elif method == "BF":
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1, desc2, k=k)      
    
    return lowe_filter(matches)


def calc_local_descs_matches(museum_dataset, query_dataset, desc_method, match_method, keypoint_size):


    if not os.path.exists("descriptors"):
        os.mkdir("descriptors")

    museum_fp = "BBDD_" + "_".join((desc_method, str(keypoint_size))) + ".pkl"

    if os.path.exists(os.path.join("descriptors", museum_fp)):
        with open(os.path.join("descriptors", museum_fp), "rb") as file:
            bbdd_descs = pickle.load(file)

    else:
        bbdd_descs = []

        for img in museum_dataset:
            _, descs = get_local_descriptors(desc_method, img, keypoint_size)
            bbdd_descs.append(descs)
        
        with open(os.path.join("descriptors", museum_fp), "wb") as file:
            pickle.dump(bbdd_descs, file)
            
    img_matches = []

    for img in query_dataset:

        painting_list = []
        if not isinstance(img, list):
            img = [img]

        for painting in img:

            _, img_desc = get_local_descriptors(desc_method, painting, keypoint_size)

            if img_desc is None:
                img_desc = []

            else:
                bbdd_match = []
                for bbdd_desc in bbdd_descs:
                    bbdd_match.append(match_algorithm(bbdd_desc, img_desc, match_method))

                painting_list.append(bbdd_match)

        img_matches.append(painting_list)
        
    return img_matches


def get_map_for_local_descs(img_matches, ld_method, query_set, cur_path, match_threshold, pckl=True, mode="eval", k=10):

    final_match = []

    for img in img_matches:

        img_match = []

        for painting in img:
            painting_match = []

            for matches in painting:
                painting_match.append(len(matches))
                
            if np.max(np.array(painting_match)) < match_threshold:
                img_match.append([-1])
            else:
                res = np.argsort(np.array(painting_match))[::-1][:k]
                img_match.append([i.item() for i in res])
        
        if not img_match:
            final_match.append([[-1]])
        else:
            final_match.append(img_match)
            
    if pckl:
        query_path = mode + "_results"
        if not os.path.exists(query_path):
            os.mkdir(query_path)
         
        with open(os.path.join(query_path, ld_method + "_result.pkl"), "wb") as file:
            pickle.dump(final_match, file)
            
    if mode=="eval":
        labels = get_query_set_labels(cur_path, query_set)
        return final_match, round(mapk(labels, final_match, 1), 3)
    else:
        return final_match