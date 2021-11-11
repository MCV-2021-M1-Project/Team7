import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from distances import find_text_distance, find_color_distance
import os

from textbox import extract_text
from typing import List
import pickle
from utils import color_spaces, compute_zig_zag, denoise_image


def calc_1d_hist(image, clr_spc, hist_size=256):
    """
    Calculate the color histogram of an image, 
    calculating each channel seperately then
    putting them together into an array 
    """
    channel_res = []
    
    #Split the 3d array to 3 seperate arrays 
    for i, channel in enumerate(cv2.split(image)):
        # Standart hist range is 0-255 except for the first channel of HSV
        if i==0 and clr_spc=="HSV":
            hist_range = [0, 180]
            bins = int(180/(256/hist_size))
        elif i==2 and clr_spc=="HSV":
            bins = int(hist_size)/4
        else:
            hist_range = [0, 256]
            bins = hist_size
        #Calculate the histogram for each channel, standart hist size is 256
        hist = cv2.calcHist([channel], [0], None, [int(bins)], hist_range)
        hist = cv2.normalize(hist, hist)
        channel_res.extend(hist)
    #Return the histogram as a numpy array
    return np.stack(channel_res).flatten()


def calc_3d_hist(image, clr_spc, hist_size=16):
    """
    Calculate the color histogram of an image
    in 3 dimension
    Hist sizes and hist ranges should be changed according to
    the color space
    """
    if clr_spc == "HSV":
        hist_range = [0, 180, 0, 256, 0, 256]
        bins = [hist_size*2, hist_size, int(hist_size/2)]

    elif clr_spc in ["LAB", "YCRCB"]:
        hist_range = [0, 256, 0, 256, 0, 256]
        bins = [int(hist_size/4), hist_size*2, hist_size*2]

    else:
        hist_range = [0, 256, 0, 256, 0, 256]
        bins = [hist_size, hist_size, hist_size]

    hist = cv2.calcHist([image], [0,1,2], None, bins, hist_range)
    hist = cv2.normalize(hist, hist)

    return hist.flatten()


def pyramid_rep_hist(img, clr_spc, level=4, desc_method="3d", hist_size=8):

    split_size = 2**(level-1)

    x, y = 0, 0
    h, w = img.shape[:2]
    split_h = int(np.ceil(h/split_size))
    split_w = int(np.ceil(w/split_size))

    features = []
    for i in range(x, x+h, split_h):
        for j in range(y, y+w, split_w):
            split_image = img[i:i+split_h, j:j+split_w]

            if desc_method == "1d":
                split_image = calc_1d_hist(split_image, clr_spc, hist_size)
            
            elif desc_method == "LBP" :
                split_image = lbp_histogram(split_image, points=24, radius = 3.0)
            
            elif desc_method == "DCT":
                split_image = dct_coefficients(split_image, num_coeff=25) 
            
            else:
                split_image = calc_3d_hist(split_image, clr_spc, hist_size)
            
            features.extend(split_image)

    return np.stack(features).flatten()


def dct_coefficients(image:np.ndarray, num_coeff:int=10) -> np.ndarray:
    # image --> grayscale --> DCT --> get top N coefficients using zig-zag scan
    """
    Extracts DCT coefficients from an image. 
    """    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    block_dct = cv2.dct(np.float32(image)/255.0)
    
    features = compute_zig_zag(block_dct[:10,:10])[:num_coeff]
    return features
    

def lbp_histogram(image:np.ndarray, points:int=24, radius:float=3.0) -> np.ndarray:
    """
    Extract LBP descriptors from an image.
    """    
    # image --> grayscale --> lbp --> histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (local_binary_pattern(image, points, radius, method="default")).astype(np.uint8)

    bins = points + 2
    hist = cv2.calcHist([image],[0], None, [bins], [0, bins])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


def find_imgs_with_text(qsd_imgs, cur_path, query_set, museum_texts, distance_metric, k=10):

    if not os.path.exists(os.path.join(cur_path, query_set + "_texts")):
        os.mkdir(query_set + "_texts")

    if not isinstance(qsd_imgs[0], list):
        qsd_imgs = [[img] for img in qsd_imgs]

    text_res = []
    text_dist = []
    bboxes = []

    for i, query_img in enumerate(qsd_imgs):    
        file_name = os.path.join(query_set + "_texts", str(i).zfill(5) + ".txt")

        temp_res = []
        temp_dist = []
        temp_bboxes = []

        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                text = f.readlines()

                for q_txt in text:
                    distances = [find_text_distance(q_txt.lower(), txt, distance_metric) for txt in museum_texts]
                    temp_dist.append(distances)

                    preds = np.argsort(distances)[:k]
                    temp_res.append([i.item() for i in preds])

                text_res.append(temp_res)
                text_dist.append(temp_dist)

        else:
            with open(file_name, "w") as f:

                for img in query_img:
                    img = denoise_image(img)
                    bbox, text = extract_text(img)
                    f.write(text)
                    f.write('\n')

                    distances = [find_text_distance(text.lower(), txt, distance_metric) for txt in museum_texts]

                    preds = np.argsort(distances)[:k]
                    temp_res.append([i.item() for i in preds])

                    temp_dist.append(distances)
                    temp_bboxes.append(list(bbox))

                text_res.append(temp_res)
                text_dist.append(temp_dist)
                bboxes.append(temp_bboxes)

    if bboxes: 
        with open(os.path.join(query_set + "_texts", "text_boxes.pkl"), "wb") as f:
            pickle.dump(bboxes, f)

    return text_res, text_dist


# Fetches the histograms for the given dataset
def get_descriptor(dataset_name:str, cur_path:str, imgs:List[np.ndarray], level:int=2, desc_method:str="3d", clr_spc:str="RGB", hist_size:int=16):
    """
    Parameters
    ----------
    dataset_name = string
                   Name of the dataset which histograms are going to be calculated.
    cur_path = string
               Current working path
    imgs = list of numpy array
           List of images which histograms are going to be calculated.
    level =  int
             Image split level
    desc_method = string
                  Which descriptor is going to be used? 
    clr_spc = string
              What color space is going to be used?
    hist_size = int
                Size of the bins of histograms.
    Returns 
    ----------
    imgs_hists: list of lists or a list with numpy arrays
                If museum dataset, returns a list of numpy arrays with calculated histograms
                else returns a list of lists with numpy arrays with calculated histograms.
    """

    print("Getting the", desc_method, "descriptors for the", dataset_name, "dataset")

    if not os.path.exists(os.path.join(cur_path, "descriptors")):
        os.mkdir("descriptors")

    if desc_method in ["1d", "3d"]:
        file_name =  "-".join(("Desc", dataset_name, desc_method, clr_spc, str(level), str(hist_size))) + ".pkl"
    else:
        file_name =  "-".join(("Desc", dataset_name, desc_method, str(level) )) + ".pkl"
    file_path = os.path.join("descriptors", file_name)

    # If the descriptors are already calculated and stored in a pickle file
    # reads them from it
    if os.path.exists(os.path.join(cur_path, file_path)):

        file = open(os.path.join(cur_path, file_path), "rb")
        imgs_hists = pickle.load(file)

    # If the descriptro for the given dataset and color space isn't calculated before
    # calculate and write it to a pickle file

    else: 
        # If museum dataset, don't return list of lists, just a list.
        if dataset_name == "BBDD":

            if desc_method == "text":
                imgs_hists = imgs

            else:

                if desc_method in ["1d", "3d"]:
                    imgs_cs = [cv2.cvtColor(img, color_spaces()[clr_spc]) \
                               for img in imgs]
                else: 
                    imgs_cs = imgs

                imgs_hists = [pyramid_rep_hist(img, clr_spc, level, desc_method, hist_size) for img in imgs_cs]

        else:    

            if desc_method == "text":
                imgs_hists = imgs

            else:
            # If a query dataset, return a list of lists.
                imgs_hists = []

                if not isinstance(imgs[0], list):
                    imgs = [[img] for img in imgs]

                for img_list in imgs:
                    imgs_cs = []
                    for img in img_list:
                        if desc_method in ["1d","3d"]:
                            imgs_cs.append(pyramid_rep_hist(cv2.cvtColor(img, color_spaces()[clr_spc]), 
                                                            clr_spc, level, desc_method, hist_size))
                        else:
                            imgs_cs.append(pyramid_rep_hist(img, clr_spc, level, desc_method, hist_size))
                    imgs_hists.append(imgs_cs)
            
        file = open(file_path, "wb")
        pickle.dump(imgs_hists, file)   
    
    return imgs_hists

# Search for an image in the museum dataset with the given distance metric
def image_search(desc1, desc2_arr, distance_metric="cosine", k=5):
    """
    Parameters
    ----------
    hist1 = numpy array
            Histogram of the image we want to find
    hist2_arr = list of numpy arrays
                Histograms of the museum dataset images
    distance_metric = string, optional
                      Distance metric you want to use, it should be
                      from the available metrics.
    k =  int, optional
         Determines how many top results to get.
    Returns 
    ----------
    list of predictions: list of floats
                         Top k predictions from hist2_arr which has the least distance with hist1
    """

    dists = [find_color_distance(desc1, mus_hist, distance_metric) for mus_hist in desc2_arr]


    preds = np.argsort(np.array(dists))[:k]
    preds = [i.item() for i in preds]
    
    return dists, preds


# Find an image in the museum dataset
def find_single_image(img, level, hist_method="3d", clr_spc="RGB", hist_size=[16,16,16], distance_metric="cosine", k=5):

    """
    Parameters
    ----------
    imgs = numpy array
           Numpy array of the image we want to find.
    level =  int
             Image split level
    hist_method = string, optional
                  Which histogram is going to be used? 
    clr_spc = string, optional
              What color space is going to be used?
    hist_size = int, optional
                Size of the bins of histograms.
    distance_metric = string, optional
                      Distance metric you want to use, it should be
                      from the available metrics.
    k =  int, optional
         Determines how many top results to get.
    Returns 
    ----------
    list of predictions: list of floats
                         Top k predictions from hist2_arr which has the least distance with hist1
    """

    img_hist = pyramid_rep_hist(img, level, clr_spc, hist_method, hist_size)
    return image_search(img_hist, get_descriptor("BBDD", hist_method, clr_spc, hist_size), distance_metric, k) 