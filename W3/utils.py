import cv2
import numpy as np
import os
from distances import find_distance, distance_metrics
from histogram import pyramid_rep_hist
import pickle
import background_removal as bg
import evaluation as eval
import get_images_and_labels


def color_spaces():

    """
    Returns a dictionary of color spaces. This is used to convert the color space of an image.

    Available color spaces:
    RGB
    HSV
    YCRCB
    LAB (CIELAB)
    """
    # Available color_spaces
    return {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB
    }


# Search for an image in the museum dataset with the given distance metric
def image_search(hist1, hist2_arr, distance_metric="cosine", k=5):
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

    res = [find_distance(hist1, mus_hist, distance_metric) for mus_hist in hist2_arr]
    pred = np.argsort(np.array(res))[:k]

    return list(pred)


# Fetches the histograms for the given dataset
def get_histograms(dataset_name, cur_path, imgs, level, hist_method, clr_spc, hist_size):
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

    hist_method = string
                  Which histogram is going to be used? 

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

    print("Getting the histograms for the", dataset_name)

    if not os.path.exists(os.path.join(cur_path, "histograms")):
        os.mkdir("histograms")

    file_name =  "-".join(("Hist", dataset_name, hist_method, clr_spc, str(level), str(hist_size))) + ".pkl"
    file_path = os.path.join("histograms", file_name)

    # If the histograms are already calculated and stored in a pickle file
    # reads them from it
    if os.path.exists(os.path.join(cur_path, file_path)):

        file = open(os.path.join(cur_path, file_path), "rb")
        imgs_hists = pickle.load(file)

    # If the histogram for the given dataset and color space isn't calculated before
    # calculates and writes it to a pickle file

    else: 
        # If museum dataset, don't return list of lists, just a list.
        if dataset_name == "BBDD":

            imgs_cs = [cv2.cvtColor(img, color_spaces()[clr_spc]) \
                      for img in imgs]
            imgs_hists = [pyramid_rep_hist(img, clr_spc, level, hist_method, hist_size) for img in imgs_cs]

        else:    
        # If a query dataset, return a list of lists.
            imgs_hists = []

            if not isinstance(imgs[0], list):
                imgs = [[img] for img in imgs]

            for img_list in imgs:
                imgs_cs = []
                for img in img_list:
                    imgs_cs.append(pyramid_rep_hist(cv2.cvtColor(img, color_spaces()[clr_spc]), 
                                                    clr_spc, level, hist_method, hist_size))
                imgs_hists.append(imgs_cs)
            
        file = open(file_path, "wb")
        pickle.dump(imgs_hists, file)   
    
    return imgs_hists


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
    return image_search(img_hist, get_histograms("BBDD", hist_method, clr_spc, hist_size), distance_metric, k) 


#Calculates the opening of an image
def opening(image, kernel_size=(30, 30)):
    """
    Parameters
    ----------

    image = numpy array
            Image we want to do the opening.

    kernel_size = tuple of ints, optional
                  Size of the kernel which will be used for opening.

    Returns
    ----------
    image: numpy array
           Returns the transformed image.
    """
    kernel = np.ones(kernel_size, np.uint8) 
    image = cv2.erode(image, kernel, iterations=1) 
    image = cv2.dilate(image, kernel, iterations=1) 
    return image


#Calculates the closing of an image
def closing(image, kernel_size=(30, 30)):

    """
    Parameters
    ----------

    image = numpy array
            Image we want to do the closing.

    kernel_size = tuple of ints, optional
                  Size of the kernel which will be used for closing.

    Returns
    ----------
    image: numpy array
           Returns the transformed image.
    """
    kernel = np.ones(kernel_size, np.uint8) 
    image = cv2.dilate(image, kernel, iterations=1) 
    image = cv2.erode(image, kernel, iterations=1) 
    return image


# Remove the background from the images
def remove_background(imgs, cur_path, query_set, eval_masks):
    """
    Parameters
    ----------

    imgs = list of numpy arrays
           List of the image we want to remove the background.

    cur_path = string
               Current working path.

    query_set = string
                Name of the query set in which background removal will be done.

    eval_masks = boolean
                 Whether to do evaluation on mask results.

    Returns
    ----------
    res: list of lists of numpy arrays
         Returns a list of lists with the paintings in images.
    """

    print("Removing backgrounds!")

    # Masks are going to be saved as photos.
    if not os.path.exists(os.path.join(cur_path, query_set + "_bg_masks")):
        os.mkdir(query_set + "_bg_masks")

    res = []
    masks = []
    i = 0
    for img in imgs:
        # Get the mask and each painting in the image as list.
        mask, paintings, _ = bg.extract_paintings_from_image(img)
        cv2.imwrite(os.path.join(query_set + "_bg_masks", str(i).zfill(5) + ".png"), mask.astype(np.int8)*255)
        i += 1
        res.append(paintings[::-1])
        masks.append(mask)
        #print(len(paintings))

    # Calculate Precision, Recall, F1 for background removal.
    if eval_masks:
        real_masks = get_images_and_labels.get_qsd2_masks(cur_path, query_set)
        mask_res = eval.evaluate_masks(masks, real_masks)
        
        print("Precision:", np.mean([i[0] for i in mask_res]), "Recall:", np.mean([i[1] for i in mask_res]),\
              "F1:", np.mean([i[2] for i in mask_res]))

    return res

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : Tuple
        (tlx, tly, brx, bry)
    bb2 : Tuple
        K(tlx, tly, brx, bry)
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou