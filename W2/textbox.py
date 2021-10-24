from typing import List, Tuple
import cv2
import numpy as np
from utils import opening, closing
import background_removal as bg


def blackhat(img:np.ndarray, size=(25,25)) -> np.ndarray:
    """

    """
    kernel = np.ones(size, np.uint8)

    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(10, int(img_orig.shape[1]/6) ))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0)


def tophat(img:np.ndarray, size=(25,25)) -> np.ndarray :
    kernel = np.ones(size, np.uint8) 

    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(10, int(img_orig.shape[1]/6) ) )

    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0)


def get_textbox_score(mask:np.ndarray) -> float:
    """

    """
    mask = mask.copy()
    
    # we generate the minimum bounding box for the extracted mask
    y,x,w,h = cv2.boundingRect(mask.astype(np.uint8))
    
    # some upper and lower thresholding depending on its size and the painting size.
    if w < 10 or h < 10 or h > w:
        return 0
    if h >= mask.shape[0]/4 or w >= mask.shape[1]*0.8:
        return 0

    # we compute the score according to its shape and its size
    shape_score = np.sum(mask[x:x+h, y:y+w]) / (w*h)

    return shape_score


def get_textbox(mask):
    """

    """
    mask_copy = mask.copy()
    MIN_SCORE = 0.2
    
    best_score = 0    
    found = False
    
    while not found:
        component = bg.get_biggest_connected_component(mask=mask_copy, check_bg=True)
        score = get_textbox_score(component) 
        
        if np.sum(component) < 1 :
            return 0, None

        if score > MIN_SCORE:
            mask = component
            best_score = score
            found = True
        else:
            mask_copy -= component

    x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))
    
    #We add a margin arround the text
    MARGIN = 0.02

    tlx,tly, brx,bry = (x - int(MARGIN*mask.shape[1]),y - int(MARGIN*mask.shape[0]) ,
                        x+w + int(MARGIN*mask.shape[1]) , y+h + int(MARGIN*mask.shape[0]) )

    return best_score, (max(0,tlx), max(0,tly), min(brx,mask.shape[1]-1), min(bry,mask.shape[0]-1))


def extract_textbox(image:np.ndarray):
    """
    """
    score_bright, bbox_bright = get_textbox(tophat(image))
    score_dark, bbox_dark = get_textbox(blackhat(image))
    if score_bright > score_dark:
        return score_bright, bbox_bright
    else:
        return score_dark, bbox_dark


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
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

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