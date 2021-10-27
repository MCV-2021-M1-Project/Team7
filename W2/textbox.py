from typing import List, Tuple
import cv2
import numpy as np
from utils import opening, closing
import background_removal as bg
import matplotlib.pyplot as plt


def blackhat(img:np.ndarray, size=(25,25)) -> np.ndarray:
    """

    """
    kernel = np.ones(size, np.uint8)

    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = closing(img_orig, kernel_size=(10, int(img_orig.shape[1]/6) ))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0)


def tophat(img:np.ndarray, size=(25,25)) -> np.ndarray :
    kernel = np.ones(size, np.uint8) 

    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = closing(img_orig, kernel_size=(10, int(img_orig.shape[1]/6) ) )

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
    mask_copy = mask.copy().astype(np.uint8)
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
    MARGIN = 0.5

    tlx,tly, brx,bry = (x - int(MARGIN*h),y - int(MARGIN*h) ,
                        x+w + int(MARGIN*h) , y+h + int(MARGIN*h) )

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

def extract_textbox_hsv(image):
    """
    
    """
    abs_v = np.absolute(image - np.amax(image) / 2)

    blackhat = cv2.morphologyEx(abs_v, cv2.MORPH_BLACKHAT, np.ones((3,3), np.uint8))
    blackhat = blackhat / np.max(blackhat)
    
    mask = np.zeros_like(image)
    mask[blackhat > 0.4] = 1

    #Morphological filters
    mask = closing(mask, (2,10)) ##Fill letter
    mask = opening(mask, (4,4))
    mask = closing(mask, (1, int(image.shape[1]/6)))

    #Get the biggest connected component
    component  = bg.get_biggest_connected_component(mask)

    #Get the component bounding box
    x,y,w,h = cv2.boundingRect(component.astype(np.uint8))
    w,h = min(w, mask.shape[1]-x-1), min(h, mask.shape[0]-y-1) 
    
    box = np.zeros((image.shape[0], image.shape[1] + 202))
    mask = np.zeros_like(box)
    
    box[y:(y+h), 116:box.shape[1] - 116] = blackhat[y:(y+h), 15:image.shape[1]-15]
    box = box / np.amax(box)
    mask[box > 0.46] = 1

    #Morphological filters
    mask = closing(mask, kernel_size=(9,15))
    mask = opening(mask, kernel_size=(3,3))
    mask = closing(mask, kernel_size=(1, int(image.shape[1]/4)))
    mask = opening(mask, kernel_size=(1,2))

    #Get the biggest connected component
    component  = bg.get_biggest_connected_component(mask.astype(np.uint8))

    if np.max(component) == 0:
        return[0,0,0,0]
    else:
        # Find component's rectangle's i coordinates
        coord_i = np.where(np.amax(component[:, 101:-101], axis=1))
        coord_j = np.where(np.amax(component[:, 101:-101], axis=0))
        top = coord_i[0][0]
        bottom = coord_i[0][-1]
        left = coord_j[0][0]
        right = coord_j[0][-1]

        # Expand coordinates and take original image's values in that zone
        inter = int((bottom - top) * 0.5)
        top = top - inter if top - inter > 0 else 0
        bottom = bottom + inter if bottom + inter < image.shape[0] else image.shape[0]
        left = left - inter if left - inter > 0 else 0
        right = right + inter if right + inter < image.shape[1] else image.shape[1]

        return [left, top, right, bottom]
