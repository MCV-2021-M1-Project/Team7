from typing import Tuple
import cv2
import numpy as np
import background_removal as bg
from utils import opening, closing
import re

import pytesseract as pt
pt.pytesseract.tesseract_cmd = r'D:\Uygulamalar\Tesseract\tesseract.exe'


def blackhat(img:np.ndarray, kernel_size:Tuple[int, int]=(25,25)) -> np.ndarray:
    """
    Computes the text location mask using balckhat morphological operation. Should work well on dark text over bright background.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to get the textbox mask.
    kernel_size : Tuple(int,int)
            Shape of the kernel used for blackhat morphological operation.
    Returns
    -------
    mask : numpy array
            The mask corresponding to the believed position of the textbox on the image.
    """
    kernel = np.ones(kernel_size, np.uint8)

    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = bg.closing(img_orig, kernel_size=(10, int(img_orig.shape[1]/6) ))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0)


def tophat(img:np.ndarray, kernel_size:Tuple[int,int]=(25,25)) -> np.ndarray :
    """
    Computes the text location mask using tophat morphological operation. Should work well on bright text over dark background.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to get the textbox mask.
    kernel_size : Tuple(int,int)
            Shape of the kernel used for blackhat morphological operation.
    Returns
    -------
    mask : numpy array
            The mask corresponding to the believed position of the textbox on the image.
    """
    kernel = np.ones(kernel_size, np.uint8) 

    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    THRESHOLD = 150
    img_orig[(img_orig[:,:,0] < THRESHOLD) | (img_orig[:,:,1] < THRESHOLD) | (img_orig[:,:,2] < THRESHOLD)] = (0,0,0)
    
    img_orig = bg.closing(img_orig, kernel_size=(10, int(img_orig.shape[1]/6) ) )

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


def extract_textbox(image:np.ndarray) -> Tuple[float, Tuple[int,int,int,int]]:
    """
    Extract textbox of an image using the tophat/blackhat methods.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to get the textbox mask.
    Returns
    -------
    (tlx,tly,brx,bry) : Tuple(int,int,int,int)
            Bounding box corners location of the expected textbox location.
    """
    score_bright, bbox_bright = get_textbox(tophat(image))
    score_dark, bbox_dark = get_textbox(blackhat(image))
    if score_bright > score_dark:
        return score_bright, bbox_bright
    else:
        return score_dark, bbox_dark

def extract_textbox_hsv(image:np.ndarray):
    """
    Extract textbox of an image using the HSV color space.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to get the textbox mask.
    Returns
    -------
    (tlx,tly,brx,bry) : Tuple(int,int,int,int)
            Bounding box corners location of the expected textbox location.
    """
    #Connvert image to HSV and only keep the V channel.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image[:,:,2]
    
    #Compute image "absolute value" so that bright-over-dark = dark-over-bright = grey
    abs_v = np.absolute(image - np.amax(image) / 2)

    #Compute blackhat morphological opertation on the obtained image 
    blackhat = cv2.morphologyEx(abs_v, cv2.MORPH_BLACKHAT, np.ones((3,3), np.uint8))
    blackhat = blackhat / np.max(blackhat)
    
    #Create a mask of the believed position of the textbox
    mask = np.zeros_like(image)
    mask[blackhat > 0.45] = 1

    #Morphological filters to enhance mask quality, remove artifacts
    mask = closing(mask, (2,10)) #Fill letters
    mask = opening(mask, (4,4))
    mask = closing(mask, (1, int(image.shape[1]/6))) #Fill the gap between letters

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

    #Morphological filters to enhance mask quality, remove artifacts
    mask = bg.closing(mask, kernel_size=(9,15))
    mask = bg.opening(mask, kernel_size=(3,3))
    mask = bg.closing(mask, kernel_size=(1, int(image.shape[1]/4)))
    mask = bg.opening(mask, kernel_size=(1,2))

    #Get the biggest connected component
    component  = bg.get_biggest_connected_component(mask.astype(np.uint8))

    if np.max(component) == 0:
        return (0,0,0,0)
    else:
        # Find component's rectangle's coordinates
        coord_i = np.where(np.amax(component[:, 101:-101], axis=1))
        coord_j = np.where(np.amax(component[:, 101:-101], axis=0))
        top = coord_i[0][0]
        bottom = coord_i[0][-1]
        left = coord_j[0][0]
        right = coord_j[0][-1]

        # Expand coordinates and take original image's values in that zone
        inter = int((bottom - top) * 0.5)
        tlx = max(left - inter,0)
        tly = max(top - inter,0)
        bry = min(bottom + inter, image.shape[0] - 1)
        brx = min(right + inter, image.shape[1] - 1)

        return (tlx, tly, brx, bry)


def extract_text(img):

    bbox = extract_textbox_hsv(img)
    if bbox == (0, 0, 0, 0):
        text = " "

    else:
        text = pt.image_to_string(img[bbox[1]:bbox[3], bbox[0]:bbox[2]]).strip()
        text = " ".join(re.findall("[a-zA-Z]+", text))

    return bbox, text


