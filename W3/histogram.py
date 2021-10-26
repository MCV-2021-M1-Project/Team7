import cv2
import numpy as np
from PIL import Image

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
        hist = cv2.calcHist([channel], [0], None, [bins], hist_range)
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


def pyramid_rep_hist(img, clr_spc, level=2, hist_method="3d", hist_size=8):

    split_size = 2**(level-1)

    x, y = 0, 0
    h, w = img.shape[:2]
    split_h = int(np.ceil(h/split_size))
    split_w = int(np.ceil(w/split_size))

    features = []
    for i in range(x, x+h, split_h):
        for j in range(y, y+w, split_w):
            split_image = img[i:i+split_h, j:j+split_w]

            if hist_method == "1d":
                split_image = calc_1d_hist(split_image, clr_spc, hist_size)
            else:
                split_image = calc_3d_hist(split_image, clr_spc, hist_size)
            
            features.extend(split_image)

    return np.stack(features).flatten()
