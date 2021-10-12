import cv2
import numpy as np


def calc_1d_hist(image, mask=None, hist_size=[256], hist_range=[0,256]):
    """
    Calculate the color histogram of an image, 
    calculating each channel seperately then
    putting them together into an array 
    """
    channel_res = []
    #Split the 3d array to 3 seperate arrays 
    for channel in cv2.split(image):
        #Calculate the histogram for each channel, standart hist size is 256
        #and hist range is 0-255
        hist = cv2.calcHist([channel], [0], mask, hist_size, hist_range)
        hist = cv2.normalize(hist, hist)
        channel_res.extend(hist)
    #Return the histogram as a numpy array
    return np.stack(channel_res).flatten()


def calc_3d_hist(image, mask=None, hist_size=[16,16,16], hist_range=[0,256,0,256,0,256]):
    """
    Calculate the color histogram of an image
    in 3 dimension

    Hist sizes and hist ranges should be changed according to
    the color space
    """
    hist = cv2.calcHist([image], [0,1,2], mask, hist_size, hist_range)
    hist = cv2.normalize(hist, hist)

    return hist.flatten()