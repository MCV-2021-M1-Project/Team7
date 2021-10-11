import cv2
from scipy import spatial

"""
Different distance measures for comparing the similarity of histograms

For each function:

Args: Two array-like objects u,v

Returns: Distance between arrays

"""

def cosine(u, v):
    return spatial.distance.cosine(u, v)

def manhattan(u, v):
    return spatial.distance.cityblock(u, v)

def euclidean(u, v):
    return spatial.distance.euclidean(u, v)

def intersect(u, v):
    return -cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)

def kl_div(u, v):
    return cv2.compareHist(u, v, cv2.HISTCMP_KL_DIV)

def hellinger(u, v):
    return cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)

def bhattacharyya(u, v):
    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)

def corr(u, v):
    return 1-cv2.compareHist(u, v, cv2.HISTCMP_CORREL)

def chisqr(u, v):
    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


distance_metrics = {

    "cosine": cosine,
    "manhattan": manhattan,
    "euclidean": euclidean,
    "intersect":intersect,
    "kl_div":kl_div,
    "bhattacharyya":bhattacharyya,
    "hellinger":hellinger,
    "correlation":corr,
    "chisqr":chisqr

    }


def find_distance(u, v, method="cosine"):
    
    """
    Calculate distance between the vectors with the chosen method
    
    Returns: Distance between arrays

    Available distance metrics:

    cosine, \n
    manhattan, \n
    euclidean, \n
    intersect, \n
    kl_div, \n
    bhattacharyya, \n
    helinger, \n
    corr, \n
    chisqr
    """

    return distance_metrics[method](u, v)
