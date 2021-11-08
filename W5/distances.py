import cv2
import textdistance as td
import numpy as np
from numpy import dot
from numpy.linalg import norm

"""
Different distance measures for comparing the similarity of histograms

For each function:

Args: Two array-like objects u, v

Returns: Distance between arrays

"""

def cosine(u, v):
    """
    Calculate distance with cosine similarity using scipy
    """
    return dot(u, v)/(norm(u)*norm(v))


def intersect(u, v):
    """
    Calculate the intersection of two histograms
    If the intersection is big distance must be small,
    so we multiply the intersection with -1
    """
    return -cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)

def kl_div(u, v):
    """
    Calculate the Kullback–Leibler divergence
    between two histograms
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_KL_DIV)

def hellinger(u, v):
    """
    Calculate distance with Hellinger distance
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)


def corr(u, v):
    """
    Calculate the correlation between two histograms
    If the correlation is strong, it means that they are
    similar so we subtract the score from 1
    """
    return 1-cv2.compareHist(u, v, cv2.HISTCMP_CORREL)

def chisqr(u, v):
    """
    Calculate distance with Chi-Square distance
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


color_distance_metrics = {

    "cosine": cosine,
    "intersect":intersect,
    "kl_div":kl_div,
    "hellinger":hellinger,
    "correlation":corr,
    "chisqr":chisqr

    }


def find_color_distance(u, v, method="cosine"):
    """

    Calculate distance between the vectors with the chosen method
    
    Returns: Distance between arrays

    Available distance metrics:

    cosine, \n
    intersect, \n
    kl_div, \n
    helinger, \n
    corr, \n
    chisqr
    """
    if u.shape != v.shape:
        u = np.concatenate((u, np.zeros(v.shape[0]- u.shape[0])))
    return color_distance_metrics[method](u, v)


def hamming(txt1, txt2):
    return td.hamming.normalized_distance(txt1, txt2)


def cosine_text(txt1, txt2):
    return td.cosine.normalized_distance(txt1, txt2)


def jaccard(txt1, txt2):
    return td.jaccard.normalized_distance(txt1, txt2)


def levenshtein(txt1, txt2):
    return td.levenshtein.normalized_distance(txt1, txt2)


text_distance_metrics = {
    "hamming":hamming,
    "cosine_text":cosine_text,
    "jaccard": jaccard,
    "levenshtein": levenshtein
}


def find_text_distance(txt1, txt2, method="jaccard"):
    return text_distance_metrics[method](txt1, txt2)
