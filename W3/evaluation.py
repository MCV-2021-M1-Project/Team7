from typing import List
import numpy as np
from utils import *
import pickle
import os
import time
import csv
from distances import find_distance, distance_metrics
import get_images_and_labels
import background_removal as bg


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    #print(f"{actual} VS {predicted}")
    scores = []
    
    #print((actual, predicted))
    for (actual, predicted) in zip(actual, predicted):
        #print(f">{(actual, predicted)}")
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p == actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            scores.append(0.0)
        else:
            scores.append(score / min(1, k))
    return scores


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    apks = [apk(a, p, k) for a, p in zip(actual, predicted)]
    return np.mean([a for a_s in apks for a in a_s])


# Evaluate the whole query set with the given conditions
def evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, query_set="qsd2_w2", hist_method="3d", clr_spc="RGB", 
                       distance_metric="cosine", k=5, hist_size=16, pckl=True):

    """
    Parameters
    ----------
    query_set_imgs = list of numpy array
                     List of images of the query set which is going to be evaluated.

    museum_imgs = list of numpy array
                  List of images of the museum images.

    cur_path = string
               Current working path

    level =  int
             Image split level

    query_set = string, optional
                Name of the query set which is going to be evaluated.

    hist_method = string, optional
                  Which histogram is going to be used? 

    clr_spc = string, optional
              What color space is going to be used?

    distance_metric = string, optional
                      Distance metric you want to use, it should be
                      from the available metrics.

    k =  int, optional
          Determines how many top results to get.

    hist_size = int, optional
                Size of the bins of histograms.

    pckl = boolean, optional
           Whether to write results to a pickle file.

    Returns 
    ----------
    query_set_preds, map: list of lists of integers and float
                          Returns the indexes of the predictions and mean average precision.

    """                        

    museum_imgs_hists = get_histograms("BBDD", cur_path, museum_imgs, level, hist_method, clr_spc, hist_size) 
    query_imgs_hists = get_histograms(query_set, cur_path, query_set_imgs, level, hist_method, clr_spc, hist_size)

    qs_labels = get_images_and_labels.get_query_set_labels(cur_path, query_set)

    print("Getting results for the validation set", query_set)

    query_set_preds = []
    
    for temp_hist in query_imgs_hists:
        temp_preds = []
        for query_hist in temp_hist:
            res = image_search(query_hist, museum_imgs_hists, distance_metric, k)

            temp_preds.append(res)

        if not temp_preds:
            for i in range(k):
                temp_preds.append(i)
            temp_preds = [temp_preds]
        query_set_preds.append(temp_preds)

    map = round(mapk(qs_labels, query_set_preds, k), 4)
    print("For Color Space:", clr_spc, "and Distance Metric:", distance_metric, \
          "and Hist. Method:", hist_method, "and k:", k, "AP is: ", map)

    if pckl:
        if not os.path.exists(os.path.join(cur_path, "eval_results")):
            os.mkdir("eval_results")

        file_name = "-".join((query_set, clr_spc, distance_metric, str(level), hist_method, str(hist_size))) + ".pkl"
        file = open(os.path.join(cur_path, "eval_results", file_name), "wb")
        pickle.dump(query_set_preds, file)

    return query_set_preds, map


# Test the whole query set with the given conditions
def test_query_set(query_set_imgs, museum_imgs, cur_path, level, query_set="qst1_w2", hist_method="3d", clr_spc="RGB", 
                   distance_metric="cosine", k=5, hist_size=16, pckl=True):

    """
    Parameters
    ----------
    query_set_imgs = list of numpy array
                     List of images of the query set which is going to be evaluated.

    museum_imgs = list of numpy array
                  List of images of the museum images.

    cur_path = string
               Current working path

    level =  int
             Image split level

    query_set = string, optional
                Name of the query set which is going to be evaluated.

    hist_method = string, optional
                  Which histogram is going to be used? 

    clr_spc = string, optional
              What color space is going to be used?

    distance_metric = string, optional
                      Distance metric you want to use, it should be
                      from the available metrics.

    k =  int, optional
          Determines how many top results to get.

    hist_size = int, optional
                Size of the bins of histograms.

    Returns 
    ----------
    query_set_preds, map: list of lists of integers and float
                          Returns the indexes of the predictions and mean average precision.

    """    

    museum_imgs_hists = get_histograms("BBDD", cur_path, museum_imgs, level, hist_method, clr_spc, hist_size)  
    query_imgs_hists = get_histograms(query_set, cur_path, query_set_imgs, level, hist_method, clr_spc, hist_size)

    print("Getting results for the test set!")

    query_set_preds = []
    
    for temp_hist in query_imgs_hists:
        temp_preds = []
        for query_hist in temp_hist:
            res = image_search(query_hist, museum_imgs_hists, distance_metric, k)

            temp_preds.append(res)

        if not temp_preds:
            for i in range(k):
                temp_preds.append(i)
            temp_preds = [temp_preds]
        query_set_preds.append(temp_preds)

    if pckl:
        if not os.path.exists(os.path.join(cur_path, "test_results")):
            os.mkdir("test_results")

        file_name = "-".join((query_set, clr_spc, distance_metric, str(level), hist_method, str(hist_size))) + ".pkl"
        file = open(os.path.join(cur_path, "test_results", file_name), "wb")
        pickle.dump(query_set_preds, file)

    return query_set_preds


# Evaluate validation query sets for each combination of
# color spaces, distance metrics and histogram methods.

def evaluate_all(bins, pckl, cur_path, level, eval_masks):

    """
    Parameters
    ----------
    bins = int
           Size of the histogram bins

    pckl = boolean, optional
           Whether to write results to a pickle file.

    cur_path = string
               Current working path

    level =  int
             Image split level

    eval_masks = boolean
                 Whether to do evaluation on mask results.

    """    

    print("### Getting Images ###")
    museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)

    with open(str(bins) + "_w2_eval_results.csv", "w", encoding='UTF8', newline='') as f:

        header = ['Query_Set', 'Color_Space', 'Distance_Metric', 'Hist_Method', 'K', "mAP"]
        writer = csv.writer(f)
        writer.writerow(header)

        for query_set in ["qsd1_w2", "qsd2_w2"]:
            start_time = time.time()
            print(query_set)
            query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)

            if query_set == "qsd2_w2":
                query_set_imgs = remove_background(query_set_imgs, cur_path, query_set, eval_masks)     

            for clr_spc in color_spaces().keys():
                for distance_metric in distance_metrics.keys():

                    for hm in ["1d", "3d"]:
                        for k in [1,5,10]:
                            _, mAP = evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                        query_set, hm, clr_spc, distance_metric, k, hist_size=bins, pckl=pckl)

                            writer.writerow([query_set, clr_spc, distance_metric, hm, k, mAP])

            print("--- %s seconds ---" % (time.time() - start_time), '\n')


# Test query sets for each combination of color spaces,
# distance metrics and histogram methods.
def test_all(bins, pckl, cur_path, level):
    """
    Parameters
    ----------
    bins = int
           Size of the histogram bins

    pckl = boolean, optional
           Whether to write results to a pickle file.

    cur_path = string
               Current working path

    level =  int
             Image split level

    """    

    print("### Getting Images ###")
    museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)

    for query_set in ["qst1_w2", "qst2_w2"]:

        print(query_set)
        start_time = time.time()
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)
        if query_set == "qst2_w2":
            query_set_imgs = remove_background(query_set_imgs, cur_path, query_set, False)

        for clr_spc in color_spaces().keys():

            for distance_metric in distance_metrics.keys():

                for hm in ["1d", "3d"]:

                    for k in [1,5,10]:
                        _ = test_query_set(query_set_imgs, museum_imgs, cur_path, level, query_set, 
                                           hm, clr_spc, distance_metric, k, hist_size=bins, pckl=pckl)

        print("--- %s seconds ---" % (time.time() - start_time), '\n')


def evaluate_mask(predicted_mask, true_mask):
    """
    Evaluates predicted mask precision, recall and F1-measure.
    Parameters
    ----------
    predicted_mask : numpy array
            An array containing the mask predicted by the background remover.
    true_mask : numpy array
            An array containing the true mask of the picture location.
    Returns
    -------
    precision : float
            Precision of the prediction = TP/(TP+FP)
    Recall : float
            Recall of the prediction = TP/(TP+FN)
    F1-Measure : float
            F1-Measure of the prediction = 2*(Precision . Recall)/(Precision + Recall)
    """

    #Compute the True Positives, False Positives, False Negatives
    TP = np.sum(true_mask*predicted_mask)
    FP = np.sum(predicted_mask) - TP
    FN = np.sum(true_mask) - TP

    #Compute the precision, recall, f1-score
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score


def evaluate_masks(predicted_masks, true_masks):
    """
    Evaluates predicted masks precision, recall and F1-measure.
    Parameters
    ----------
    predicted_masks : numpy array
            A list of arrays containing the mask predicted by the background remover.
    true_masks : numpy array
            A list of arrays containing the true mask of the picture location.
    Returns
    -------
    evaluations : list
            A list of evaluation score tuples (precision, recall, f1_score) for each mask.
    """

    evaluations = []
    for idx in range(len(predicted_masks)):
        #Compute the precision, recall, f1-score
        precision, recall, f1_score = evaluate_mask(predicted_masks[idx], true_masks[idx])
        evaluations.append((precision, recall, f1_score))

    return evaluations
