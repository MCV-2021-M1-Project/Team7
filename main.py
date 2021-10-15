import cv2
import numpy as np
import os
from distances import find_distance, distance_metrics
from histogram import calc_3d_hist, calc_1d_hist
import get_images_and_labels
import evaluation as eval
import background_removal
import pickle
import glob


# You should change the path according to your computer
# We can take the path as a command line argument
#cur_path = "D://Belgeler//CV-Projects//M1//Week1"
cur_path = os.getcwd()

# Available color_spaces
color_spaces = {
"RGB": cv2.COLOR_BGR2RGB,
"HSV": cv2.COLOR_BGR2HSV,
"YCRCB": cv2.COLOR_BGR2YCrCb,
"LAB": cv2.COLOR_BGR2LAB
}


# Get all 3 image datasets at the start
museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)
query_set1_imgs = get_images_and_labels.get_query_set_images(cur_path, "qsd1")
query_set2_imgs = get_images_and_labels.get_query_set_images(cur_path, "qsd2")

#Get Masks from the query set 2
query_set2_masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype("bool") for mask in get_images_and_labels.get_qsd2_masks(cur_path)]

# Fetches the histograms for the given dataset
def get_histograms(dataset_name, color_space, mask, hist_size, hist_range):
    
    """
    Calculates the histogram for all the images in the dataset in the given color space
    
    Args:
    
    dataset_name: 'BBDD', 'qsd1' or 'qsd2'
    color_space: 'RGB', 'HSV', 'LAB', 'YCRCB'
    
    Returns a list of histograms
    """

    file_path = dataset_name + "_hist_" + color_space + ".pkl"

    # If the histograms are already calculated and stored in a pickle file
    # reads the histograms from it
    if os.path.exists(os.path.join(cur_path, file_path)):

        file = open(os.path.join(cur_path, file_path), "rb")
        imgs_hists = pickle.load(file)


    # If the histogram for the given dataset and color space isn't calculated before
    # calculates it and writes it to a pickle file
    elif dataset_name == "BBDD":

        imgs_cs = [cv2.cvtColor(museum_img, color_spaces[color_space]) \
                  for museum_img in museum_imgs]
        imgs_hists = [calc_3d_hist(img, mask, hist_size, hist_range) for img in imgs_cs]  


    elif dataset_name == "qsd1":

        imgs_cs = [cv2.cvtColor(query_img, color_spaces[color_space]) \
                  for query_img in query_set1_imgs]
        imgs_hists = [calc_3d_hist(query_img, mask, hist_size, hist_range) for query_img in imgs_cs]  


    elif dataset_name == "qsd2":

        imgs_cs = [cv2.cvtColor(query_img, color_spaces[color_space]) \
                  for query_img in query_set2_imgs]
        imgs_hists = [calc_3d_hist(query_img, mask, hist_size, hist_range) for query_img in imgs_cs]  

    else:
        raise Exception("Dataset name should be 'BBDD', 'qsd1' or 'qsd2'!!")


    file = open(file_path, "wb")
    pickle.dump(imgs_hists, file)   

    return imgs_hists


# Calculate the distance of the given image's histogram with the histograms of museum dataset 
# with the given distance metric
def image_search(hist1, hist2_arr, distance_metric="cosine", k=5):
    
    """
    Args: 
    
    hist1 = Histogram of the image we want to find
    hist2_arr = List of all the histograms of the museum dataset    
    
    Returns top k predictions which have the least distance with hist1
    """

    res = [find_distance(hist1, mus_hist, distance_metric) for mus_hist in hist2_arr]
    pred = np.argsort(np.array(res))[:k]

    return list(pred)


# Find an image in the museum dataset
def find_single_image(img, hist_size=[16,16,16], hist_range=[0,256,0,256,0,256], \
                      mask=None, color_space="RGB", distance_metric="cosine", k=5):
    
    """
    Find the image in the museum dataset
    
    Args:
    
    img: Should be a numpy array
    
    Returns a list of best k predictions
    """

    img_hist = calc_3d_hist(img, mask, hist_size, hist_range)

    return image_search(img_hist, get_histograms("BBDD", color_space, mask, hist_size, hist_range), distance_metric, k)  


# Evaluate the whole query set with the given conditions
def evaluate_query_set(query_set="qsd1", color_space="RGB", distance_metric="cosine", k=5, \
                       hist_size=[16,16,16], hist_range=[0,256,0,256,0,256], mask=None):
    
    """
    Evaluates the whole query set for the given parameters
    
    Args:
    
    query_set: 'qsd1' or 'qsd2'
    color_space: 'RGB', 'HSV', 'LAB', 'YCRCB'
    
    Returns the predictions of all the images in a list of lists and the mean average precision for the query
    """

    # Get the histograms
    museum_imgs_hists = get_histograms("BBDD", color_space, mask, hist_size, hist_range)   
    query_imgs_hists = get_histograms(query_set, color_space, mask, hist_size, hist_range)

    # Get ground truth labels
    qs_labels = get_images_and_labels.get_query_set_labels(cur_path, query_set)

    # For each image find the top k predictions
    query_set_preds = []
    for query_hist in query_imgs_hists:
        query_set_preds.append(image_search(query_hist, museum_imgs_hists, distance_metric, k))

    # Calculate mean average precision for the whole query set
    map = round(eval.mapk(qs_labels, query_set_preds, k), 4)
    print("For Color Space:", color_space, "and Distance Metric:", distance_metric, \
          "and k:", k, "AP is: ", map)

    return query_set_preds, map
    
    
# Evaluate both query sets for all the possible color space 
# and distance metric combinations and for k=1 and k=5 
def evaluate_all():
    for query_set in ["qsd1","qsd2"]:
        print(query_set)
        for clr_spc in color_spaces.keys():
            for distance_metric in distance_metrics.keys():
                for k in [1,5]:
                    evaluate_query_set(query_set, clr_spc, distance_metric, k, hist_size=[24,24,24])
                    
    # Removes all the pickle files
    [os.remove(file) for file in glob.glob(os.path.join(cur_path, '*.pkl'))]

def evaluate_background_removal(query_set, query_set_masks):

    predicted_masks = [background_removal.background_removal(image) for image in query_set]
    enhanced_masks = []

    for mask in predicted_masks:
        enhanced_mask, _ = background_removal.enhance_mask(mask)
        enhanced_masks.append(enhanced_mask)

    #Evaluate Masks
    scores = background_removal.evaluate_masks(enhanced_masks, query_set_masks)
    mean_precision = np.mean([score[0] for score in scores])
    mean_recall = np.mean([score[1] for score in scores])
    mean_f1 = np.mean([score[2] for score in scores])

    print(f"Average precision : {mean_precision} - Average recall : {mean_recall} - Average F1 : {mean_f1}")

if __name__ =="__main__" :
    #Query Set 1-2 painting retrieval.
    #evaluate_all()

    #Query Set 2 Background removal
    evaluate_background_removal(query_set2_imgs, query_set2_masks)