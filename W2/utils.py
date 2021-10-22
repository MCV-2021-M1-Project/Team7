import cv2
import numpy as np
import os
from distances import find_distance, distance_metrics
from histogram import pyramid_rep_hist
import get_images_and_labels
import evaluation as eval
import pickle
import background_removal as bg


def color_spaces():

    # Available color_spaces
    return {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB
    }


def remove_background(imgs, cur_path, query_set, eval_masks):

    print("Removing backgrounds!")

    if not os.path.exists(os.path.join(cur_path, "masks")):
        os.mkdir("masks")

    res = []
    masks = []
    i = 0
    for img in imgs:
        mask, (x,y,w,h) = bg.enhance_mask(bg.background_removal(img))
        masks.append(mask)
        cv2.imwrite(os.path.join("masks", str(i).zfill(5) + ".png"), mask.astype(np.int8)*255)
        i += 1
        res.append(img[x:(x+h),y:(y+w)])

    if eval_masks:
        real_masks = get_images_and_labels.get_qsd2_masks(cur_path, query_set)
        mask_res = eval.evaluate_masks(masks, real_masks)
        
        print("Precision:", np.mean([i[0] for i in mask_res]), "Recall:", np.mean([i[1] for i in mask_res]),\
               "F1:", np.mean([i[2] for i in mask_res]))

    return res


# Search for an image in the museum dataset with the given distance metric
def image_search(hist1, hist2_arr, distance_metric="cosine", k=5):

    res = [find_distance(hist1, mus_hist, distance_metric) for mus_hist in hist2_arr]
    pred = np.argsort(np.array(res))[:k]

    return list(pred)


# Fetches the histograms for the given dataset
def get_histograms(dataset_name, cur_path, imgs, level, hist_method, clr_spc, hist_size):

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
        imgs_cs = [cv2.cvtColor(img, color_spaces()[clr_spc]) \
                  for img in imgs]

        imgs_hists = [pyramid_rep_hist(img, clr_spc, level, hist_method, hist_size) for img in imgs_cs]

        file = open(file_path, "wb")
        pickle.dump(imgs_hists, file)   
    
    return imgs_hists


# Find an image in the museum dataset
def find_single_image(img, level, hist_method="3d", clr_spc="RGB", hist_size=[16,16,16], distance_metric="cosine", k=5):

    img_hist = pyramid_rep_hist(img, level, clr_spc, hist_method, hist_size)

    return image_search(img_hist, get_histograms("BBDD", hist_method, clr_spc, hist_size), distance_metric, k)  