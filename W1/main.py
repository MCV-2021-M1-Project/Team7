import cv2
import numpy as np
import os
from distances import find_distance, distance_metrics
from histogram import calc_3d_hist, calc_1d_hist
import get_images_and_labels
import evaluation as eval
import pickle
import argparse
import sys
import glob
import csv
import background_removal as bg


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-all", "--all", action="store_true",
        help = "See results for all possible combinations")

    parser.add_argument(
        "-p", "--pickle", action="store_true", default=True,
        help = "Generate pickle file with results")

    parser.add_argument(
        "-m", "--mode", default="eval",
        help = "Choose between evaluation and test modes: eval, test")

    parser.add_argument(
        "-em", "--eval_masks", action="store_true", default=True,
        help = "Choose whether there will be a mask evaluation")

    parser.add_argument(
        "-r", "--dataset_paths", default=os.getcwd(),
        help = "Path to the folder where image datasets are. \
                Each dataset should be in a folder in this path")
        
    parser.add_argument(
        "-q", "--query_set", default="qsd1_w1",
        help = "Which query set to use: qsd1_w1, qsd2_w2, qst_w1, qst_w2")

    parser.add_argument(
        "-cs", "--color_space",  default="YCRCB",
        help = "Histogram calculation method: RGB, HSB, LAB, YCRCB")

    parser.add_argument(
        "-hm", "--hist_method", default="3d",
        help = "Histogram calculation method: 1d, 3d")

    parser.add_argument(
        "-dm", "--distance_metric", default="hellinger",
        help = "Similarity measure to compare images: \
                cosine, manhattan, euclidean, intersect, kl_div, bhattacharyya, hellinger, chisqr, correlation")

    parser.add_argument(
        "-b", "--bins",default="8", type=int, 
        help = "Number of bins to use for histograms.")

    parser.add_argument(
        "-k","--k", default="10", type=int,
        help = "Mean average precision for top-K results")

    args = parser.parse_args(args)
    return args


def remove_background(imgs, eval_masks):

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
        real_masks = get_images_and_labels.get_qsd2_masks(cur_path)
        mask_res = bg.evaluate_masks(masks, real_masks)
        
        print("Precision:", np.mean([i[0] for i in mask_res]), "Recall:", np.mean([i[1] for i in mask_res]),\
               "F1:", np.mean([i[2] for i in mask_res]))

    return res


# Search for an image in the museum dataset with the given distance metric
def image_search(hist1, hist2_arr, distance_metric="cosine", k=5):

    res = [find_distance(hist1, mus_hist, distance_metric) for mus_hist in hist2_arr]
    pred = np.argsort(np.array(res))[:k]

    return list(pred)


# Fetches the histograms for the given dataset
def get_histograms(dataset_name, imgs, hist_method, clr_spc, hist_size):

    print("Getting the histograms for the", dataset_name)

    if not os.path.exists(os.path.join(cur_path, "histograms")):
        os.mkdir("histograms")

    file_name =  "-".join(("Hist", dataset_name, hist_method, clr_spc, str(hist_size))) + ".pkl"
    file_path = os.path.join("histograms", file_name)

    # If the histograms are already calculated and stored in a pickle file
    # reads them from it
    if os.path.exists(os.path.join(cur_path, file_path)):

        file = open(os.path.join(cur_path, file_path), "rb")
        imgs_hists = pickle.load(file)

    # If the histogram for the given dataset and color space isn't calculated before
    # calculates and writes it to a pickle file
    else: 
        imgs_cs = [cv2.cvtColor(img, color_spaces[clr_spc]) \
                  for img in imgs]

        if hist_method == "1d":
            imgs_hists = [calc_1d_hist(img, clr_spc, hist_size) for img in imgs_cs] 

        elif hist_method == "3d":
            imgs_hists = [calc_3d_hist(img, clr_spc, hist_size) for img in imgs_cs] 

        else:
            raise Exception("Hist method should be '1d' or '3d'!!")

        file = open(file_path, "wb")
        pickle.dump(imgs_hists, file)   

    return imgs_hists


# Find an image in the museum dataset
def find_single_image(img, hist_method="3d", clr_spc="RGB", hist_size=[16,16,16], distance_metric="cosine", k=5):

    img_hist = calc_3d_hist(img, clr_spc, hist_size)

    return image_search(img_hist, get_histograms("BBDD", hist_method, clr_spc, hist_size), distance_metric, k)  


# Evaluate the whole query set with the given conditions
def evaluate_query_set(query_set_imgs, eval_masks, query_set="qsd1_w1", hist_method="3d", clr_spc="RGB", distance_metric="cosine", k=5, \
                       hist_size=16, pckl=True):

    museum_imgs_hists = get_histograms("BBDD", museum_imgs, hist_method, clr_spc, hist_size) 

    query_imgs_hists = get_histograms(query_set, query_set_imgs, hist_method, clr_spc, hist_size)

    qs_labels = get_images_and_labels.get_query_set_labels(cur_path, query_set)

    print("Getting results for the validation set", query_set)
    query_set_preds = []
    for query_hist in query_imgs_hists:
        query_set_preds.append(image_search(query_hist, museum_imgs_hists, distance_metric, k))

    map = round(eval.mapk(qs_labels, query_set_preds, k), 4)
    print("For Color Space:", clr_spc, "and Distance Metric:", distance_metric, \
          "and Hist. Method:", hist_method, "and k:", k, "AP is: ", map)

    if pckl:
        if not os.path.exists(os.path.join(cur_path, "eval_results")):
            os.mkdir("eval_results")

        file_name = "-".join((query_set, clr_spc, distance_metric, hist_method)) + ".pkl"
        file = open(os.path.join(cur_path, "eval_results", file_name), "wb")
        pickle.dump(query_set_preds, file)

    return query_set_preds, map


# Test the whole query set with the given conditions
def test_query_set(query_set_imgs, eval_masks, query_set="qst1_w1", hist_method="3d", clr_spc="RGB", 
                  distance_metric="cosine", k=5, hist_size=16, pckl=True):

    museum_imgs_hists = get_histograms("BBDD", museum_imgs, hist_method, clr_spc, hist_size)  

    query_imgs_hists = get_histograms(query_set, query_set_imgs, hist_method, clr_spc, hist_size)

    print("Getting results for the test set!")

    query_set_preds = []
    for query_hist in query_imgs_hists:
        query_set_preds.append(image_search(query_hist, museum_imgs_hists, distance_metric, k))

    if pckl:
        if not os.path.exists(os.path.join(cur_path, "test_results")):
            os.mkdir("test_results")

        file_name = "-".join((query_set, clr_spc, distance_metric, hist_method)) + ".pkl"
        file = open(os.path.join(cur_path, "test_results", file_name), "wb")
        pickle.dump(query_set_preds, file)

    return query_set_preds


def evaluate_all(bins, pckl, eval_masks):

    with open("eval_results.csv", "w", encoding='UTF8', newline='') as f:

        header = ['Query_Set', 'Color_Space', 'Distance_Metric', 'Hist_Method', 'K', "mAP"]
        writer = csv.writer(f)
        writer.writerow(header)

        for query_set in ["qsd1_w1", "qsd2_w1"]:
            print(query_set)
            query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)

            if query_set == "qsd2_w1":
                query_set_imgs = remove_background(query_set_imgs, eval_masks)     

            for clr_spc in color_spaces.keys():
                for distance_metric in distance_metrics.keys():

                    for hm in ["1d", "3d"]:
                        for k in [1,5,10]:
                            preds, mAP = evaluate_query_set(query_set_imgs, eval_masks, 
                                                            query_set, hm, clr_spc, distance_metric, k, hist_size=bins)

                            if pckl:
                                if not os.path.exists(os.path.join(cur_path, "eval_results")):
                                    os.mkdir("eval_results")

                                file_name = "-".join((query_set, clr_spc, distance_metric, hm)) + ".pkl"
                                file = open(os.path.join(cur_path, "eval_results", file_name), "wb")
                                pickle.dump(preds, file)
                            writer.writerow([query_set, clr_spc, distance_metric, hm, k, mAP])

    [os.remove(file) for file in glob.glob(os.path.join(cur_path, '*.pkl'))]


def test_all(bins, pckl):

    for query_set in ["qst1_w1", "qst2_w1"]:

        print(query_set)
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)
        if query_set == "qst2_w1":
            query_set_imgs = remove_background(query_set_imgs, False)

        for clr_spc in color_spaces.keys():

            for distance_metric in distance_metrics.keys():

                for hm in ["1d", "3d"]:

                    for k in [1,5,10]:
                        preds = test_query_set(query_set_imgs, False, query_set, hm, clr_spc, distance_metric, k, hist_size=bins)
                        if pckl:

                            if not os.path.exists(os.path.join(cur_path, "test_results")):
                                os.mkdir("test_results")

                            file_name = "-".join((query_set, clr_spc, distance_metric, hm)) + ".pkl"
                            file = open(os.path.join(cur_path, "test_results", file_name), "wb")
                            pickle.dump(preds, file)

    [os.remove(file) for file in glob.glob(os.path.join(cur_path, '*.pkl'))]   


if __name__ == '__main__':

    args = parse_args()
    print("Passed arguments are:", args)

    # You should change the path according to your computer
    # We can take the path as a command line argument
    cur_path = args.dataset_paths

    # Available color_spaces
    color_spaces = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB
    }

    # Get all 3 image datasets at the start
    print("### Getting Images ###")
    museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)

    if args.all:
    
        if args.mode == "eval":
            evaluate_all(args.bins, args.pickle, args.eval_masks)

        else:
            test_all(args.bins, args.pickle)

    else:

        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, args.query_set)

        if args.query_set == "qsd2_w1":
            query_set_imgs = remove_background(query_set_imgs, args.eval_masks)  

        elif args.query_set == "qst2_w1":
            query_set_imgs = remove_background(query_set_imgs, False) 

        if args.mode == "eval":
            evaluate_query_set(query_set_imgs, args.eval_masks, args.query_set, args.hist_method, 
                               args.color_space, args.distance_metric, args.k, args.bins, args.pickle)

        else:
            test_query_set(query_set_imgs, args.eval_masks, args.query_set, args.hist_method, 
                           args.color_space, args.distance_metric, args.k, args.bins, args.pickle)


