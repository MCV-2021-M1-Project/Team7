import numpy as np
from utils import color_spaces, get_iou
import pickle
import os
import time
import csv
from distances import color_distance_metrics, text_distance_metrics
import get_images_and_labels
import cv2
import background_removal as bg
from descriptor import find_imgs_with_text, denoise_image, get_descriptor, image_search
import itertools
from rotations import rotate_point, get_angles_and_rotations


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
def evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, desc_methods, mode, query_set="qsd1_w3", clr_spc="LAB", 
                       distance_metric="hellinger", text_distance="jaccard", texture_distance="correlation",
                       k=5, hist_size=8, pckl=True):

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

    results = {}
    final_preds = []

    if mode == "eval":
        qs_labels = get_images_and_labels.get_query_set_labels(cur_path, query_set)
    
    if pckl:
        if not os.path.exists(os.path.join(cur_path, mode + "_results")):
            os.mkdir(mode + "_results")

    if isinstance(desc_methods, str):
        desc_methods = [desc_methods]

    for desc_method in desc_methods:

        if desc_method == "text":

            museum_descs = get_images_and_labels.get_museum_text(cur_path)
            desc_preds, desc_dists = find_imgs_with_text(query_set_imgs, cur_path, query_set, museum_descs, text_distance, k)

        else:

            museum_imgs_descs = get_descriptor("BBDD", cur_path, museum_imgs, level, clr_spc=clr_spc, 
                                                desc_method=desc_method, hist_size=hist_size) 
                                                
            query_imgs_descs = get_descriptor(query_set, cur_path, query_set_imgs, level, clr_spc=clr_spc, 
                                              desc_method=desc_method, hist_size=hist_size)    

            print("Getting results for the validation set", query_set, "and desc method", desc_method)

            desc_preds = []
            desc_dists = []
            
            for temp_desc in query_imgs_descs:

                temp_preds = []
                temp_dists = []
                for query_desc in temp_desc:
                    
                    if desc_method in ["LBP", "DCT"]:
                        dists, res = image_search(query_desc, museum_imgs_descs, texture_distance, k)
                    else:
                        dists, res = image_search(query_desc, museum_imgs_descs, distance_metric, k)

                    temp_preds.append(res)
                    temp_dists.append(dists)

                if not temp_preds:
                    for i in range(k):
                        temp_preds.append(i)

                    temp_preds = [temp_preds]

                if not temp_dists:
                    for i in range(k):
                        temp_dists.append(i)

                    temp_dists = [temp_dists]
                    
                desc_preds.append(temp_preds)
                desc_dists.append(temp_dists)

        results[desc_method] = {"Distances": desc_dists,
                                "Preds": desc_preds}

    if len(results.keys()) != 1:
        
        for i in range(len(results[desc_methods[0]]["Distances"])):
            img_preds = []
            for j in range(len(results[desc_methods[0]]["Distances"][i])):
                total_dists = np.zeros(len(results[desc_methods[0]]["Distances"][i][j]))
                
                for key in results.keys():
                    #print(results[key]["Distances"][i][j])
                    if key in ["LBP", "DCT"]:
                        total_dists = total_dists + 0.4*np.array(results[key]["Distances"][i][j])
                    elif key == "3d":
                        total_dists = total_dists + 0.3*np.array(results[key]["Distances"][i][j])
                    else:
                        total_dists = total_dists + 0.3*np.array(results[key]["Distances"][i][j])

                total_preds = np.argsort(np.array(total_dists))[:k]
                total_preds = [k.item() for k in total_preds]
                
                img_preds.append(list(total_preds))

            final_preds.append(img_preds)

        file_name = "-".join((query_set, "-".join(desc_methods))) + ".pkl"

    else:
        final_preds = desc_preds

        if list(results.keys())[0] == "text":
            file_name = "-".join((query_set, text_distance, "-".join(desc_methods))) + ".pkl"

        elif list(results.keys())[0] in ["LBP", "DCT"]:
            file_name = "-".join((query_set, texture_distance, "-".join(desc_methods))) + ".pkl"

        else:
            file_name = "-".join((query_set, clr_spc, distance_metric, 
                str(level), "-".join(desc_methods), str(hist_size))) + ".pkl"

    file = open(os.path.join(cur_path, mode + "_results", file_name), "wb")
    pickle.dump(final_preds, file)

    if mode == "eval":

        map = round(mapk(qs_labels, final_preds, k), 4)

        if len(results.keys()) == 1:

            if list(results.keys())[0] == "text":
                print("For Distance Metric:", text_distance, \
                    "and Desc. Method:", desc_methods, "and k:", k, "AP is: ", map)

            elif list(results.keys())[0] in ["LBP", "DCT"]:
                print("For Distance Metric:", texture_distance, \
                    "and Desc. Method:", desc_methods, "and k:", k, "AP is: ", map)

            else:
                print("For Color Space:", clr_spc, "and Distance Metric:", distance_metric, \
                    "and Desc. Method:", desc_methods, "and k:", k, "AP is: ", map)

        else:
            print("For Desc. Method:", desc_methods, "and k:", k, "AP is: ", map)

        return final_preds, map


def evaluate_combs_all(pckl, cur_path, eval_masks, mode):
    ##lbp, hellinger LAB 3d, jaccard dist
    desc_methods = ["3d", "DCT", "text"]
    desc_combs = []
    for i in range(2, 4):
        desc_combs.extend(itertools.combinations(desc_methods, i))

    museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)

    current_time = time.time() * 1000

    if mode == "eval":

        file = open(str(current_time) + "_w5_eval_comb_results.csv", "w", encoding='UTF8', newline='')

        header = ['Query_Set', 'Desc_Method', 'K', "mAP"]
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)

        #query_sets = ["qsd1_w2", "qsd2_w2", "qsd1_w3", "qsd2_w3"]

        query_sets = ["qsd1_w5"]

    else:
        query_sets = ["qst1_w5"]

    for query_set in query_sets:
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)

        if int(query_set[-1]) >= 3:
            query_set_imgs = [denoise_image(img) for img in query_set_imgs]

        if query_set[-1] == "5":
            query_set_imgs, angles = get_angles_and_rotations(query_set_imgs, query_set, cur_path) 

        if query_set[3] == "2" or query_set[-1] > "4":
            _, query_set_imgs, boxes = remove_background_and_eval(query_set_imgs, cur_path, query_set, eval_masks) 

        if query_set[-1] == "5":
            if mode == "eval":
                box_acc, angle_acc, frame_res = get_frames_and_eval(query_set_imgs, query_set, cur_path,
                                                                    angles, boxes, mode)
                print("Box accuracy:", box_acc, "Angle acc:", angle_acc) 
                
            else:
                frame_res = get_frames_and_eval(query_set_imgs, query_set, cur_path,
                                                angles, boxes, mode)

        for comb in desc_combs:
            for k in [1,5,10]:
                if mode == "eval":
                    _, mAP = evaluate_query_set(query_set_imgs, museum_imgs, cur_path, 4, 
                                                comb, mode, query_set, k=k, pckl=pckl)
                    writer.writerow([query_set, comb, k, mAP])
                else:
                    evaluate_query_set(query_set_imgs, museum_imgs, cur_path, 4, 
                                        comb, mode, query_set, k=k, pckl=pckl)


# Evaluate validation query sets for each combination of
# color spaces, distance metrics and histogram methods.
def evaluate_all(bins, pckl, cur_path, level, eval_masks, mode):

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

    current_time = time.time() * 1000

    if mode == "eval":

        file = open(str(current_time) + "_w5_eval_results.csv", "w", encoding='UTF8', newline='')

        header = ['Query_Set', 'Color_Space', 'Distance_Metric', 'Desc_Method', 'K', "mAP"]
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)

        #query_sets = ["qsd1_w2", "qsd2_w2", "qsd1_w3", "qsd2_w3"]
        query_sets = ["qsd1_w5"] 

    else:
        query_sets = ["qst1_w5"]

    for query_set in query_sets:
        start_time = time.time()
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, query_set)

        if int(query_set[-1]) >= 3:
            query_set_imgs = [denoise_image(img) for img in query_set_imgs]

        if query_set[-1] == "5":
            query_set_imgs, angles = get_angles_and_rotations(query_set_imgs, query_set, cur_path) 

        if query_set[3] == "2" or query_set[-1] > "4":
            _, query_set_imgs, boxes = remove_background_and_eval(query_set_imgs, cur_path, query_set, eval_masks)     

        if query_set[-1] == "5":
            if mode == "eval":
                box_acc, angle_acc, frame_res = get_frames_and_eval(query_set_imgs, query_set, cur_path,
                                                                    angles, boxes, mode)
                print("Box accuracy:", box_acc, "Angle acc:", angle_acc)   
            else:
                frame_res = get_frames_and_eval(query_set_imgs, query_set, cur_path,
                                                angles, boxes, mode)

        for hm in ["1d", "3d", "LBP", "DCT", "text"]:
            for k in [1,5,10]:

                if hm == "text":
                    for txt_distance in  text_distance_metrics.keys():
                        if mode == "eval":
                            _, mAP = evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                        hm, mode, query_set, text_distance=txt_distance, k=k, 
                                                        hist_size=bins, pckl=pckl)
                        
                            writer.writerow([query_set, clr_spc, txt_distance, hm, k, mAP])

                        else:
                            evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                        hm, mode, query_set, text_distance=txt_distance, k=k, 
                                                        hist_size=bins, pckl=pckl)

                else:                    
                    for distance_metric in color_distance_metrics.keys():
                        if hm in ["1d", "3d"]:
                            for clr_spc in color_spaces().keys():
                                if mode == "eval":

                                    _, mAP = evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                                hm, mode, query_set, clr_spc, distance_metric, k=k, 
                                                                hist_size=bins, pckl=pckl)

                                    writer.writerow([query_set, clr_spc, distance_metric, hm, k, mAP])

                                else:
                                    evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                        hm, mode, query_set, clr_spc, distance_metric, k=k, 
                                                        hist_size=bins, pckl=pckl)

                        else:
                            if mode == "eval":
                                _, mAP = evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                                hm, mode, query_set, texture_distance=distance_metric, k=k, 
                                                                hist_size=bins, pckl=pckl)
                                writer.writerow([query_set, clr_spc, distance_metric, hm, k, mAP])

                            else:
                                evaluate_query_set(query_set_imgs, museum_imgs, cur_path, level, 
                                                hm, mode, query_set, clr_spc, texture_distance=distance_metric, k=k, 
                                                hist_size=bins, pckl=pckl)

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


# Remove the background from the images
def remove_background_and_eval(imgs, cur_path, query_set, eval_masks):
    """
    Parameters
    ----------
    imgs = list of numpy arrays
           List of the image we want to remove the background.
    cur_path = string
               Current working path.
    query_set = string
                Name of the query set in which background removal will be done.
    eval_masks = boolean
                 Whether to do evaluation on mask results.
    Returns
    ----------
    res: list of lists of numpy arrays
         Returns a list of lists with the paintings in images.
    """

    print("Removing backgrounds!")

    # Masks are going to be saved as photos.
    if not os.path.exists(os.path.join(cur_path, query_set + "_bg_masks")):
        os.mkdir(query_set + "_bg_masks")

    res = []
    masks = []
    bboxes = []
    i = 0
    for img in imgs:
        # Get the mask and each painting in the image as list.
        mask, paintings, boxes = bg.extract_paintings_from_image(img)

        cv2.imwrite(os.path.join(query_set + "_bg_masks", str(i).zfill(5) + ".png"), mask.astype(np.int8)*255)
        i += 1
        
        masks.append(mask)
        if len(paintings) == 3:
            sorted_painting = sorted(zip(paintings, boxes), key=lambda x:x[1][0])
        
            res.append([painting for painting, box in sorted_painting])
            bboxes.append([box for painting, box in sorted_painting])
            
        else:
            res.append(paintings)
            bboxes.append(boxes)
        #print(len(paintings))

    # Calculate Precision, Recall, F1 for background removal.
    if eval_masks:
        real_masks = get_images_and_labels.get_qsd2_masks(cur_path, query_set)
        mask_res = evaluate_masks(masks, real_masks)
        
        print("Precision:", np.mean([i[0] for i in mask_res]), "Recall:", np.mean([i[1] for i in mask_res]),\
              "F1:", np.mean([i[2] for i in mask_res]))

    return masks, res, bboxes



def get_frames_and_eval(qsd_images, query_set, cur_path, angles, boxes, mode="eval"):
    frame_res = []

    for img, angle, img_box in zip(qsd_images, angles, boxes):
        
        temp_res = []

        rot_angle = angle

        if rot_angle > 90:
            rot_angle = angle - 180.0
        else: 
            rot_angle = angle

        for painting, box in zip(img, img_box):
            (x1, y1) = rotate_point(painting, (box[0], box[1]), rot_angle)
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            (x2, y2) = rotate_point(painting, (box[0]+box[2], box[1]), rot_angle)
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0

            (x3, y3) = rotate_point(painting, (box[0]+box[2], box[1]+box[3]), rot_angle)
            if x3 < 0:
                x3 = 0
            if y3 < 0:
                y3 = 0

            (x4, y4) = rotate_point(painting, (box[0], box[1]+box[3]), rot_angle)

            if x4 < 0:
                x4 = 0
            if y4 < 0:
                y4 = 0

            temp_res.append([angle, [[x1, y1], [x2, y2], [x3, y3], [x4,y4]]])
        frame_res.append(temp_res)
        
    with open(query_set + "_frame_res.pkl", "wb") as f:
        pickle.dump(frame_res, f)

    if mode == "test":
        return frame_res
    
    else:
        labels = get_images_and_labels.get_query_set_frames(cur_path, query_set)
        box_acc = 0
        angle_acc = 0
        counter = 0
        
        for i, img in enumerate(frame_res):
            for j, painting in enumerate(img):
                sorted_bbs = np.sort(np.array(painting[1]), axis=0)
                tlx1, tly1 = sorted_bbs[0]
                brx1, bry1 = sorted_bbs[-1]
                
                p_box = (tlx1, tly1, brx1, bry1)
                p_angle = painting[0]+90 if painting[0] < 90 else 180-painting[0]

                sorted_lbls = np.sort(np.array(labels[i][j][1]), axis=0)
                tlx2, tly2 = sorted_lbls[0]
                brx2, bry2 = sorted_lbls[-1]
                
                r_box = (tlx2, tly2, brx2, bry2)
                r_angle = labels[i][j][0] if labels[i][j][0] < 90 else 180-labels[i][j][0]
                
                box_acc = box_acc + (get_iou(r_box, p_box))
                angle_acc = angle_acc + np.abs(p_angle-r_angle)
                counter += 1

        box_acc = box_acc/counter
        angle_acc = angle_acc/counter
        
        return box_acc, angle_acc, frame_res