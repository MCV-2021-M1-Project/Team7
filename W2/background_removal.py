import cv2
import numpy as np


def background_removal(image, limit=10):
    """
    Remove background from image.
    This functions takes an image as input and return a mask of where it believes the picture is.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to remove the background.
    Returns
    -------
    mask : numpy array
            The mask corresponding to the believed position of the painting on the image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    mask = np.zeros_like(image) + 1
    
    for i in range(h):
        #starting from the left side
        for j in range(w-1):
                if abs(int(image[i,j]) - int(image[i,j+1])) <= limit:
                        mask[i,j] = 0
                else:
                        break
        #starting from the right side
        for j in range(w-1,0,-1):
                if abs(int(image[i,j]) - int(image[i,j-1])) <= limit:
                        mask[i,j] = 0
                else:
                        break
    for j in range(w):
        for i in range(h-1):
                if abs(int(image[i,j]) - int(image[i+1,j])) <= limit:
                        mask[i,j] = 0
                else:
                        break
        for i in range(h-1,0,-1):
                if abs(int(image[i-1,j]) - int(image[i,j])) <= limit:
                        mask[i,j] = 0
                else:
                        break

    return mask

def enhance_mask(mask, bw_min_ratio=0.6):
    """
    Enhance mask quality by removing artifacts.
    This functions takes a mask as input and return a better quality mask by keeping only the biggest connected component.
    The mask is then beaing straightened.
    Parameters
    ----------
    mask : numpy array
            An array containing the mask you want to enhance.
    bw_min_ratio : float [0-1]
            The minimum ratio of white pixel in a line for this line to be considered part of the mask.
    Returns
    -------
    mask : numpy array
            The enhanced mask.
    """
    num_labels, im_labels = cv2.connectedComponents(mask)
    
    max_area, best_label = 0 , -1
    for label in range(num_labels):
        #If the background area is larger than the picture, we don't want the background
        if im_labels[0,0] == label:
            continue

        area = np.sum(im_labels == label)
        if area > max_area:
            best_label = label
            max_area = area
    
    enhanced_mask = np.array(im_labels == best_label)
    
    #Straighten the mask

    #Find the predicted mask contour (large)
    contours,_ = cv2.findContours(enhanced_mask.astype("uint8"), 1, 2)
    cnt = contours[0]
    
    #Get the Bounding Box coordinates, check that it doesn't exceed array size
    y,x,w,h = cv2.boundingRect(cnt)
    w,h = min(w, mask.shape[1]-y-1), min(h, mask.shape[0]-x-1) 

    can_improve = True

    #If most of the border isn't predicted to be from the picture, cut the border by 1 pixel. 
    while can_improve:
        #Top, right, bottom, left corners average binary value.
        borders = [np.mean(enhanced_mask[x,y:(y+w)]), np.mean(enhanced_mask[x:(x+h),y+w]) , \
                                    np.mean(enhanced_mask[x+h,y:(y+w)]), np.mean(enhanced_mask[x:(x+h),y+w]) ]
        #Check if at least one border is mostly black pixels
        if np.min(borders) < bw_min_ratio:
            #cut the "most black" border
            to_cut = np.argsort(borders)[0]
            if to_cut == 0:
                x += 1
            elif to_cut == 1 :
                w -= 1
            elif to_cut == 2:
                h -=1
            elif to_cut == 3:
                y += 1    
        else:
            can_improve = False

    straight_mask = np.zeros_like(enhanced_mask).astype("bool")
    straight_mask[x:(x+h),y:(y+w)] = True

    return straight_mask, (x,y,w,h)

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