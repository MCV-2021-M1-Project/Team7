from typing import List, Tuple
import cv2
import numpy as np
from utils import opening, closing

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
    mask = closing(opening(mask, kernel_size=(25,25)), kernel_size=(25,25))
    return mask

def enhance_mask(mask:np.ndarray, bw_min_ratio:float=0.6) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
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

    biggest_component = get_biggest_connected_component(mask)
    
    #Straighten the mask
    straight_mask, (x,y,w,h) = straighten_mask(biggest_component, bw_min_ratio)

    return straight_mask, (x,y,w,h)

def straighten_mask(mask:np.ndarray, bw_min_ratio:float=0.6) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    This functions takes a mask as input and return a better quality mask by straightening it.
    Parameters
    ----------
    mask : numpy array
            An array containing the mask you want to enhance.
    bw_min_ratio : float [0-1]
            The minimum ratio of white pixel in a line for this line to be considered part of the mask.
    Returns
    -------
    straight_mask : numpy array
            The straightened mask.
    (x,y,w,h) : Tuple(int,int,int,int)
        Coordinates of the straightened mask bounding box
    """

    #Find the predicted mask contour (large)
    contours,_ = cv2.findContours(mask.astype("uint8"), 1, 2)
    cnt = contours[0]
    
    #Get the Bounding Box coordinates, check that it doesn't exceed array size
    x,y,w,h = cv2.boundingRect(cnt)
    w,h = min(w, mask.shape[1]-x-1), min(h, mask.shape[0]-y-1) 

    can_improve = True

    #If most of the border isn't predicted to be from the picture, cut the border by 1 pixel. 
    while can_improve:
        #Top, right, bottom, left corners average binary value.
        borders = [np.mean(mask[y,x:(x+w)]), np.mean(mask[y:(y+h),x+w]) , \
                                    np.mean(mask[y+h,x:(x+w)]), np.mean(mask[y:(y+h),x]) ]
        #Check if at least one border is mostly black pixels
        if np.min(borders) < bw_min_ratio:
            #cut the "most black" border
            to_cut = np.argsort(borders)[0]
            if to_cut == 0:
                y += 1
                h -= 1
            elif to_cut == 1 :
                w -= 1
            elif to_cut == 2:
                h -=1
            elif to_cut == 3:
                x += 1
                w -= 1    
        else:
            can_improve = False

    straight_mask = np.zeros_like(mask).astype("bool")
    straight_mask[y:(y+h),x:(x+w)] = True
    
    return straight_mask, (x,y,w,h)

def get_biggest_connected_component(mask:np.ndarray, check_bg:bool=True) -> np.ndarray:
    """
    Gets the biggest connected component from a mask.
    Parameters
    ----------
    mask: numpy array
        An array containing the mask you want to extract the biggest connected component.
    check_th : bool
        A boolean indicating whether or not we skip the connected component containing mainly 0.
    Returns
    ----------
    mask : numpy array
        An array with 1 in the biggest component and 0 outside
    """
    num_labels, im_labels = cv2.connectedComponents(mask)

    max_area, best_label = 0 , -1

    for label in range(num_labels):
    #If the background area is larger than the picture, we don't want the background
        if np.max(mask[im_labels == label]) == 0  and check_bg:
            continue

        area = np.sum(im_labels == label)
        if area > max_area:
            best_label = label
            max_area = area

    return np.array(im_labels == best_label)

def enhance_mask_multi(mask:np.ndarray, bw_min_ratio:float=0.6) :
    """
    Enhance mask quality by removing artifacts.
    This functions takes a mask as input and return a better quality mask by keeping only the biggest connected components.
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
    #Maximum number of paintings on the same image.
    MAX_PAINTINGS = 2
    
    #Percentage of the image occupied by the connected component to be considered a painting
    OCCUPANCY_THTRRESHOLD = 0.05

    bboxes = []
    final_mask = np.zeros_like(mask)
    for i in range(MAX_PAINTINGS):
        biggest_component = get_biggest_connected_component(mask)
        
        mask -= biggest_component

        #If the connected component is a painting
        if np.mean(biggest_component) > OCCUPANCY_THTRRESHOLD:
            #We straighten the mask and get the bounding box coordinates
            straight_mask, bbox = straighten_mask(mask = biggest_component, bw_min_ratio=bw_min_ratio)
            
            final_mask += straight_mask
            bboxes.append(bbox)
    
    return final_mask, bboxes

def extract_paintings_from_image(image:np.ndarray) -> List[np.ndarray]:
    """
    Extracts paintings from an image.
    This functions takes an image as an input and returns a list of images corresponding to the paintings found in the image.
    Parameters
    ----------
    image : numpy array
            An array containing the image you want to extract the paintings from.
    Returns
    -------
    paintings : List[numpy array]
            A list of images corresponding to the paintings found in the image.
    """
    paintings = []

    mask = background_removal(image=image)
    enhanced_mask, bboxes = enhance_mask_multi(mask=mask)

    painting_boxes = []

    for bbox in bboxes:
        x,y,w,h = bbox
        painting_boxes.append([x,y,w,h])
        paintings.append(image[y:y+h, x:x+w,:])


    return enhanced_mask, paintings, painting_boxes

