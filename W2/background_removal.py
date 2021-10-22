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

def enhance_mask(mask:np.ndarray, bw_min_ratio:float=0.6) -> np.ndarray:
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

def straighten_mask(mask:np.ndarray, bw_min_ratio:float=0.6):
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
    y,x,w,h = cv2.boundingRect(cnt)
    w,h = min(w, mask.shape[1]-y-1), min(h, mask.shape[0]-x-1) 

    can_improve = True

    #If most of the border isn't predicted to be from the picture, cut the border by 1 pixel. 
    while can_improve:
        #Top, right, bottom, left corners average binary value.
        borders = [np.mean(mask[x,y:(y+w)]), np.mean(mask[x:(x+h),y+w]) , \
                                    np.mean(mask[x+h,y:(y+w)]), np.mean(mask[x:(x+h),y+w]) ]
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

    straight_mask = np.zeros_like(mask).astype("bool")
    straight_mask[x:(x+h),y:(y+w)] = True
    
    return straight_mask, (x,y,w,h)

def get_biggest_connected_component(mask:np.ndarray) -> np.ndarray:
    """
    Gets the biggest connected component from a mask.
    Parameters
    ----------
    mask: numpy array
        An array containing the mask you want to extract the biggest connected component.
    Returns
    ----------
    mask : numpy array
        An array with 1 in the biggest component and 0 outside
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

    return np.array(im_labels == best_label)

def enhance_mask_multi(mask, bw_min_ratio=0.6):
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
    MAX_PAINTINGS = 2
    OCCUPANCY_THTRRESHOLD = 0.15

    bboxes = []
    final_mask = np.zeros_like(mask)

    for i in range(MAX_PAINTINGS):
        biggest_component = get_biggest_connected_component(mask)
        mask -= biggest_component

        if np.sum(biggest_component) > OCCUPANCY_THTRRESHOLD:

            straight_mask, bbox = straighten_mask(mask = biggest_component, bw_min_ratio=bw_min_ratio)
            final_mask += straight_mask
            bboxes.append(bbox)
    
    return final_mask, bboxes
