import cv2
import numpy as np


def color_spaces():

    """
    Returns a dictionary of color spaces. This is used to convert the color space of an image.
    Available color spaces:
    RGB
    HSV
    YCRCB
    LAB (CIELAB)
    """
    # Available color_spaces
    return {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB,
    #"GRAY": cv2.COLOR_BGR2GRAY
    }


#Calculates the opening of an image
def opening(image, kernel_size=(30, 30)):
    """
    Parameters
    ----------
    image = numpy array
            Image we want to do the opening.
    kernel_size = tuple of ints, optional
                  Size of the kernel which will be used for opening.
    Returns
    ----------
    image: numpy array
           Returns the transformed image.
    """
    kernel = np.ones(kernel_size, np.uint8) 
    image = cv2.erode(image, kernel, iterations=1) 
    image = cv2.dilate(image, kernel, iterations=1) 
    return image


#Calculates the closing of an image
def closing(image, kernel_size=(30, 30)):

    """
    Parameters
    ----------
    image = numpy array
            Image we want to do the closing.
    kernel_size = tuple of ints, optional
                  Size of the kernel which will be used for closing.
    Returns
    ----------
    image: numpy array
           Returns the transformed image.
    """
    kernel = np.ones(kernel_size, np.uint8) 
    image = cv2.dilate(image, kernel, iterations=1) 
    image = cv2.erode(image, kernel, iterations=1) 
    return image



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : Tuple
        (tlx, tly, brx, bry)
    bb2 : Tuple
        K(tlx, tly, brx, bry)
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def compute_zig_zag(array:np.ndarray):
    return np.concatenate([np.diagonal(array[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-array.shape[0], array.shape[0])])


# Take an image and denoise it with the given filter.
def denoise_image(img, type="median", kernel_size=3, d=20, sigma_color=150, sigma_space=150):

    if type == "median":
        img = cv2.medianBlur(img, kernel_size)

    elif type == "gaussian":
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

    elif type == "bilateral":
        img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    else:
        print("Unrecognized Type!")

    return img