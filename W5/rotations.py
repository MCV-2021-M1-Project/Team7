import cv2
import numpy as np
from pyclustering.cluster.kmedians import kmedians as kmed


def rotate_point(img, point, angle):
    
    points = [point, ]
    
    (h, w) = img.shape[:2]
    origin = (w//2, h//2)
    
    M_inv = cv2.getRotationMatrix2D(origin, angle, 1.0)
    ones = np.ones(shape=(len(points), 1))

    points_ones = np.hstack([points, ones])
    transformed_points = M_inv.dot(points_ones.T).T
    
    return [int(round(p)) for p in transformed_points[0]]
    
    
def rotate_image(img, angle=0, mask=False):

    tmpImg = img.copy()
    
    (h, w) = img.shape[:2]
    origin = (w//2, h//2)
    
    mat = cv2.getRotationMatrix2D(origin, angle, 1.0)
    
    return cv2.warpAffine(
        tmpImg, mat, (w, h), flags=cv2.INTER_NEAREST if mask else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



def extract_angle(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = cv2.Canny(img, 50, 200)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 15, None, 10, img.shape[0] / 10)
    result_lines = np.zeros_like(img)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(result_lines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

            
    contours = cv2.findContours(result_lines.astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_copy = np.zeros_like(result_lines)
    cv2.drawContours(mask_copy, contours[0], -1, (255,255,255), thickness = 4)

    
    linesP = cv2.HoughLinesP(mask_copy, 1, np.pi / 180, 15, None, 50, 100)
    topaint2 = np.zeros_like(mask_copy)
    angles = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            v1 = (l[2] - l[0], l[3] - l[1])
            v1 = v1 / np.linalg.norm(v1)
            v2 = (1, 0)

            rads = np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])
            grades = 180*rads/np.pi
            angles.append(grades)
        
    angles = np.array(angles)
    
    max_, max_val = -90, 0
    current = -90
    eps = 5 # epsilon
    step = 0.1
    for i in range(int(90 / step)):
        current_count = np.where(((np.array(angles) > current - eps) & (np.array(angles) < current + eps)) | 
                                 ((np.array(angles) > current + 90 - eps) & (np.array(angles) < current + 90 + eps)))
        count = len(current_count[0])
        if count > max_val:
            max_val = count
            max_ = current
        current += step

        
    angles = [a for a in angles if (a > max_-eps and a < max_+eps) or 
                                   (a > max_-eps+90 and a < max_+eps+90)]
        
        
    # We find the median of the two clusters, initialized on the extreme values of the window
    kmedians_instance = kmed(np.array([(a,0) for a in angles]), [ [max_, 0.0], [max_+90,0.0] ]);
    kmedians_instance.process();
    kmedians_instance.get_clusters();
    
    # We return the minimum angle (in absolute value), which is the one we are looking for
    angles = [m[0] for m in kmedians_instance.get_medians()]
    angle = angles[np.argmin(np.abs(angles))]
    if angle < 0:
        return 180.0 + angle
    return angle


def get_angles_and_rotations(query_images, query_set, cur_path):
    
    angles = [extract_angle(img) for img in query_images]
    
    rot_angles = []
    for angle in angles:
        if angle > 90:
            rot_angles.append(angle - 180.0)
        else: 
            rot_angles.append(angle)

    rotated_imgs = [rotate_image(img, -angle) for img, angle in zip(query_images, rot_angles)]
    
    return rotated_imgs, angles