import argparse
import cv2
import os
import sys
from descriptor import get_descriptor
from get_images_and_labels import get_museum_dataset

from sklearn.cluster import KMeans
import pandas as pd


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", default=os.getcwd(),
        help = "Give filepath for museum dataset")

    parser.add_argument(
        "-dm", "--desc_method", default="3d",
        help = "Which descriptor method to extract features")

    parser.add_argument(
        "-cs", "--color_space", default="RGB",
        help = "Which color space to use")

    parser.add_argument(
        "-cn", "--cluster_number", default=10, type=int,
        help = "Number of clusters")

    args = parser.parse_args(args)
    return args


def cluster_imgs(museum_imgs, desc_method, cur_path, clr_spc="RGB", cluster_num=10):

    museum_features = get_descriptor("BBDD", cur_path, museum_imgs, 1, desc_method, clr_spc, 8)
    kmeans = KMeans(cluster_num, n_init=20, max_iter=1000)
    kmeans.fit(museum_features)

    preds = kmeans.predict(museum_features)
    print("Found Clusters!")
    img_clusters = pd.DataFrame()
    
    img_clusters["imgs"] = [img for img in museum_imgs]
    img_clusters["clusters"] = preds
    
    print("Writing images to their respective cluster folder!")
    for cluster in set(img_clusters["clusters"].values):

        fp = desc_method + "_clusters"
        imgs_of_clusters = img_clusters[img_clusters["clusters"] == cluster]

        if not os.path.exists(fp):
            os.mkdir(fp)
            
        fp =  fp + "//" + str(cluster)
        if not os.path.exists(fp):
            os.mkdir(fp)

        for i, _ in enumerate(imgs_of_clusters["imgs"].values):
            img_meta = imgs_of_clusters.iloc[[i]]
            cv2.imwrite(fp + "//" + "bbdd_" + str(img_meta.index[0]).zfill(5) + ".jpg", img_meta["imgs"].values[0])



if __name__ == '__main__':

    args = parse_args()
    print("Passed arguments are:", args)

    museum_imgs = get_museum_dataset(os.getcwd())
    cluster_imgs(museum_imgs, args.desc_method, args.path, args.color_space, args.cluster_number)