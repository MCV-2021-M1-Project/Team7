import os
import get_images_and_labels
import evaluation as eval
import argparse
import sys
from utils import denoise_image
from local_descriptors import calc_local_descs_matches, get_map_for_local_descs
from rotations import get_angles_and_rotations


# Parser to get arguments from command line
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-all", "--all", action="store_true",
        help = "See results for all possible combinations")

    parser.add_argument(
        "-adc", "--all_desc_combs", action="store_true",
        help = "See results for all possible descriptor combinations")

    parser.add_argument(
        "-kp", "--keypoint_size", default=1000, type=int,
        help = "Keypoint size for local descriptors"
    )

    parser.add_argument(
        "-mt", "--match_threshold", default=50, type=int,
        help = "Match threshold for local descriptors. If given query set image\
                doesn't have a number of matches higher than the threshold for \
                any museum image, it is labeled as [-1]"
    )

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
        "-l", "--level", default=4, type=int,
        help = "Determine the level of the image split. Image is split into 2^(level-1) equal pieces \
                and histograms will be calculated seperately for each piece and will be concatenated\
                into a 1d array.")

    parser.add_argument(
        "-r", "--dataset_paths", default=os.getcwd(),
        help = "Path to the folder where image datasets are. \
                Each dataset should be in a folder in this path")
        
    parser.add_argument(
        "-q", "--query_set", default="qsd1_w5",
        help = "Which query set to use")

    parser.add_argument(
        "-cs", "--color_space",  default="LAB",
        help = "Histogram calculation method: RGB, HSB, LAB, YCRCB")

    parser.add_argument(
        "-dm", "--desc_method", default="ORB", 
        help = "Which descriptors are going to be used? If multiple, \
                should be given with commas and there shouldn't be \
                space between commas: 1d, 3d, DCT, LBP, text, ORB, SIFT, DOG, HP")

    parser.add_argument(
        "-cdm", "--color_distance_metric", default="hellinger",
        help = "Similarity measure for color based histograms to compare images: \
                cosine, manhattan, euclidean, intersect, kl_div, hellinger, chisqr, correlation")

    parser.add_argument(
        "-tdm", "--text_distance_metric", default="jaccard",
        help = "Similarity measure to compare texts: \
                cosine_text, jaccard, hamming, levenshtein")

    parser.add_argument(
        "-tudm", "--texture_distance_metric", default="correlation",
        help = "Similarity measure for texture base histograms to compare images: \
                cosine, manhattan, euclidean, intersect, kl_div, hellinger, chisqr, correlation")

    parser.add_argument(
        "-b", "--bins",default="8", type=int, 
        help = "Number of bins to use for histograms.")

    parser.add_argument(
        "-k","--k", default="10", type=int,
        help = "Mean average precision for top-K results")

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':

    args = parse_args()
    print("Passed arguments are:", args)

    cur_path = args.dataset_paths

    desc_methods = args.desc_method.split(",")

    if args.mode == "test":
        args.eval_masks = False


    # If -all command is given evaluates or tests all sets of that mode
    # for every possible combination of color space, distance metrics
    # and desc_methods.
    if args.all:
        eval.evaluate_all(args.bins, args.pickle, cur_path, args.level, args.eval_masks, args.mode)

    elif args.all_desc_combs:
        eval.evaluate_combs_all(args.pickle, cur_path, args.eval_masks, args.mode)

    else:
        museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, args.query_set)

        if int(args.query_set[-1]) >= 3:

            query_set_imgs = [denoise_image(img) for img in query_set_imgs]

        if args.query_set[-1] == "5":
            query_set_imgs, angles = get_angles_and_rotations(query_set_imgs, args.query_set, cur_path)

        # Don't evaluate mask if query set is a test set.
        if args.query_set[3] == "2" or args.query_set[-1] > "4":

            if args.mode == "eval":
                _, query_set_imgs, boxes = eval.remove_background_and_eval(query_set_imgs, cur_path, args.query_set, 
                                                                           args.eval_masks)  
            else:
                _, query_set_imgs, boxes = eval.remove_background_and_eval(query_set_imgs, cur_path, args.query_set, False) 

        if args.query_set[-1] == "5":

            if args.mode == "eval":
                box_acc, angle_acc, frame_res = eval.get_frames_and_eval(query_set_imgs, args.query_set, cur_path,
                                                                         angles, boxes, args.mode)
                print("Box accuracy:", box_acc, "Angle acc:", angle_acc)

            else:
                frame_res = eval.get_frames_and_eval(query_set_imgs, args.query_set, cur_path,
                                                     angles, boxes, args.mode)

        if any(x in ["ORB", "SIFT", "DOG", "HP"] for x in desc_methods):
            for desc_method in desc_methods:
                img_matches = calc_local_descs_matches(museum_imgs, query_set_imgs, desc_method, "BF", args.keypoint_size)

                if args.mode == "eval":
                    _, mAP = get_map_for_local_descs(img_matches, desc_method, args.query_set, cur_path, 
                                                     args.match_threshold, args.pickle, args.mode, args.k)
                    print("For Keypoint Size:", args.keypoint_size, "and match threshold:", args.match_threshold, "mAP is", mAP)

                else:
                    get_map_for_local_descs(img_matches, desc_method, args.query_set, cur_path, 
                                            args.match_threshold, args.pickle, args.mode, args.k)

        else:
        # If -all command is not given evaluate or test the given query set.
            eval.evaluate_query_set(query_set_imgs, museum_imgs, cur_path, args.level, desc_methods, args.mode, args.query_set,  
                                    args.color_space, args.color_distance_metric, args.text_distance_metric, args.texture_distance_metric,
                                    args.k, args.bins, args.pickle)


