import os
import get_images_and_labels
import evaluation as eval
import argparse
import sys
import utils


# Parser to get arguments from command line
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
        "-l", "--level", default=4, type=int,
        help = "Determine the level of the image split. Image is split into 2^(level-1) equal pieces \
                and histograms will be calculated seperately for each piece and will be concatenated\
                into a 1d array.")

    parser.add_argument(
        "-r", "--dataset_paths", default=os.getcwd(),
        help = "Path to the folder where image datasets are. \
                Each dataset should be in a folder in this path")
        
    parser.add_argument(
        "-q", "--query_set", default="qsd1_w2",
        help = "Which query set to use: qsd1_w2, qsd2_w1, qsd2_w2, qst1_w2, qst2_w2")

    parser.add_argument(
        "-cs", "--color_space",  default="YCRCB",
        help = "Histogram calculation method: RGB, HSB, LAB, YCRCB")

    parser.add_argument(
        "-hm", "--hist_method", default="3d",
        help = "Histogram calculation method: 1d, 3d")

    parser.add_argument(
        "-dm", "--distance_metric", default="hellinger",
        help = "Similarity measure to compare images: \
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

    # If -all command is given evaluates or tests all sets of that mode
    # for every possible combination of color space, distance metrics
    # and hist_methods.
    if args.all:
    
        if args.mode == "eval":
            eval.evaluate_all(args.bins, args.pickle, cur_path, args.level, args.eval_masks)
        else:
            eval.test_all(args.bins, args.pickle, cur_path, args.level)

    # If -all command is not given evaluate or test the given query set.
    else:

        print("### Getting Images ###")
        museum_imgs = get_images_and_labels.get_museum_dataset(cur_path)
        query_set_imgs = get_images_and_labels.get_query_set_images(cur_path, args.query_set)

        # Don't evaluate mask if query set is a test set.
        if args.query_set == "qsd2_w2":
            query_set_imgs = utils.remove_background(query_set_imgs, cur_path, args.query_set, args.eval_masks)  

        elif args.query_set == "qst2_w2":
            query_set_imgs = utils.remove_background(query_set_imgs, cur_path, args.query_set, False) 


        if args.mode == "eval":
            eval.evaluate_query_set(query_set_imgs, museum_imgs, cur_path, args.level, args.query_set, args.hist_method, 
                                    args.color_space, args.distance_metric, args.k, args.bins, args.pickle)
        else:
            eval.test_query_set(query_set_imgs, museum_imgs, cur_path, args.level, args.query_set, args.hist_method, 
                                args.color_space, args.distance_metric, args.k, args.bins, args.pickle)


