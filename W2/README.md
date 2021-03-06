## Week 2

Block based histograms are introduced this week. Images are split into equal parts and each part's histogram is calculated seperately, and as a final step all parts are concatenated into a 1D array. 

Also this week's query images have text in them and query set 2 images may have more than 1 painting in a single image. We created morphological operators to detect text boxes and chaned our background removal code to detect more than 1 painting in an image.

<b> Usage: </b> <br>

```

python main.py [-all all] [-p pickle] [-m mode] [-em eval_masks] [-r dataset_path] [-q query_set] [-cs color_space] 
               [hm hist_method] [-dm distance_method] [-b bins] [-k k] [-l level]
                    

Arguments: 


all, -all:   If this argument is given code is run for all possible combinations of color spaces, distance metrics, 
             histogram methods and k sizes [1, 5, 10] for the given mode. If mode is "eval", then a csv with the 
             results of each combination is created.
Default: False


pickle, -p:  Creates pickle file/files with top 10 results for each image in the query set
Default: True


mode, -mode:  Chooses which to do: "eval" or "test". If eval is given prints the mAP result for the given parameters.
Default: eval
Arguments: eval, test


eval_masks, -em: If given evaluates the background removal results.
Default: True
                  

dataset_path, -r: If given evaluates the background removal results.
Default: os.getcwd()
                  
 
query_set, -q: Name of the query set we want to use. If -all is given this argument doesn't matter because the code
               will run for all query sets in the given mode.
Default: qsd1_w2
Arguments: qsd1_w1, qsd2_w2, qst_w1, qst_w2
                  
                  
color_space, -cs:  Name of the color space we want to use. If -all is given this argument doesn't matter because 
                   the code will run for all color spaces in the given mode.
Default: YCRCB
Arguments: RGB, HSB, LAB, YCRCB 


hist_method, -hm: Name of the query set we want to use. If -all is given this argument doesn't matter because
                  the code will run for all histogram methods in the given mode.   
Default: 3d
Arguments: 1d, 3d


dm, -distance_metric: Name of the distance metric we want to use. If -all is given this argument doesn't matter
                      because the code will run for all distance metrics in the given mode.
Default: hellinger
Arguments: cosine, manhattan, euclidean, intersect, kl_div, bhattacharyya, hellinger, chisqr, correlation
                  
                  
bins, -b: Size of the bins.
Default: 8  
Arguments: Any integer value                


k, -k: How many of the top results for mean AP we are going to get. 
Default: 10
Arguments: Any integer value  


level, -l:   This argument determines the level of image split. Image is split into 2^(level-1) parts in the x and y axes. 
             In total, we have 2*(2^(level-1)) same sized patches of an image. Each patch's histogram is calculated 
             seperately.
Default: 4
Arguments: Any integer value  
```
