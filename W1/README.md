## Week 1

This week, we are using 1D and 3D histograms for image retrieval. Images that have background are processed to remove their backgrounds.

<b> Usage: </b> <br>

```

python main.py [-all all] [-p pickle] [-m mode] [-em eval_masks] [-r dataset_path] [-q query_set] [-cs color_space] 
               [hm hist_method] [-dm distance_method] [-b bins] [-k k]
                    

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
Default: qsd1_w1
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
```


