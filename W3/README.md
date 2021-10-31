## Week 3

Texture based and text based methods are included this week besides the color based histograms.

<b> Usage: </b> <br>

```

python main.py [-all all] [-p pickle] [-m mode] [-em eval_masks] [-r dataset_path] [-q query_set] [-cs color_space] 
               [hm hist_method] [-dm distance_method] [-b bins] [-k k] [-l level]
                    

Arguments: 


all, -all:   If this argument is given code, is run for all possible descriptor methods for the given mode. 
             Descriptors are run seperately and are not combined. If you want to see the all possible descriptor
             combinations use all_desc_combs argument. If mode is "eval", then a csv with the results of each 
             combination is created.
Default: False


all_desc_combs, -adc:   If this argument is given, code is run for all possible combinations of descriptor methods.
                        When this argument is True, we don't look at the descriptor methods argument since the code
                        will run for all combinations regardless of the given desciptors. If mode is "eval", then a 
                        csv with the results of each combination is created.                     
Default: False


pickle, -p:  Creates pickle file/files with top 10 results for each image in the query set
Default: True


mode, -mode:  Chooses which to do: "eval" or "test". If eval is given prints the mAP result for the given parameters.
Default: eval
Arguments: eval, test


eval_masks, -em: If given, evaluates the background removal results.
Default: True
                  

dataset_path, -r: If given, evaluates the background removal results.
Default: os.getcwd()
                  
 
query_set, -q: Name of the query set we want to use. If -all is given this argument doesn't matter because the code
               will run for all query sets in the given mode.
Default: qsd1_w1
                  
                  
color_space, -cs:  Name of the color space we want to use. If -all is given this argument doesn't matter because 
                   the code will run for all color spaces in the given mode.
Default: LAB
Arguments: RGB, HSB, LAB, YCRCB 


desc_method, -dm: Which descriptors are going to be used? If multiple, should be given with commas and
                  there shouldn't be space between commas If -all_desc_methods or -all is given this 
                  argument doesn't matter because the code will run for all descriptor methods in the given mode.   
Default: 3d
Arguments: At least one from [1d, 3d, DCT, LBP, text]


color_distance_metric, -cdm: Name of the distance metric we want to use for color based histograms. 
                             If -all_desc_methods or -all is given this argument doesn't matter because 
                             the code will run for all distance metrics in the given mode.
Default: hellinger
Arguments: cosine, manhattan, euclidean, intersect, kl_div, bhattacharyya, hellinger, chisqr, correlation


text_distance_metric, -tdm:  Name of the distance metric we want to use for text based descriptors. 
                             If -all_desc_methods or -all is given this argument doesn't matter because 
                             the code will run for all distance metrics in the given mode.
Default: jaccard
Arguments: cosine_text, jaccard, hamming, levenshtein


texture_distance_metric, -tudm: Name of the distance metric we want to use for texture based descriptors. 
                                If -all_desc_methods or -all is given this argument doesn't matter because 
                                the code will run for all distance metrics in the given mode.
Default: cosine
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

