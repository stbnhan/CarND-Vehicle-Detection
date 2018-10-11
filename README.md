# **Vehicle Detection** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1_1]: ./output_images/Vehicles1.png
[image1_2]: ./output_images/Vehicles2.png
[image1_3]: ./output_images/Vehicles3.png
[image2_1]: ./output_images/NonVehicles1.png
[image2_2]: ./output_images/NonVehicles2.png
[image2_3]: ./output_images/NonVehicles3.png
[image3]: ./output_images/HOG_example.png
[image4_1]: ./output_images/test4_window_img.png
[image4_2]: ./output_images/test6_window_img.png
[image5_1]: ./output_images/test4_boxes.png
[image5_2]: ./output_images/test6_boxes.png
[image6_1]: ./output_images/test4_heatmap_final.png
[image6_2]: ./output_images/test6_heatmap_final.png
[image7_1]: ./output_images/test4_final.png
[image7_2]: ./output_images/test6_final.png
[video1]: ./P5_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  

#### 2. Provide source code.
This entire project is self-contained in one file: `vehicle_detection.py`  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 61 - 142 which contains all the feature extraction functions.  

First step was to load all the vehicle and non-vehicle image data set.  Here's an example of one of each of the classes:  

[Vehicle]
![alt text][image1_1]
![alt text][image1_2]
![alt text][image1_3]
  
[Non-Vehicle]
![alt text][image2_1]
![alt text][image2_2]
![alt text][image2_3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Final values used for hog parameters:  
```
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)  
cspace ='YCrC'
```

Here is an example using above parameters:  
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and got to a very high accuracy result. I was satisfied with its accuracy and moved on knowing future steps (like combination of hot box and threshold method would filter out false positives.  

[Color Feature Parameters]  

| Parameters     | Value          |  
|:--------------:|:--------------:|  
| Color Space    | `YCrCb`        |  
| spatial_size   | 32, 32         |  
| hist_bins      | 32             |  
| hist_range     | 0, 256         |  

[HOG Feature Parameters]  

| Parameters     | Value          |  
|:--------------:|:--------------:| 
| Orient         | 9              | 
| pix_per_cell   | 8              |
| cell_per_block | 2              |
| hog_channel    | `ALL`          |

[Feature Classifier Enable flag]  

| Parameters     | Value          |  
|:--------------:|:--------------:|
| spatial_feat   | True           |
| hist_feat      | True           |
| hog_feat       | True           |
  
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifiers are coded in lines mentioned above.  
I trained a linear SVM as it was the fastest and most efficient classifier.  
Then I used combination of Spatial Binning of Color, Histograms of Color, and Histogram of Oriented Gradients to use as my pipeline. Combination of resulted vectors are concatenated and normalized to prevent to make sure one feature's vector doesn't saturate the other features. By splitting the training data into 8:2 ratio sets to produce testing set, I was able to test this pipeline which resulted in 99.1% accuracy.  

Training process is coded in lines 165 - 235:
```
#!python
Training images loaded  
Training data saved  
15.46 Seconds to train SVC...  
Test Accuracy of SVC =  0.991  
My SVC predicts:      [0. 0. 1. 1. 0. 1. 1. 0. 1. 1.]  
For these 10 labels:  [0. 0. 1. 1. 0. 1. 1. 0. 1. 1.]  
0.00045 Seconds to predict 10 labels with SVC  
Total time:  596.41267 seconds  
```

After training is done, I saved the trained model and X_scaler for the future prediction process to load from.  
```
# Save trained model
joblib.dump(svc, 'svc.joblib')
# Load trained model
svc = joblib.load('svc.joblib')

# Save X_scaler
joblib.dump(X_scaler, 'X_scaler.joblib')
# Load X_scaler
X_scaler = joblib.load('X_scaler.joblib')
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To save computation efforts, I applied region of interest on where the sliding window will search for vehicles. Given that the cars don't fly around in the sky and camera is facing forward at a consistent angle, the sliding window will only focus on bottom half of each frame (also excluding the vehicle's hood). Then sliding window scale is defined to pick up the most vehicles with most accuracy. This code can be found in lines 296 - 319 for test images and lines 391 - 465 for video pipeline. 

![alt text][image4_1]
![alt text][image4_2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Like mentioned above, combination of feature extractions and normalizing provided good results.  
Here's an example:  

![alt text][image5_1]
![alt text][image5_2]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./P5_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I collected the positive detections in each frame of the video in a list. From this list, I created a heatmap and then applied thresholded to that map to identify and confirm vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Each blob corresponded to a vehicle, and overlapping assured the quality of the detection. Finally, I constructed bounding boxes to cover the area of each blob detected.  

Finally, I took the result of `scipy.ndimage.measurements.label()` of previous frame's heatmap to add confidence in the current heatmap. This is done by using a threshold of previous heatmap's mean value and adding to add confidence in the current frame while increasing `apply_threshold()` function. This adds another filtering process relative to previous frame.  

### Here are sample frames and their corresponding heatmaps:

![alt text][image6_1]
![alt text][image6_2]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7_1]
![alt text][image7_2]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The vehicles are well detected, but the bounding boxes are very jittery. It also does pretty well on not falsely detecting vehicles. The biggest struggle was to figure out the robust combination of feature extractions and parameters. First thing comes into my mind that my pipeline may not be able handle is the camera view. Each frame was assumed to be driving on a relatively straight or slightly curved lanes. This will expose blind spots. Another bigger challenge I see is that the computation speed is too slow to be done in real time. Given that the vehicle will need to react fast enough to avoid detected vehicles, it'll need a lot more robust and fast pipeline. I'll try deep learning neural network (such as Yolo) to fully utilize GPU to speed things up.