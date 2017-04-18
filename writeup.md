


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_car.png
[image3]: ./examples/hog_notcar.png
[image4]: ./examples/scaled_features.png
[image5]: ./examples/bin_spatial.png
[image6]: ./examples/histograms.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/heatmaps.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the IPython notebook into the function `extract_features`.

In order to be able to execute this step it is necessary to load all the vehicle and non-vehicles images. This is done in the second cell. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

We decided to use the same parameters as in the lesson, except for the number of orientations. This number is now equal to 6 which contributes to cut down the processing time. We grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

Various combinations of parameters were tried and it was decided to use those defined at cell 6, which in the case of the hog_features are very similar to those used in the lesson.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A rbf SVM (cell 15) was trained combining HOG, bin spatial and color features in order to achieve a good accuracy. On the other hand, the choice of values for the parameters C and gamma was based on previous works. The features are extracted in cell 11, and then scaled in cell 12. The result cen be seen below:

![alt text][image4]

The accuracy achieved by this classifier is 0.999149418769 (see cell 16).

As for the bin spatial features, we reduced the size of the original image from (64,64) to (16,16) as you can see below:

![alt text][image5]

Regarding the color features, we computed the histogram of each color channel with 64 bins and then we stacked these three feature vector into one feature vector. Here, you can see the histograms for each channel for color space `YCrCb`.

![alt text][image6]

It can be observed that for Cr and Cb channels the histogram of non-vehicle images has a lot more peaks than the histogram for vehicle images.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In order to go over the image we used the function `find_cars` defined at cell 18 and explained in the lesson because it allows us to only have to extract the Hog features once. 

The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

In this case, we use only one value for scale (1.5) but we could use several values to enhance the performace of our pipeline.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As said before we searched only on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/adricostas/CarND-Vehicle-Detection/blob/master/output2.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As seen before the classifier is not perfect. In some cases, it reports multiple overlapping instances of the same car or even reports cars where there are none. These are known as duplicates and false positives and we have tried to filter them out.

In order to do that, we recorded the positions of positive detections in each frame of the video.  From the positive detections we created a heatmap and then thresholded that map to identify vehicle positions.  We then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Afterwards we assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image8]

In the pipeline for processing the video we added a filter in order to achieve a smoother result. We consider the positions of positive detections during the last 5 frames.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As you can observe in the video, there exist false positives in some frames. I would be necessary to tweak the params of threshold and frames used to solve this issue. Moreover, the system could fail by having a lot of shadows or high contrast light. Not learned vehicles like trucks or bikes will probably fail, too.

To improve the system, these other measures could be taken into account:

* By further improving the input feature and parameters the system could gain in stability
* Using different scales the detection could be also improved. Vehicles in the vertical center of a video appear smaller because they are further away. On the other side, the closer a car, the bigger it appears. By using different window sizes (scales) the system could take account of this phenomenon.



```python

```
