
## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines 14 through 31 of the file called helper_funcs.py).

I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:

<table><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/image0000.png"></td><td><img src="../ztemp/temp/image6.png"></td><td></table>

I then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.

Here is an example using the YCrCb color space and HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):

<table><tr><td>ch1 car</td><td>ch2 car</td><td>ch3 car</td><td>ch1 HOG car</td><td>ch2 HOG car</td><td>ch3 HOG car</td></tr><tr><td><img src="../ztemp/temp/ch1 car.png"></td><td><img src="../ztemp/temp/ch2 car.png"></td><td><img src="../ztemp/temp/ch3 car.png"></td><td><img src="../ztemp/temp/ch1 HOG car.png"></td><td><img src="../ztemp/temp/ch2 HOG car.png"></td><td><img src="../ztemp/temp/ch3 HOG car.png"></td><tr><td>ch1 not car</td><td>ch2 not car</td><td>ch3 not car</td><td>ch1 HOG not car</td><td>ch2 HOG not car</td><td>ch3 HOG car</td></tr><tr><td><img src="../ztemp/temp/ch1 Not car.png"></td><td><img src="../ztemp/temp/ch2 Not car.png"></td><td><img src="../ztemp/temp/ch3 Not car.png"></td><td><img src="../ztemp/temp/ch1 HOG Not car.png"></td><td><img src="../ztemp/temp/ch2 HOG Not car.png"></td><td><img src="../ztemp/temp/ch3 HOG Not car.png"></td></table>

### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and ended up using the following:

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

It seems to do the job at a relatively reasonable time. For instance training using spatial_size and hist_bins uing 32 the training took a very long time with minor improvement.

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG and color features. The code is in the second cell of find_car.ipynb. I started with stacking car and notcar features and then scaled them I stacked two other arrays with 1 and zero values with the same size as car and notcar arrays to represent y values. I then splited them into train and test sets and then fitted them with with skitlearn svm function. 

## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

In the seventh cell of find_car.ipynb, under find_car function slinding window is implemented. Each window is 64 pixels and each cell is 8. windows slide from left to right and top to bottom two cells at a time. I tried different scales and 1.5 seems to do a reasonable job for detecting the cars at a reasonable distance range. as per the steps or over lap 2 cells step seem to do a good job in giving enough number of over lap of the car for the heatmap to detect the car from false positive and detect the whole size of the car for drawing the bounding boxes.

### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale(scale of 1.5 provides sufficient detection for cars that are not too far and within reasonable range) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

<table><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/example1.jpg"></td><td><img src="../ztemp/temp/example2.jpg"></td><tr><td><img src="../ztemp/temp/example3.jpg"></td><td></table>

## Video Implementation

## 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

here is a link to my video: 'https://github.com/ainode/Vehicle_Detection_and_Tracking_Project5/main_files/processed.mp4'

## 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:

Here are seven frames and their corresponding heatmaps:

<table><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame0.jpg"></td><td><img src="../ztemp/temp/heatmap0.jpg"></td><td><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame1.jpg"></td><td><img src="../ztemp/temp/heatmap1.jpg"></td><td><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame2.jpg"></td><td><img src="../ztemp/temp/heatmap2.jpg"></td><td><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame3.jpg"></td><td><img src="../ztemp/temp/heatmap3.jpg"></td><td><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame4.jpg"></td><td><img src="../ztemp/temp/heatmap4.jpg"></td><td><tr><td>car</td><td>Not-car</td></tr><tr><td><img src="../ztemp/temp/boxed_frame5.jpg"></td><td><img src="../ztemp/temp/heatmap5.jpg"></td><td><tr><td><img src="../ztemp/temp/boxed_frame6.jpg"></td><td><img src="../ztemp/temp/heatmap6.jpg"></td><td></table>

### Here is the output of scipy.ndimage.measurements.label() on the integrated heatmap from all six frames:

<img src="../ztemp/temp/label_img.jpg">

## Here the resulting bounding boxes are drawn onto the last frame in the series:

<img src="../ztemp/temp/draw_img.jpg">

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

-This project was a fun project and there were not many issues and problems implementing it. I would say that the hardware constraints that I had, would be the main problem in training the classifier. It took a long time to train and given the time limit, I did not get the chance to do as many experimentation as I wanted to, with regards to training. I still did quite a few.
The parameters chosen in the lessons seemed to be relatively fine for this project.

-The pipeline might fail when processing two or more cars driving close (it will probably put them all in one box). It might also fail when the lighting and scenery changes drastically.

-as per the last question, I would collect and integrate many more varied samples and would experiment more with different parameters and classifiers.
 


```python

```
