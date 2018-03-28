# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_data/bars.png "Visualization"
[image2]: ./writeup_data/new_image.png "Sampling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/image_1.png "Traffic Sign 1"
[image5]: ./new_images/image_2.png "Traffic Sign 2"
[image6]: ./new_images/image_3.png "Traffic Sign 3"
[image7]: ./new_images/image_4.png "Traffic Sign 4"
[image8]: ./new_images/image_5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Orionxx/udacity_carND_Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples per class are given in the data. Note that the data is not very balanced between classes.

![Graphs showing Examples per class][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to generate new images until there are the same amount of training samples per class.

The new images are generated from the existing samples per class using a random perspective transform.

Here is an example for a new image:

![New Image][image2]

As a last step, I normalized the image data because we get better results with if the image data has zero mean and small variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x64 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x128	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 6x6x128  				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x256 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x512 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 6x6x1024	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 3x3x1024 				|
| Avg pooling 3x3	    | 1x1 stride,  outputs 1x1x1024 				|
| Flatten		        | output 1024      								|
| Fully connected		| output 512        							|
| RELU					|												|
| Dropout 0.5			|												|
| Fully connected		| output 256        							|
| RELU					|												|
| Dropout 0.5			|												|
| Fully connected		| output 43         							|
| Softmax				|            									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size 128.
I trained two times 30 Epochs with learning rate 0.0001.
For the second 30 epochs the additional images were newly generated, so the complete amount of images seen by the model is higher. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.97 
* test set accuracy of 0.96

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose an architecture i developed at my work for a similar image classification task and adjusted the size of the layers to work with the image shape and the number of classes of the traffic sign task.

The general approach is to generate first more depth and information while shrinking the image size with convolutional layers and max pooling, then reduce the depth with fully connected layers to the desired number of classes.
During the reduce part dropout is used to prevent overfitting.

The result of 96% test set accuracy is an evidence that the model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are cropped to the signs and resized to the same shape the training data has.

The ice/snow image could be difficult to recognize because it is covered by snow so that the red triangle is barely visisble.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ice/Snow         		| Bicycles     									| 
| Wild animals 			| Wild animals									|
| Roundabout			| Roundabout									|
| 50 km/h	      		| 50 km/h   					 				|
| Stop Sign 			| Stop sign         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96%. Only the very difficult ice/snow image is not correctly guessed.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all images except for the Snow/Ice image the model is extremly sure with its prediction (100% after softmax). 

Results for the Snow/Ice sign:

| Softmax            	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Bicycles   									| 
| 0.000035 				| Wild animals									|
| 0.0000023 			| Priority										|
| 0.000000046  			| No vehicles  					 				|
| 0.000000043		    | Roundabout           							|

The top 2 predictions Bicycles and Wild animals are very similar to the Snow/Ice sign but the correct result is not in the top 5.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The output of the layer 1 feature maps shows that different feature maps show different characteristica of the original image. For each edge oft the image, there is at least one feature map which shows high activation at this particular edge. Feature map 3 for example shows high activation near the upper left edge. There are also feature maps that show high activation on different aspects of the image. Feature map 6 for example has high activation in the inner white parts of the image whereas feature map 10 has high activation in the inner black parts of the image.
This combined information can be used by the higher layers in the model to activate on complex features.
