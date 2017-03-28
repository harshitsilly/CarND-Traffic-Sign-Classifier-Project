#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/test0.png "Traffic Sign 1"
[image5]: ./examples/test2.png "Traffic Sign 2"
[image6]: ./examples/test4.png "Traffic Sign 3"
[image7]: ./examples/test6.png "Traffic Sign 4"
[image8]: ./examples/test7.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/harshitsilly/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1) 
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of data available for the unique class

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because  it will reduce the length of the input instead of 3 channels there will be 
only  1 channel. Input instead of 3x will be only x.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data with zero mean so that all the input variables have the same treatment and the coefficients of a model are not scaled with respect to the units of the inputs.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,2x2 size, outputs 14x14x6 			|
| Convolution 5x5      	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,2x2 size,  outputs 5x5x16 			|
| Droput              	| probability = .75			                    |
| Flatten   	      	| outputs 400                        			|
| Fully connected		| Input = 400,Output = 120      				|
| RELU					|												|
| Droput              	| probability = .75			                    |
| Fully connected		| Input = 120,Output = 84         				|
| RELU					|												|
| Droput              	| probability = .75			                    |
| Fully connected		| output = 43                                   |
| Softmax				|										    	|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an adam optimizer, batch size is 100 , epochs is 30 , learning rate is 0.001, mean = 0 and sigma = 0.1 for generating initail weights

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.956
* test set accuracy of 0.940


If a well known architecture was chosen:
* What architecture was chosen?
- I have used the Lenet architecture and modify it by adding dropoyt layers.
* Why did you believe it would be relevant to the traffic sign application?
- Traffic sign application needs association of inputs as it is a image processing task. If we simpy use pooling with larger size we will loose the association and the information. Lenet uses CNN to scale down the input without losing much information thus our diminish input is almost remain intact with the information.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Training accuracy will be almost 100% as the model is trained on train data but as we can see the validation accuracy and test accuracy differ by only 1% and are around 95% so we can say our model is performing great. We can further improve it by augumenting the data to make all the input data equal for all the classes.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because it is diminish somewhat like a picture taken in low light condition.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14 cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield          		| Yield       									| 
| no passing     		| No passing									|
| General Caution		| General Caution								|
| 70 km/h	      		| Bumpy Road					 				|
| Turn right			| No entry       							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Comparing the accuracy to the test set will be somewhat harsh as we are testing on 5 images not the large datasets of all the classes as the test set

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

