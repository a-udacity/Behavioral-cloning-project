#**Behavioral-cloning-project** 
TODO: Flipped images
ACTUAL images


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Overview

The objective of this project is to clone human driving behavior using a Deep Neural Network.
In order to achieve this, we are going to use a simple Car Simulator that captures the images from 3 cameras (center, left, right) along with steering angles.
Then those recorded images and the corresponding sheering angles are used to train the 5 layers neural network.
Trained model was tested on two tracks, namely track1 and track2. The two following you videos show the performance of the final model track 1 and track 2 in Autonomous Mode.


[![Evaluation Track](sample/track1.jpg)](https://youtu.be/Xplj9rNayR8)
[![Track 2](sample/track2.jpg)](https://youtu.be/zxCuKIyuFzo)

* model.py - The script used to create and train the model.
* drive.py - The script to drive the car. I have modified to work correctly.
* model.json - The model json filexxxx
* model.h5 - The trained model.
* ./data/drive_log.csv  - Combined log of simulated data along with sample data provided by udacity.
* model.ipynb - Model Jupyter notebook. Note this also contains the generated images for visualization.....

[//]: # (Image References)
Here's our logo (hover to see the title text):

![image 1](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/left.png "TODO Normal center camera images")
![image 2](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/left.png "TODO Normal left camera images")
![image 3](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/left.png "TODO Normal right camera images")
![image 4](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/center.png "Pre-process center camera images")
![image 5](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/left.png "Pre-process right camera images")
![image 6](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/right.png "Pre-process left center camera images")
![image 7](https://raw.githubusercontent.com/ahararwala/Behavioral-cloning-project/master/sample/left.png "TODO Pre-process flipped images")


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* sample/*.png - visualization of images folder

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
