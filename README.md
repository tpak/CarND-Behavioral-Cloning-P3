# Project: Behavioral Cloning
## Autonomously drive a simulated mode car around a track - just keep it on the road!

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

##Overview

The objective of this project was to demonstrate the basic application of "behavioral cloning" as it applies to self driving car models. We trained a model and saved it. Then after moving the model to a new machine and running it as a server process able to accept incoming connections and images in "real time" we used the same model we had trained to issue corrections to the "car" such that the car was able to steer around the track on it's own. We only issued steering corrections at a fixed speed for this excercise. The real objective was learning how to re-use or "clone" a model. In the real world you might update the model in a car as new information is gleaned and understood and the model is improved. This excercise is the basics of doing that. 

##Behavioral Cloning Project

For this project we did the following:

* Used the simulator to collect data of good driving behavior
	* the collected data is not included in the repository due to space constraints
	* as a "non-gamer" collecting data was a challenge, I eventually enlisted my son to help collect several laps worth of simulated driving data

* Built, a convolutional neural network (CNN) in Keras on top of Tensorflow that predicts steering angles from images
	* Train and validate the model with a training and validation set

* Test that the model successfully drives the car around track #1 without leaving the road
* 	this involved saving the model and moving it to a computer running the simulator in autonomous mode and seeing of the car could complete a lap without running off the track or crossing onto a hazard such as the yellow lines, the river (a popular option in the early version of my models), or the dirt
* Summarize the results with a written report (this doc)
 

## Files Submitted 

#### 1. My project includes the following files and can be used to run the simulator in autonomous mode: 

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (provided by Udacity)
* model.zip (model.h5) containing a trained convolution neural network 
	* my model was too large to upload to gitbub so I submitted a zip version
	* the model seemed to expand by a facor of 10x with one change - this should be explored more to understand why
* README.md -- this file
* video.mp4 -- a video of my "car" successfully making a lap around track 1 from the simulator


#### 2. Submission includes functional code
Using the Udacity provided simulator and the originally provided drive.py file, the car can be driven autonomously around the track (#1) by executing the following commands.

```sh
unzip model.zip
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code I used for training and saving the CNN. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My initial approach was to use a very simple model just to get the image processing pipeline working and then progress as sugessted in the coursework to an nVidia model clone. Regardless of the model, I stuggled to keep the car in the lanes until I added the fully connected layer with 1164 neurons (line103:  "model.add(Dense(1164))" ). This had the unfortunate side effect of bloating the model file from ~11MB to ~120MB which is not trivial for embeded systems. This would need deeper exploration to augment my understanding of how CNN's work and how they consume memory. I tried numerous variations of optimizers, droputs, and layering but this is the one that proved most reliable.

My model summary is as follows:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          9834636     flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================
Total params: 10,088,055
Trainable params: 10,088,055
Non-trainable params: 0
____________________________________________________________________________________________________

```

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 43-48). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

I found that the model is currently very sensitive to variations in the quality of the training data. per nVidia this could likely be improved by adding various transforms to the collected images and also various shading techniques. 

Doing this for real I would add various automated test harnesses that would run an exhustive list of variotions and track the results looking for the best fit of model parameters. Doing this by hand was error prone.

#### 3. Model parameter tuning

The submitted model usea an adam optimizer. I atempted using Adamax and also playing with the learning rate in both models with no success. The stock adam optimizer provided by the Keras framework worked the most reliably.

#### 4. Appropriate training data

I drove several laps on my own to create training data. I'm not good at video games was the conclusion of the excercise so I had my son drive several laps and collect data. I also had him do some intentional "set ups" whereby he aimed the car at various hazards and corrected for them.

### Conclusion

There are a lot af variables that go into any model and training it. The quality of the data seems to be paramount as does properly augmenting it. If I could start over I would spend a lot more time up front collecting good quality training data and also additional time on augmentation methods beyond the simple flip and rotate that I did. I would add shadows, shear, and angular rotation. I would also spend more time on determining the optimal steering correction factor for the verious cameras. This seems like a really great task for a test harness in a CI/CD and test automation pipeline.

The other thing that this project highlighted for me is the need for automated test harnesses in the SDC space. Harnesses should be created that automatically vary the training data, the model depth, and the parameters. The harness should collect and plot all of the test runs anytime any change is made. For the project purpose as one person I did not have the time to do this but if I was leading a team of engineers I would want a fully formed team of T shaped skills working on these types of projects. 



