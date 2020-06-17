# *Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Camera sample image"
[image2]: ./examples/center_train.jpg  "Center Camera train image"
[image3]: ./examples/left_train.jpg "left Camera train image"
[image4]: ./examples/right_train.jpg "right Camera train image"
[image5]: ./examples/center_flip.jpg "flipped image 1"
[image6]: ./examples/center_1_flip.jpg "flipped image 2"


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The Nvidia model is used as the base architecture. It's hidden layers are modified to minimize the loss and get the desired perfomance for this project. It was noted that the incoming images have mostly black lanes, edges of road and shoulders and need to be recognized for autonomously steer the car. 

#### 2. Attempts to reduce overfitting in the model

I took following measures to keep the model from overfitting :
1) Addition of Droput layer
2) Augmented data by flipping center image 
3) Use left and right camera images with slight changes in steering angles
4) Shuffle data randomly

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 51-52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
I used adam optimizer to train the model by minimazing mean square error.(model.py line 82)

#### 4. Appropriate training data
Initially I trained the model on  the sample data provided in the repository. The car runs fairly well  on the track except around the corners. So I collected more relevant data mostly around the corners and trained the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the Nvidia architecture recommneded in the lessons. I  modified the architecture slightly with the intention to make the model learn and recognize the edges of lanes, signs and shoulder edges. I used 20% of the data to be the validation data. Although The training data is satisfactorily large to avoid overfitting,further to avoid overfitting,reduce losses and get a better trained model, I used 50% dropout of parameters and increased the number of parameters in hidden layers.

The Nvidia architecture is already a well proven architecture, but the real challenge was to collect data which can give good maneuver performance around the turns and the edges of the road. So I iteratively collected the additional data and trained model until I see the desired performance around the turns.

At the end, the vehicle is able to drive autonomously mostly in the center of the road all around the track  without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-83) consisted of a convolution neural network with the following Keras layers and layer sizes :


| Keras Layer    	      |           Description    					| 
|:-----------------:    |:---------------------------------:| 
| Input         		    | 160x320x3 RGB image  							|
| Cropping2D    		    | 70x320x3 RGB image   							|
| Normalization 		    |  About zero       			  				|
| Convolution2D 5x5x36  | 2x2 stride, valid padding        	|
| RELU					        |												            |
| Convolution2D 5x5x48  | 2x2 stride, valid padding         |
| RELU					        |												            |
| Convolution2D 5x5x48  | 2x2 stride, valid padding         |
| RELU					        | 												          |
| Convolution2D 3x3x64  | valid padding, outputs            |
| RELU		         			|	            											|
| Convolution2D 3x3x64  | valid padding, outputs            |
| RELU			         		|			            									|
| Dropout 50%           |									            			|
| Flatten				        | output 2000							          |
| Fully connected		    | 2000 input, output 120 						|
| Fully connected		    | 120 input, output 50							|
| Fully connected		    | 120 input, output 50							|
| Fully connected		    | 50 input, output 1	  						|


#### 3. Creation of the Training Set & Training Process

I went through all the sample data images provided in the repository and trained the model. The sample images are sufficient to run the car and keep it on the track until a sharp turn appears. The response was good but performance had to be improved around the sharp truns.

![Center Camera sample image][image1]

Further I t recorded the vehicle recovering from the sides of the road back to center. I refrained from collecting redundant data as it sometimes detriorate the performance. 

![Center Camera train image][image2]
![left Camera train image][image3]
![right Camera train image][image4]

I flipped images and angles to augment data and generalize the learning. For example, here is an image that has then been flipped:

![flipped image 1][image6]
![flipped image 2][image7]


After the collection process, I had 28170 number of data points. I then preprocessed this data by cropping off the useless pixels. The data is then normalized about zero.

I then randomly shuffled the data set and used 20% of the data as validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The large and quality data set ensured to achieve desired performance level and training/validation loss < 2 %. epochs was set equal to 1.
