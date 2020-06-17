import numpy as np
import csv
import cv2

from math import floor
import random
import sklearn

#generator function to get samples in batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # infinite loop
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
              for i in [0,1,2]:
                name = './data/IMG/'+batch_sample[i].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if (i==0):                 
                 images.append(center_image)
                 angles.append(center_angle)
                 images.append(cv2.flip(center_image,1))
                 angles.append(-1*center_angle)
                else:
                 images.append(center_image)
                 angles.append(center_angle*(1+0.1*(-2*i+3))) # steering angle modification based on camera location
                 images.append(cv2.flip(center_image,1))
                 angles.append(-1*center_angle*(1+0.1*(-2*i+3)))
            X_train=np.array(images)
            y_train=np.array(angles)           
            yield sklearn.utils.shuffle(X_train, y_train)
           

samples=[]
with open('./data/driving_log.csv') as csvfile:
        read=csv.reader(csvfile)
        for line in read:
                samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

batch_size=32

# Samples for training and validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Keras Neural network Model based on Nvidia architecture
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

#initialliation
model=Sequential()
#Cropping and normalization
model.add(Cropping2D(cropping=((65,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255 - 0.5))

#Hidden layers
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))

#Fully connected layers
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Adam optimiser is used to minimize mean sqr error
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = floor(len(train_samples)/batch_size), validation_data=validation_generator,validation_steps=floor(len(validation_samples)/batch_size),epochs=1, verbose=1)

model.save('model.h5')
