import tensorflow as tf
import argparse
import csv
import cv2
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

print("Using TensorFlow version:", tf.__version__)
print("Using Keras version:", keras.__version__)

def read_csv_file(csv_dir, header=False, correction=0.0):
    print("using data from: ", csv_dir)
    lines = []
    image_list = []
    measurements = []

    csv_file = csv_dir + '/driving_log.csv' # should error check this

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        if header:
            next(reader, None)
        for line in reader:
            measurement = float(line[3])
            img_center = csv_dir + '/IMG/' + line[0].split('/')[-1].strip()
            img_left = csv_dir + '/IMG/' + line[1].split('/')[-1].strip()
            img_right = csv_dir + '/IMG/' + line[2].split('/')[-1].strip()

            # add images
            image_list.append(img_center)
            image_list.append(img_left)
            image_list.append(img_right)

            # add steering and correction for side cams
            measurements.append(measurement)
            measurements.append(measurement + correction)
            measurements.append(measurement - correction)

    return (image_list, measurements)

#imagePaths, measurements = read_csv_file('./data', correction=0.3, header=True)
#imagePaths, measurements = read_csv_file('./CarNDTrackData2', correction=0.3)
#imagePaths, measurements = read_csv_file('./CarNDTrackData3', correction=0.3)
imagePaths, measurements = read_csv_file('./CarNDTrackData4', correction=0.3, header=True)
#imagePaths, measurements = read_csv_file('./CarNDTrackData5', correction=0.2, header=True)

X_train, X_valid, y_train, y_valid = train_test_split(imagePaths, measurements, test_size=0.2)
X_train, y_train = shuffle(X_train, y_train)

# print(len(imagePaths))
# print(len(measurements))
# X_train = np.array(images)
# y_train = np.array(measurements)

print('Train images: {} Train measurements: {}'.format(len(X_train), len(y_train)))
print('Validation samples: {} validation measurements: {}'.format(len(X_valid), len(y_valid)))

def generator(X, y, batch_size=32, train=True):
    X, y = shuffle(X, y)
    num_samples = len(X)
    print("generator i sprocessing sample images: ", num_samples)

    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = X[offset:offset+batch_size]
            images = []
            measurements = []
            for i in range (len(batch_samples)):
                image = cv2.imread(X[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # suggested in forums - not yet implemeted
                # do we really need to?
                # randomize brightness
                ...
                # resize and normalize
                ...
                images.append(image)
                measurements.append(y[i])
                if train:
                    # augment the data by flipping images
                    # but only when training not validating
                    images.append(cv2.flip(image, 1))
                    measurements.append(y[i] * -1.0)

            # convert to float
            images = np.array(images)
            measurements = np.array(measurements)

            # shaken not stirred, shuffle again
            yield shuffle(images, measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers import MaxPooling2D
from keras import optimizers

# basic parts of the model
def modelCommon():
    model = Sequential()
    # normalize the image
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # crop the image
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def modelnVidia(model):
    # from nVidia SDC model - good baseline to start with
    # crop so we only see the road and not the sky and the hood
    # is this the same as nVisa model actually used? significant?
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def simpleModel(model):
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return (model)

# create the full model
model = modelCommon()
#model = simpleModel(model)
model = modelnVidia(model)

adam = optimizers.Adam
#adam = optimizers.Adam(lr=0.0001)
#adam = optimizers.Adamax()
#Adamax = optimizers.Adamax(lr=0.0001)

train_generator = generator(X_train, y_train, batch_size=32, train=True)
validation_generator = generator(X_valid, y_valid, batch_size=32, train=False)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
    len(X_train), validation_data=validation_generator, \
    nb_val_samples=len(X_valid), nb_epoch=3, verbose=1)

# save the model
model.save('model.h5')
print("run complete")
