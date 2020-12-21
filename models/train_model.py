# Importing the libraries
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import time

#! Preparing the training and the test set

# Preparing the image preprocessing specfications for the training set
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
    )

# Preparing the image preprocessing specfications for the test set
test_datagen = ImageDataGenerator(
    rescale = 1./255
    )

# Preparing the training set
training_set = train_datagen.flow_from_directory(
    r'C:/Users/Selvaseetha/YouTube Codes/Pneumonia X-Ray Detector/archive/chest_xray/train',
    target_size = (416, 416),
    batch_size = 64,
    class_mode = 'binary'
    )

# Preparing the test-set
test_set = test_datagen.flow_from_directory(
    r'C:/Users/Selvaseetha/YouTube Codes/Pneumonia X-Ray Detector/archive/chest_xray/test',
    target_size = (416, 416),
    batch_size = 64,
    class_mode = 'binary'
    )

#! Introducing the neural network

# Initializing the neural network
cnn = tf.keras.models.Sequential()

# Adding layers to the neural network

# Convolutional Layers
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(416, 416, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, input_shape=(416, 416, 16)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(208, 208, 16)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape=(208, 208, 32)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(104, 104, 32)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape=(104, 104, 64)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(52, 52, 64)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape=(52, 52, 128)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(26, 26, 128)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, input_shape=(26, 26, 256)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(13, 13, 256)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, input_shape=(13, 13, 512)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(13, 13, 512)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(13, 13, 1024)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=1, activation='relu',input_shape=(13, 13, 1024)))
cnn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=1, activation='relu',input_shape=(13, 13, 125)))
cnn.add(tf.keras.layers.Flatten())

# Fully Connected layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the training & the test set into the model
tic = time.time()
cnn.fit(x=training_set, validation_data=test_set, epochs=1)
toc = time.time()

print(f"Training took {str((toc - tic)/60)} mins")