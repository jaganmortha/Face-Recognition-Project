import numpy as np
import os
import cv2
import glob as gb
import pandas as pd
import package

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from keras.callbacks import LearningRateScheduler

import package.preprocess_data


def train_model(train_path, test_path):
    # Splitting the data into training and testing
    # train_path = r'data/splitted_dataset/train'
    # test_path = r'data/splitted_dataset/val'

    images_size = 160
    batch_size = 16 # Number of images to process at a time

    # Checking for invalid or corrupted images
    # package.preprocess_data.check_images()

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
    rescale=1.0/255,                # Rescale pixel values to [0, 1] (Normalise)
    rotation_range=30,              # rotation within 30 degrees
    width_shift_range=0.3,          # horizontal shift by 30% of image width
    height_shift_range=0.3,         # vertical shift by 30% of image height
    horizontal_flip=True,           # horizontal flipping
    fill_mode='nearest'             # Fill mode for new pixels after shifts/rotations
    )

    train_generator = train_datagen.flow_from_directory(
    train_path,                     # Path to the training data directory
    target_size=(images_size, images_size),  # Resize images to this size
    batch_size=batch_size,           # Number of images in each batch
    seed=32,                         # Optional: Set a random seed for shuffling
    shuffle=True,                    # Shuffle the data during training
    class_mode='categorical'        # Mode for class labels (categorical for one-hot encoding)
    )

    # data augmentation for testing
    test_datagen = ImageDataGenerator(rescale=1.0/255)  # Rescale pixel values to [0, 1]
    # Create a generator for testing data
    test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(images_size, images_size),
    batch_size = batch_size,
    class_mode='categorical')

    # Create a learning rate schedule using Exponential Decay
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,  # Initial learning rate for training
    decay_steps=1000,            # Number of steps to wait before decaying the learning rate
    decay_rate=0.5,              # Rate by which the learning rate decreases
    )

    # Create a Learning Rate Scheduler callback using a pre-defined schedule
    lr_callback = LearningRateScheduler(learning_rate_schedule)
    # Configuring the Early Stopping callback
    early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5,      # how many epochs to wait before stopping
    restore_best_weights=True,
    )

    # Configuring the learning rate reduction callback
    learning_rate_reduce = ReduceLROnPlateau(
    monitor='val_accuracy',   # Metric to monitor for changes (usually validation accuracy)
    patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,           # Verbosity mode (0: silent, 1: update messages)
    factor=0.5,          # Factor by which the learning rate will be reduced 
    min_lr=0.00001       # Lower bound for the learning rate (it won't go below this value)
    )

    # Loading pretrained model as base model
    base_model = keras.models.load_model(r'models/facenet_keras.h5')
    model = tf.keras.models.Sequential([
    base_model,
    # customizing the model 
    layers.Dense(512,activation='relu'), 
    layers.Dense(len(list(train_generator.class_indices.keys())),activation='softmax'), # Classify images into classes using probability distribution
    ])
    model.summary()

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=optimizer,
             loss="categorical_crossentropy",
              metrics=['accuracy']
             )
    
    callback = [ lr_callback , learning_rate_reduce ,early_stopping ]

    history = model.fit(
    train_generator,
    steps_per_epoch=1,
    # steps_per_epoch=train_generator.samples // batch_size,
    # epochs=50,
    epochs=1,
    validation_data=test_generator,
    validation_steps=1,
    # validation_steps=test_generator.samples // batch_size,
    callbacks=[callback]
    )

    # Evaluate on test dataset
    score = model.evaluate(test_generator, verbose=False)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model
    model.save(r"E:\Project\models\new_model.h5")  # Save in HDF5 format

    return list(train_generator.class_indices.keys())