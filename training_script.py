#!/usr/bin/env python
# coding: utf-8

import argparse
import tensorflow as tf
import os
import sys
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from imutils import paths
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50V2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
sys.path.append('../')



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# retrieve image paths
imagePaths = args["dataset"]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


paths = os.listdir(imagePaths)

# initialize hyper parameters
IMG_SIZE = 64
CHANNELS = 3
N_LABELS=2
data = []
labels = []


#process images and label them


for imagePath in paths:
    label = imagePath
    fullPath = os.path.join(imagePaths,imagePath)
    for imagePath2 in os.listdir(fullPath):
        imagelocis = os.path.join(fullPath,imagePath2).replace("\\","/")
        for imageloci in os.listdir(imagelocis):
            image = load_img(os.path.join(imagelocis,imageloci).replace("\\","/"), target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = image/255
            data.append(image)
            labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

#categorizing data
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

labels = to_categorical(labels)
print("Current data size is")
print(data.shape)



(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=42)





#data augumentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#import transfer learning backbone
feature_extractor_layer = ResNet50V2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(IMG_SIZE,IMG_SIZE,CHANNELS)))

feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Flatten(name="flatten"),
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dropout(0.5),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')
])
#used for debugging

# model.summary()
# feature_extractor_layer.summary()



# Hyper parameters

LR = 1e-5 # Keep it small when transfer learning
EPOCHS = 20
BS = 8 # Batch size, largest possible value on the local machine

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss="binary_crossentropy",
  metrics=["accuracy"],
    run_eagerly=True)



#used for debugging

# print(model.summary())





import time
start = time.time()

aug.flow(trainX, trainY, batch_size=BS)
history = model.fit(aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	epochs=EPOCHS)





print('\nTraining took {}'.format((time.time()-start)))

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))



print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])




print('\nTraining took {}'.format((time.time()-start)))

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(predIdxs)





print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))







