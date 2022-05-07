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


# In[125]:
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# imagePaths = "test_data/"
# imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = args["dataset"]
print("the image path is " + imagePaths)
paths = os.listdir(imagePaths)
# # print(paths)
# label = paths[0].split(os.path.sep)
# print(label)
IMG_SIZE = 224
CHANNELS = 3
N_LABELS=2
data = []
labels = []


# In[126]:


for imagePath in paths:
    label = imagePath
    fullPath = os.path.join(imagePaths,imagePath)
    for imagePath2 in os.listdir(fullPath):
        imagelocis = os.path.join(fullPath,imagePath2).replace("\\","/")
        for imageloci in os.listdir(imagelocis):
            image = load_img(os.path.join(imagelocis,imageloci).replace("\\","/"), target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = image/255
#             print(image.shape)
            data.append(image)
            labels.append(label)
# print(len(data))
data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

labels = to_categorical(labels)

# print(data.shape)


# In[ ]:





# In[127]:


(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=42)


# In[124]:


# print(data[0].shape)
# print(trainX.shape)
# print(testY[0])


# In[128]:


#data augumentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


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

# model.summary()


# In[90]:


# feature_extractor_layer.summary()


# In[131]:


LR = 1e-5 # Keep it small when transfer learning
EPOCHS = 20
BS = 256

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss="binary_crossentropy",
  metrics=["accuracy"],
    run_eagerly=True)


# In[132]:


# print(model.summary())


# In[133]:


import time
start = time.time()
# aug.flow(trainX, trainY, batch_size=BS)
# (trainX, trainY)
# print(BS)
aug.flow(trainX, trainY, batch_size=BS)
history = model.fit(aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	epochs=EPOCHS)


# In[134]:


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


# In[135]:


print('\nTraining took {}'.format((time.time()-start)))

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(predIdxs)


# In[116]:





# In[109]:


print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))


# In[ ]:




