# USAGE
# python detect_mask_image.py --images testimages (should be a directory)

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse
import cv2
import os
import pandas as pd
import sys
from sklearn.preprocessing import LabelBinarizer

sys.path.append('../')


def mask_image():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
        help="path to input images")
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model(args["model"])


    imagePaths = args["images"]

    imagelocations = []
    labels = []
    paths = os.listdir(imagePaths)
    for imagePath in paths:
        label = imagePath
        fullPath = os.path.join(imagePaths,imagePath)
        for imagePath2 in os.listdir(fullPath):
            imageloci = os.path.join(fullPath,imagePath2).replace("\\","/")
            imagelocations.append(imageloci)
            
            if (label == "masked"):
                labels.append(1)
            else:
                labels.append(0)

            



    predictions =[]
    counter = 0
    for imagedir in imagelocations:
        # print(imagedir)
        image = cv2.imread(imagedir)
        orig = image.copy()
        (h, w) = image.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        # print("[INFO] computing face detections...")

        net.setInput(blob)
        detections = net.forward()
            # loop over the detections
        curPred = -1
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:

                counter = counter + 1

                # print("hello")
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (64, 64))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # pass the face through the model to determine if the face
                # has a mask or not
                (mask, withoutMask) = model.predict(face)[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                # as long as there is one none-mask in the image, recognize as unmasked.
                if mask > withoutMask and curPred != 0:
                    curPred = 1                   
                else:
                    curPred = 0
        predictions.append(curPred)
                    

            
    print(counter)
    print(len(labels))
    print(len(predictions))        
    comparison = pd.DataFrame(

    {'pic_name': imagelocations,
        'truth': labels,
     'prediction': predictions,

    })

    print(comparison)
if __name__ == "__main__":
    
    mask_image()