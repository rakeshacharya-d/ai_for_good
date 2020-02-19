#-----------------
#Author @Rakesh Acharya Dharoori
#Training Script
#-----------------

import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mycvlibrary.learningratefinder import LearningRateFinder
from mycvlibrary.clr_callback import CyclicLR
from mycvlibrary import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import sys
import os


# Argument parsing

ap = argparse.ArgumentParser()
ap.add_argument("-f","--lr-find",type=int,default=0,help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

# labeling

print("loading images...")
imagePaths = list(paths.list_images(config.DATASET_PATH))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    data.append(image)
    labels.append(label)

# data and labels to NumPy arrays
print("processing data...")
data = np.array(data,dtype="float32")
labels = np.array(labels)

# encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# spit into training and testing scrpits
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=config.TEST_SPLIT,random_state=42)

# validation split from training split
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=config.VAL_SPLIT, random_state=84)

# training data augmentation object
aug = ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

#VGG16 network

baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

# head of base model

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# looping layers so that it is freezed

for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("compiling model...")
opt = SGD(lr=config.MIN_LR,momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

if args["lr_find"]>0:

    #learning rates 1e-10 to 1e+1

    print("finding learning rate...")
    lrf =  LearningRateFinder(model)
    lrf.find( aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), 1e-10, 1e+1, stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
              epochs=20,
              batchSize=config.BATCH_SIZE)
    lrf.plot_loss()
    plt.savefig(config.LRFIND_PLOT_PATH)

    print("learning rate finder complete")
    print("examine plot and adjust learning rates before training")
    sys.exit(0)

# Cyclic learning rate finder

stepSize = config.STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
clr = CyclicLR( mode=config.CLR_METHOD,
                base_lr=config.MIN_LR,
                max_lr=config.MAX_LR,
                step_size=stepSize)

# train network

print("training network...")
H = model.fit_generator( aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
                         validation_data=(valX, valY),
                         steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE,
                         epochs=config.NUM_EPOCHES,
                         callbacks=[clr],
                         verbose=1)
# evaluating network
print("evaluating network...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.CLASSES))

#serialize model
print("serializing network to...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)



    





