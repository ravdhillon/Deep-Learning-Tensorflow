import os
import random
import pickle
import cv2
import argparse

import numpy as np
import pandas as pd
import matplotlib
# set matplotlib backend so figures can be saved in the background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imutils import paths

#import ML libraries and framework
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

# import model
from classifier.model import FashionVGGNet

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset")
ap.add_argument("-m", "--model", required=False, help="path to output model")
ap.add_argument("-l", "--labelbin", required=False, help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=False, help="path to output accuracy/loss plot")

args = vars(ap.parse_args()) # args as dictinary key-value pair

# define some constants
EPOCHS = 50
LEARNING_RATE = 1e-3 # default value for Adam Optimizer
BATCH_SIZE = 32
IMAGE_DIMENSIONS = (96, 96, 3)

#read images from the dataset and randomly shuffle them
print("[INFO]: loading Images ...")
dataset_path = args["dataset"]
image_paths = [os.path.join(dataset_path, image_path) for image_path in os.listdir(dataset_path)]
images = []

for img_path in image_paths:    
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    images = images + img_list

random.seed(43)
random.shuffle(images)

data = []
labels = []

# Let's see some images to confirm
import matplotlib.image as mpimg
# for img in images[:4]:
#     image = mpimg.imread(img)
#     implot = plt.imshow(image)
#     plt.show()

# loop over the image paths and read the images
for img in images:
    # load the image, pre-process it and store it in the data list
    image = cv2.imread(img)    
    image = cv2.resize(image, (IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0]))
    image = img_to_array(image)
    data.append(image)

    l = label =img.split(os.path.sep)[-2].split('_')
    labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(images), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(mlb.classes_)
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model ... ")
model = FashionVGGNet.build(width=IMAGE_DIMENSIONS[1], height=IMAGE_DIMENSIONS[0], depth=IMAGE_DIMENSIONS[2],
                            classes=len(mlb.classes_), final_act="sigmoid")

# initialize the optimizer
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

# compile the model using Binary cross-entropy rather than categorical cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.summary())
# train the network
print("[INFO] training network ...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), validation_data=(testX, testY),
steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
 
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])