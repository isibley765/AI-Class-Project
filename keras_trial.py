from keras.preprocessing import image
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from keras import layers
import keras.backend as K

import os
import cv2
import json
import numpy as np
import PIL
from pprint import pprint as pp


"""
model = Sequential([
    layers.Conv2D(32, 6, strides=(2,2), input_shape=(train[0].shape), data_format="channels_last"),
    layers.Reshape((3, 169344)),
    layers.ReLU(max_value=.3),
    layers.Dense(2),
])

model.compile(
    optimizer="adagrad",
    loss="mean_squared_error",
    metrics=["accuracy"])

pp(points)

for p in points:
    cv2.circle(img, p, 1, (0, 0, 255), 2)
"""

class TrioDetector:
    def __init__(self, trainFile="./train/train.json", modelweights=["TrioDetectorWeights.dat", "done_weightsTrioDetectorWeights.dat"], save=False):
        self.trainPath = trainFile
        self.modelWeights = modelweights
        self.save = save

        self.gen = image.ImageDataGenerator(data_format="channels_first")

        self.trainData = []
        self.labels = []

        # filesave = "{pixel_accuracy:.4e}_"+self.modelWeights[0]
        filesave = self.modelWeights[0]

        self.callbacks = [ModelCheckpoint(filesave, monitor="pixel_accuracy", mode="max", save_best_only=True, save_weights_only=True)]

        self.prepareTraining()

        self.model = Sequential([
            layers.Conv2D(32, 6, strides=(2,2), input_shape=(self.trainData[0].shape), data_format="channels_first"),
            layers.Reshape((3, 169344)),
            layers.Dense(32),
            layers.ReLU(max_value=.3),
            layers.Dense(2),
        ])

        def pixel_loss(yTrue, yPred):
            a = K.sum(K.square(yTrue[0][0]-yPred[0][0]))
            b = K.sum(K.square(yTrue[0][1]-yPred[0][1]))
            c = K.sum(K.square(yTrue[0][2]-yPred[0][2]))
            return a+b+c

        def pixel_accuracy(yTrue, yPred):
            a = K.sum(K.square(yTrue[0][0]-yPred[0][0]))
            b = K.sum(K.square(yTrue[0][1]-yPred[0][1]))
            c = K.sum(K.square(yTrue[0][2]-yPred[0][2]))
            return 1.0/(a+b+c)

        self.model.compile(
            optimizer="adagrad",
            loss=pixel_loss,
            metrics=[pixel_accuracy])
        
    def prepareTraining(self, file=None):
        if file is None:
            file = self.trainPath
        
        if file is None:
            pp("Can't train without a file to train on")
            return
        
        with open(file, "r") as f:
            data = json.loads(f.read())

        for el in data:
            # pp(el)
            img = cv2.imread(el["img"], cv2.IMREAD_GRAYSCALE)
            self.trainData.append(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).transpose())   # Only for greyscale images being put on a list later

            img = cv2.imread(el["img"])
            self.trainData.append(img.transpose())
            
            self.labels.append(np.array(el["trio"]))
            self.labels.append(np.array(el["trio"]))
        
        self.trainData = np.asarray(self.trainData)
    
    def addWeights(self, weights=None):
        if weights is None:
            weights = self.modelWeights
        def callMe(weights):
            if not type(weights) is str:
                return False
            try:
                self.model.load_weights(weights)
                return True
            except:
                return False
        if type(weights) is list:

            for weight in weights:
                if callMe(weight):
                    pp("Weight {} successfully pulled in".format(weight))
                    return True
        elif type(weights) is str:
            if callMe(weights):
                pp("Weight {} successfully pulled in".format(weights))
                return True
        
        return False
    
    def saveWeights(self, file="done_weightsTrioDetectorWeights.dat"):
        self.model.save_weights(file)

    def saveModelArchitecture(self, file="done_arc_TrioDetector.json"):
        out = self.model.to_json()

        with open(file, "w") as fp:
            fp.write(out)
    
    def saveWholeModel(self, file="done_whole_TrioDetector.dat"):
        self.model.save(file)
    
    def trainModel(self):
        fakeValidX = self.trainData[0]
        fakeValidX = fakeValidX.reshape((1,)+fakeValidX.shape)

        fakeValidY = self.labels[0]
        fakeValidY = fakeValidY.reshape((1,)+fakeValidY.shape)

        self.model.fit_generator(
            self.gen.flow(self.trainData, self.labels, batch_size=24),
            steps_per_epoch=24, epochs=100, use_multiprocessing=True,
            callbacks=self.callbacks, validation_data=(fakeValidX, fakeValidY))
        
        if self.save:
            self.saveWeights()
    
    def runModel(self, file="trial_sets/Ian_Sibley/Ian_Sibley_L.jpg"):
        image = cv2.resize(cv2.imread(file), (256, 256))
        img = np.asarray(image.transpose())   # Only for greyscale images being put on a list later
        
        points = self.model.predict(img.reshape((1,)+img.shape))

        for p in points[0]:
            cv2.circle(image, tuple(p), 1, (0, 0, 255), 2)
        
        self.show(image)
        
        
    def show(self, img):
        cv2.imshow("Model", img)
        cv2.waitKey(250)
        cv2.destroyAllWindows()

        
if __name__ == "__main__":
    td = TrioDetector(save=True)
    td.addWeights()
    td.trainModel()
    
    path = "./smallset/"

    """
    for name in os.listdir(path):
        where = os.path.join(path, name)
        for folder in os.listdir(where):
            bucket = os.path.join(where, folder)
            if folder.endswith(".jpg"):
                td.runModel(file=bucket)
            else:
                for f in os.listdir(bucket):
                    if f.endswith(".jpg"):
                        td.runModel(file=os.path.join(bucket, f))
    """