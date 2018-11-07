from keras.preprocessing import image
from keras.models import Sequential, model_from_json, Model
from keras.callbacks import ModelCheckpoint
from keras import layers
import keras.backend as K

import os
import cv2
import json
import random
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

        self.callbacks = [ModelCheckpoint(filesave, monitor="pixelavg_accuracy", mode="min", save_best_only=True, save_weights_only=True)]

        self.prepareTraining()

        self.imgSize = (3, 256, 256)

        mainIn = layers.Input(shape=self.imgSize, name="TrioIn")
        conv = layers.Conv2D(64, 6, strides=(2,2), data_format="channels_first")(mainIn)


        y1 = self.makePointModel("point1", conv)
        y2 = self.makePointModel("point2", conv)
        y3 = self.makePointModel("point3", conv)

        x = layers.concatenate([y1, y2, y3])
        x = layers.Reshape((3,2))(x)

        self.model = Model(inputs=[mainIn], outputs=[x])

        self.compile()
    
    def makePointModel(self, name, tensIn):
        x = layers.LeakyReLU()(tensIn)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4, strides=(2,2), data_format="channels_first")(x)
        x = layers.BatchNormalization()(x)
        #pp(K.int_shape(x))

        size = 1
        for el in K.int_shape(x)[1:]:   # dynamic finding of previous layer shape, skipping first None value for batches
            size *= el
        
        x = layers.Reshape((1, size))(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(24)(x)
        x = layers.LeakyReLU()(x)
        pointOut = layers.Dense(2, name=name)(x)

        return pointOut
    
    def compile(self):
        def pixel_loss(yTrue, yPred):
            a = K.sum(K.square(yPred[0][0]-yTrue[0][0]))
            b = K.sum(K.square(yPred[0][1]-yTrue[0][1]))
            c = K.sum(K.square(yPred[0][2]-yTrue[0][2]))
            return a+b+c
        
        def pixel_accuracy(index):  # Decided to make a function defining function for simplicity
            def internal(yTrue, yPred):
                a = K.sum(K.square(yPred[0][index]-yTrue[0][index]))
                return (a)
            return internal
        
        pixela_accuracy = pixel_accuracy(0)
        pixelb_accuracy = pixel_accuracy(1)
        pixelc_accuracy = pixel_accuracy(2)
        
        def pixelavg_accuracy(yTrue, yPred):
            a = pixela_accuracy(yTrue, yPred)
            b = pixelb_accuracy(yTrue, yPred)
            c = pixelc_accuracy(yTrue, yPred)
            return (a+b+c)/3.0


        self.model.compile(
            optimizer="adagrad",
            loss=pixel_loss,
            metrics=[pixelavg_accuracy,pixela_accuracy,pixelb_accuracy,pixelc_accuracy,])
        
    def prepareTraining(self, file=None):
        if file is None:
            file = self.trainPath
        
        if not type(file) is str:
            pp("Can't train without a file to train on, trainFile parameter must be string type")
            return
        
        with open(file, "r") as f:
            data = json.loads(f.read())

        for el in random.sample(range(len(data)), len(data)):
            el = data[el]
            img = cv2.imread(el["img"])
            self.trainData.append(img.transpose())   # Only for greyscale images being put on a list later

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
    
    def saveWeights(self, file="TrioDetectorWeights.dat"):
        self.model.save_weights(file)
        pp("Successfully saved weights to {}".format(file))

    def saveModelArchitecture(self, file="done_arc_TrioDetector.json"):
        out = self.model.to_json()

        with open(file, "w") as fp:
            fp.write(out)
    
    def saveWholeModel(self, file="done_whole_TrioDetector.dat"):
        self.model.save(file)
    
    def trainModel(self, epochs=50, steps=24):
        if not type(self.trainPath) is str:
            pp("Cannot train on a file, if path to a file hasn't been properly given")
            return

        fakeValidX = self.trainData[0]
        fakeValidX = fakeValidX.reshape((1,)+fakeValidX.shape)

        fakeValidY = self.labels[0]
        
        #fakeValidY = fakeValidY.reshape((1,)+fakeValidY.shape)

        self.model.fit_generator(
            self.gen.flow(self.trainData, self.labels, batch_size=steps),
            steps_per_epoch=steps, epochs=epochs, use_multiprocessing=True,
            callbacks=self.callbacks)
        
        if self.save:
            self.saveWeights()
    
    def runModel(self, file="trial_sets/Ian_Sibley/Ian_Sibley_L.jpg"):
        if type(file) == str:
            image = cv2.resize(cv2.imread(file), (256, 256))
        else:
            image = file
        img = np.asarray(image.transpose())
        
        return self.model.predict(img.reshape((1,)+img.shape))
        
        
    def show(self, img, delay=250):
        cv2.imshow("Model", img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

        
if __name__ == "__main__":
    # source venv bin activate
    td = TrioDetector(save=True, trainFile="./train/train_big_norm.json")
    td.addWeights()
    #td.trainModel(epochs=20, steps=24)
    #td.saveModelArchitecture(file="./full_version2_acc.dat")

    
    path = "./train/test_set_new.json"


    with open(path, "r") as f:
        data = json.loads(f.read())
    if len(data) > 40:
        data = [data[i] for i in sorted(random.sample(range(len(data)), 20))]
    train = []
    imgs = []
    pts = []

    for el in data:
        train.append(el["img"])   # Only for greyscale images being put on a list later
        #train.append(cv2.resize(cv2.imread(el["img"]), (256, 256)))   # Only for greyscale images being put on a list later
        pts.append(el["trio"])

    for i in range(len(train)):
        points = td.runModel(train[i])
        image = cv2.resize(cv2.imread(train[i]), (256, 256))
        pp([[int(y) for y in x] for x in points[0].tolist()])
        pp(pts[i])
        pp("")
        
        #for p in pts[i]:
        #    cv2.circle(image, tuple(p), 1, (255, 0, 0), 2)
        for p in [[int(y) for y in x] for x in points[0].tolist()]:
            cv2.circle(image, tuple(p), 1, (0, 0, 255), 2)

        cv2.imwrite("./pres/image{}.jpg".format(i), image)
        
        td.show(image, delay=500)


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
                        points = td.runModel(file=os.path.join(bucket, f))

                        for p in points[0]:
                            cv2.circle(image, tuple(p), 1, (0, 0, 255), 2)
                        
                        td.show(image)
    """