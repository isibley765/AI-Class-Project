import os
import cv2
import json
import random
import numpy as np
from copy import deepcopy
from pprint import pprint as pp
import tensorflow as tf
import face_recognition as frec

from utils.MatchFaces import TrainSet
from utils.FaceFeatPos import GetFace
from utils.FacePathList import PathList
from utils.MakeBucketImages import FaceBucketMaster
from utils.FeatDetkct import TrioDetector

class RunProcesses:
    def __init__(self, bucketList=None, saveTrioTraining=False, trainFile=None):
        if saveTrioTraining and not type(trainFile) is str:
            raise ValueError("Must have file to do training on, if trainnig is desired")
        
        self.genericFace = GetFace(np.zeros((255, 255, 3), dtype=np.uint8)) # Generic face instance, for some of the neutral capabilities

        self.bucketMaker = FaceBucketMaster()

        self.triDetect = TrioDetector(save=saveTrioTraining, trainFile=trainFile)

        if type(bucketList) == list:
            self.bucketList = deepcopy(bucketList)
        else:
            self.bucketList = []
        
        self.images_placeholder = tf.placeholder(tf.float32, shape=[None, 256 * 256])
        self.labels_placeholder = tf.placeholder(tf.int64, shape=[None])

        self.weights = None
        self.biases = None

    def findSets(self, path):   # Assumes that a folder is given of sorted images, with folders named for the people
        if type(path) != str:
            raise ValueError("Path must be string type")
        self.bucketList = []

        for name in os.listdir(path):
            npath = os.path.join(path, name)
            if os.path.isdir(npath):
                self.getSet(npath, name)
    
    def getSet(self, path, name):
        s = TrainSet(path=path, name=name)
        s.getImageFolder()

        self.bucketList.append(s)
    
    def getImgs(self, path):
        out = []

        if path.endswith(".jpg"):   # all current considered images end in jpg
            out.append(cv2.imread(path))
        elif os.path.isdir(path):
            for item in os.listdir(path):
                out.extend(self.getImgs(os.path.join(path, item)))
            
        return out
    
    def simplifyListNums(self, ints):

        for el in ints:
            if type(el) is list:
                self.simplifyListNums(el)

    def compare(self, path=None):
        if type(path) == None:
            pp("Need a path of name folders")
        self.findSets(path)
        """
            TODO
            1. For every member of the bucketList, find the folder of the same name
                - Extract all named images if present
                - If no name... make their slot empty?
            2. For every image, compare to known images
                - Start with just their images to get working
                - Expand to everyone's images
            3. Compare performance from averages to original templates
                - Output to a text file
                    > statistics per person, and overall accuracy?
        """

        out = {}
        imgList = {}

        pp("Working on image collection...")
        for person in self.bucketList:
            name = person.name

            out[name] = {"avg": 0, "norm": 0, "oavg": 0, "onorm": 0}
            imgList[name] = None

            namePath = os.path.join(path, name)
            if os.path.isdir(namePath):
                imgList[name] = self.getImgs(namePath)
        
        testOther = {}
        for name in out.keys():
            testOther[name] = []
            pp("Working on {}'s own images...".format(name))
            for person in self.bucketList:
                imgs = imgList[person.name]
                if person.name == name:
                    if len(imgs) != 0:
                        a = np.asarray(person.compareGrouptToGroup(imgs, group=person.averages["norm"])).mean()
                        b = np.asarray(person.compareGrouptToGroup(imgs, group=person.imageBins)).mean()

                        if np.isnan(a):
                            a = 0.0
                        if np.isnan(b):
                            b = 0.0
                        out[name]["avg"] = a
                        out[name]["norm"] = b
                else:
                    testOther[name].extend(imgs)

        for person in self.bucketList:
            imgs = testOther[person.name]
            if len(imgs) != 0:
                pp("Working on {}'s others...".format(person.name))
                a = np.asarray(person.compareGrouptToGroup(imgs, group=person.averages["norm"])).mean()
                b = np.asarray(person.compareGrouptToGroup(imgs, group=person.imageBins)).mean()

                if np.isnan(a):
                    a = 0.0
                if np.isnan(b):
                    b = 0.0
                out[name]["oavg"] = a
                out[name]["onorm"] = b
        
        with open("compareRes_fedup.txt", "w") as fp:
            fp.write(json.dumps(out, indent=4))
        
        with open("compareResults.txt", "w") as fp:
            sums = {"avg": 0, "norm": 0, "oavg": 0, "onorm": 0}
            count = 0
            for key in out.keys():
                fp.write(key+"\n")
                good = True
                for kind in sums.keys():
                    sums[kind] += out[key][kind]
                    good += not out[key][kind] == 0 # see if all or partial identification

                if not good:  # Skip those that couldn't be identified at all, but keep partial identifications
                    count += 1

                for kind in out[key]:
                    fp.write("    "+kind+": {}\n".format(out[key][kind]))
                fp.write("\n")

            fp.write("Totals:\n")
            for kind in sums.keys():
                fp.write("    "+kind+": {}\n".format(sums[kind]/count))
            fp.write("count: {}".format(count))
    
    def showAvgs(self, avg="norm", delay=250):
        for s in rp.bucketList:
            for el in s.buckets:
                pp("Average {}:".format(el))
                for image in s.averages[avg][el]:
                    s.show(image, delay)

    def initTraining(self):
        pCount = len(self.bucketList)
        if pCount == 0:
            pp("Need to have a set to train on")
            return
        
        self.weights = tf.Variable(tf.zeros([256 * 256, pCount]))
        self.biases = tf.Variable(tf.zeros([pCount]))



if __name__ == "__main__":
    rp = RunProcesses(saveTrioTraining=True, trainFile="./train/test_set_new.json")
    # rp.triDetect.addWeights()
    # rp.triDetect.trainModel(epochs=80, steps=48)


    # rp.findSets("./sets/")

    # rp.showAvgs(avg="spec")

    # rp.bucketMaker.getPeople("/home/rovian/Desktop/frame_images_DB/", endFolder="./real_try/", affine=True, avg=False)
    rp.compare("./real_try/")
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.open()
    a = -1

    while(a == -1):
        ret, frame = cap.read()

        face = GetFace(image_face=frame)
        faceFront = face.faceFront()
        if len(faceFront) > 0:
            pp("Start")
            faceFront = faceFront[0]
            pp(faceFront.shape)
            feats = rp.triDetect.runModel(faceFront)
            pp(feats)

            for feat in feats[0]:
                cv2.circle(faceFront, tuple(feat), 1, (0, 0, 255), 2)

            cv2.imshow("frame", faceFront)
            a = cv2.waitKey(1)

    pp(a)
    """
