import os
import cv2
import random
import numpy as np
from copy import deepcopy
from pprint import pprint as pp
import tensorflow as tf

from utils.MatchFaces import TrainSet
from utils.FaceFeatPos import GetFace
from utils.FacePathList import PathList
from utils.MakeBucketImages import FaceBucketMaster

class RunProcesses:
    def __init__(self, bucketList=None):
        self.genericFace = GetFace(np.zeros((255, 255, 3), dtype=np.uint8)) # Generic face instance, for some of the neutral capabilities

        self.bucketMaker = FaceBucketMaster()

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

        for name in os.listdir(path):
            npath = os.path.join(path, name)
            if os.path.isdir(npath):
                s = TrainSet(path=npath, name=name)
                s.getImageFolder()

                self.bucketList.append(s)
    
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
    rp = RunProcesses()

    # rp.findSets("./sets/")

    # rp.showAvgs(avg="spec")

    rp.bucketMaker.getPeople("./sets/", endFolder="./trial_sets/")