import os
import cv2
import numpy as np
from copy import deepcopy
from pprint import pprint as pp

class FaceBucketMaster:
    def __init__(self, imageList=None, bucketList=None):
        if type(imageList) == list:
            self.images = deepcopy(imageList)
        else:
            self.images = []
        
        if type(bucketList) == list:
            self.bucketList = deepcopy(bucketList)
        else:
            self.bucketList = []
    
    def takeList(self, imageList):
        temp = []
        for img in imageList:
            try:
                self.addImage(img, temp)
            except:
                continue
        
        self.images.extend(temp)
    
    def addImage(self, image, imageList=None):
        if not imageList:
            imageList = self.images
        
        if type(image) == str:
            image = cv2.imread(image)
            pp(image)
        
        self.images.append(deepcopy(image))
    
    def takeFolder(self, path):
        for name in os.listdir(path):
            tryImage = os.path.join(path, name)
            print(tryImage)
            if name.endswith("jpg"):
                print(tryImage)
            else:
                print("Directory?: {}".format(tryImage))
                if os.path.isdir(os.path.join(path, name)):
                    self.takeFolder(tryImage)

if __name__ == "__main__":
    f = FaceBucketMaster()

    # f.takeFolder("/home/rovian/Desktop/aligned_images_DB/Abel_Pacheco")
    f.addImage("/home/rovian/Documents/GitHub/head-pose-estimation/yourface1.jpg")