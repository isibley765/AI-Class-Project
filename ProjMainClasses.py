import os
import cv2
import random
import numpy as np
from copy import deepcopy
from pprint import pprint as pp

from utils.FaceFeatPos import getFace

class FaceBucketMaster:
    def __init__(self, imageList=None, bucketList=None):
        if type(imageList) == list:
            self.images = deepcopy(imageList)
        else:
            self.images = {}
        
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
        images = []
        for name in os.listdir(path):
            tryImage = os.path.join(path, name)
            if name.endswith((".jpg", ".jpeg", ".png")):
                images.append({"img": cv2.imread(tryImage), "name": name})
            else:
                # print("Directory?: {}".format(tryImage))
                if os.path.isdir(os.path.join(path, name)):
                    images += self.takeFolder(tryImage)
        
        return images

    """
        Expects a directory labeled by name, with images inside, possibly in subfolders
        Moves images to local image repo with L/R/C categories for further processing
        Doesn't load locally and keep just in RAM, because that blows up into huge memory real fast
    """
    def getPeople(self, path):
        names = sorted(os.listdir(path))
        for name in names:
            namepath = os.path.join(path, name)
            if os.path.isdir(namepath):
                images = self.takeFolder(namepath)
                if len(images) > 40: # Find random sample of 40 from list that's bigger than 40
                    images = [images[i] for i in sorted(random.sample(range(len(images)), 40))]
                pp(len(images))
                self.makeBins(name, images)
    
    def makeBins(self, name, imageList):
        bucket = os.path.join(os.path.join(os.getcwd(), "buckets"))
        namebucket = os.path.join(bucket, name)
        pp("Making {}'s training folder".format(name))

        looks = {
            "SL": os.path.join(namebucket, "SL"),   # Slight left or right
            "SR": os.path.join(namebucket, "SR"),
            "L":  os.path.join(namebucket, "L"),    # Normal left, right, or center
            "R":  os.path.join(namebucket, "R"),
            "C":  os.path.join(namebucket, "C"),
            "HL": os.path.join(namebucket, "HL"),   # Hard left or right
            "HR": os.path.join(namebucket, "HR"),
        }

        # These need to exist before we make the L, C, and R subfolders
        if not os.path.exists(bucket):
            os.makedirs(bucket)
              
        if not os.path.exists(namebucket):
            pp(namebucket)
            os.makedirs(namebucket)

        for sub in looks.keys():
            if not os.path.exists(looks[sub]):
                os.makedirs(looks[sub])
        
        for image in imageList:
            feat = getFace(image["img"])

            pose = feat.findAngle()
            # pp(pose)

            """
                Pose indexes correspond to faces found in the image
                Only one index means only one face found in the image
                Since this is making our training set, we don't want to identify the desired face just yet
            """
            if len(pose) == 1:
                pose = pose[0]
                if pose[1] < -0.45:
                    out = os.path.join(looks["HL"], image["name"])
                elif pose[1] < -0.20:
                    out = os.path.join(looks["L"], image["name"])
                elif pose[1] > 0.45:
                    out = os.path.join(looks["HR"], image["name"])
                elif pose[1] > 0.20:
                    out = os.path.join(looks["R"], image["name"])
                else:
                    out = os.path.join(looks["C"], image["name"])
            
                cv2.imwrite(out, image["img"])



if __name__ == "__main__":
    f = FaceBucketMaster()

    # f.takeFolder("/home/rovian/Desktop/aligned_images_DB/Abel_Pacheco")
    # f.addImage("/home/rovian/Documents/GitHub/head-pose-estimation/yourface1.jpg")

    f.getPeople("/home/rovian/Desktop/aligned_images_DB/")