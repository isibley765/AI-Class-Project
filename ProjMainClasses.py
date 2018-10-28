import os
import cv2
import random
import numpy as np
from copy import deepcopy
from pprint import pprint as pp

from utils.FaceFeatPos import GetFace

class PathList:
    def __init__(self, path, images=None, name=None):
        self.path = path
        self.images = []
        self.name = name

        self.shape = None

        if type(images) == list:
            self.addImageList(images)
        elif images is not None:
            raise ValueError("images parameter must be a List if present, even if only one image present")
    
    def addImage(self, image, name):
        if type(image) is np.ndarray:
            if not self.shape:
                self.shape = image.shape
            
            if image.shape == self.shape:
                self.images.append({"img": image, "name": name})
            else:
                raise ValueError("Images must be of the same size for averaging")
        else:
            raise ValueError("Only numpy array images allowed in PathList instances, for simplicity")
    
    def addImageList(self, images):
        if type(images) == list:
            for image in images:
                self.addImage(image["img"], image["name"])
        else:
            raise ValueError("PathList: images are Lists or Bust on initialization and addImageList calls")
    
    def getAllImages(self):
        return self.images
    
    def getImageAverage(self):
        if len(images) != 0:
            avg = np.zeros(self.shape, np.float32)

            N = len(images)
            for image in images:
                avg += image["img"].astype(np.float32) / N
            
            return avg
        
        else:
            return np.zeros(1, dtype=np.uint8)


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
        Doesn't load locally and keep just in RAM, because that blows up into huge memory with large datasets

        affine depicts whether the re-sorting is done with original images, or
        scaled 256x256 images of just the face
    """
    def getPeople(self, path, affine=False):
        names = sorted(os.listdir(path))
        for name in names:
            namepath = os.path.join(path, name)
            if os.path.isdir(namepath):
                images = self.takeFolder(namepath)
                if len(images) > 40: # Find random sample of 40 from list that's bigger than 40
                    images = [images[i] for i in sorted(random.sample(range(len(images)), 40))]
                pp(len(images))
                self.makeBins(name, images, affine)
    
    """
        This function takes a folder named for the person it represents,
        and cycles through the named subfolders and saves the set in a new place, either with
        the original image, or the affine transformation 256x256 image that holds the centered face

        name is the name of the person being sampled
        imageList is a list of their images
        affine is boolean, whether to do an affine transformation or not
    """
    def makeBins(self, name, imageList, affine=False, avg=False):
        bucket = os.path.join(os.path.join(os.getcwd(), "buckets"))
        namebucket = os.path.join(bucket, name)

        pp("Making {}'s training folder".format(name))

        looks = {
            "SL": PathList(os.path.join(namebucket, "SL"), name=name),   # Slight left or right
            "SR": PathList(os.path.join(namebucket, "SR"), name=name),
            "L":  PathList(os.path.join(namebucket, "L"), name=name),    # Normal left, right, or center
            "R":  PathList(os.path.join(namebucket, "R"), name=name),
            "C":  PathList(os.path.join(namebucket, "C"), name=name),
            "HL": PathList(os.path.join(namebucket, "HL"), name=name),   # Hard left or right
            "HR": PathList(os.path.join(namebucket, "HR"), name=name),
            "step": 0.15
        }

        # These need to exist before we make the L, C, and R subfolders
        if not os.path.exists(bucket):
            os.makedirs(bucket)
              
        if not os.path.exists(namebucket):
            pp(namebucket)
            os.makedirs(namebucket)

        for sub in looks.keys():
            if sub != "step" and not os.path.exists(looks[sub].path):
                os.makedirs(looks[sub].path)
        
        for image in imageList:
            feat = GetFace(image["img"])

            pose = feat.findAngle()
            affout = feat.warpFaceFront()
            # pp(pose)

            """
                Pose indexes correspond to faces found in the image
                Only one index means only one face found in the image
                Since this is making our training set, we don't want to identify the desired face just yet
            """
            if len(affout) == 1:   # This means only one face has been found in the image
                pose = pose[0]
                affout = affout[0]
                if pose[1] < -looks["step"]*3:
                    out = "HL"
                elif pose[1] < -looks["step"]*2:
                    out = "L"
                elif pose[1] < -looks["step"]:
                    out = "SL"
                elif pose[1] > looks["step"]*3:
                    out = "HR"
                elif pose[1] > looks["step"]*2:
                    out = "R"
                elif pose[1] > looks["step"]:
                    out = "SR"
                else:
                    out = "C"

                if affine:
                    looks[out].addImage(affout, image["name"])
                    pp("Adding image {} to {}".format(image["name"], out))
                else:
                    cv2.imwrite(os.path.join(looks[out].path, image["name"]), image["img"])
        
        if affine:
            for sub in looks.keys():
                if sub != "step":
                    pp(sub+":")
                    for image in looks[sub].getAllImages():
                        pp("    "+image["name"])
                    if avg:
                        cv2.imwrite(os.path.join(looks[sub].path, looks[sub].name+"_"+sub+".jpg"), looks[sub].getImageAverage())
                    else:
                        for image in looks[sub].getAllImages():
                            cv2.imwrite(os.path.join(looks[sub].path, image["name"]), image["img"])



if __name__ == "__main__":
    f = FaceBucketMaster()

    # f.takeFolder("/home/rovian/Desktop/aligned_images_DB/Abel_Pacheco")
    # f.addImage("/home/rovian/Documents/GitHub/head-pose-estimation/yourface1.jpg")
    
    # /home/rovian/Desktop/aligned_images_DB/
    # ./sets/

    f.getPeople("/home/rovian/Documents/GitHub/head-pose-estimation/self/", affine=True)