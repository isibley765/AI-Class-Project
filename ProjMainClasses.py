import os
import cv2
import random
import numpy as np
from copy import deepcopy
from pprint import pprint as pp

from utils.FaceFeatPos import GetFace
from utils.FacePathList import PathList

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
    
    def sampleRand40(self, list40):
        if len(list40) > 40:
            return [list40[i] for i in sorted(random.sample(range(len(list40)), 40))]
        else:
            return list40

    """
        Expects a directory labeled by name, with images inside, possibly in subfolders
        Moves images to local image repo with L/R/C categories for further processing
        Doesn't load locally and keep just in RAM, because that blows up into huge memory with large datasets

        affine depicts whether the re-sorting is done with original images, or
        scaled 256x256 images of just the face
    """
    def getPeople(self, path, affine=False, avg=False):
        if avg and not affine:
            avg = False
            pp("Can't average non-affine corrected images, skipping average")

        names = self.sampleRand40(sorted(os.listdir(path)))
        
        for name in names:
            namepath = os.path.join(path, name)
            if os.path.isdir(namepath):
                images = self.sampleRand40(self.takeFolder(namepath))
                self.makeBins(name, images, affine, avg)
    
    """
        This function takes a folder named for the person it represents,
        and cycles through the named subfolders and saves the set in a new place, either with
        the original image, or the affine transformation 256x256 image that holds the centered face

        name is the name of the person being sampled
        imageList is a list of their images
        affine is boolean, whether to do an affine transformation or not
    """
    def makeBins(self, name, imageList, affine=False, avg=False):
        # both being true only takes affect if affine and average are true
        if avg and not affine:
            avg = False
            pp("Can't average non-affine corrected images, skipping average")
        
        bucket = os.path.join(os.path.join(os.getcwd(), "trial"))
        namebucket = os.path.join(bucket, name)

        bins = ["SL", "SR", "L", "R", "C", "HL", "HR"]

        predef = {
            "name": {"path": namebucket},
            "spectrums": {"path": os.path.join(namebucket, "spectrums")},
            "step": 0.15
        }

        for key in predef.keys():
            if key != "step":
                for b in bins:
                    predef[key][b] = PathList(os.path.join(predef[key]["path"], b), name=name)
            """
                EX:
                {
                    "path": namebucket
                    "SL": PathList(os.path.join(namebucket, "SL"), name=name),   # Slight left or right
                    "SR": PathList(os.path.join(namebucket, "SR"), name=name),
                    "L":  PathList(os.path.join(namebucket, "L"), name=name),    # Normal left, right, or center
                    "R":  PathList(os.path.join(namebucket, "R"), name=name),
                    "C":  PathList(os.path.join(namebucket, "C"), name=name),
                    "HL": PathList(os.path.join(namebucket, "HL"), name=name),   # Hard left or right
                    "HR": PathList(os.path.join(namebucket, "HR"), name=name),
                }
            """

        pp("Making {}'s training folder".format(name))


        # These need to exist before we make the L, C, and R subfolders
        if not os.path.exists(bucket):
            os.makedirs(bucket)
        

        for key in predef.keys():
            if key != "step":
                if not os.path.exists(predef[key]["path"]):
                    os.makedirs(predef[key]["path"])

                for sub in predef[key].keys():
                    if sub != "path":
                        if not os.path.exists(predef[key][sub].path):
                            os.makedirs(predef[key][sub].path)

        
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
                if pose[1] < -predef["step"]*3:
                    out = "HL"
                elif pose[1] < -predef["step"]*2:
                    out = "L"
                elif pose[1] < -predef["step"]:
                    out = "SL"
                elif pose[1] > predef["step"]*3:
                    out = "HR"
                elif pose[1] > predef["step"]*2:
                    out = "R"
                elif pose[1] > predef["step"]:
                    out = "SR"
                else:
                    out = "C"

                if affine:
                    predef["name"][out].addImage(affout, image["name"])
                    predef["spectrums"][out].addImage(feat.warpedFaceSpectrum([affout])[0], image["name"])
                else:
                    cv2.imwrite(os.path.join(predef["name"][out].path, image["name"]), image["img"])
        
        if affine:
            genFace = GetFace(np.zeros((256, 256, 3), dtype=np.uint8))
            for sub in predef["name"].keys():
                if sub != "path":
                    if avg:
                        sumPic = predef["name"][sub].getImageAverage()
                        sumSpec = predef["spectrums"][sub].getImageAverage()
                        if type(sumPic) == np.ndarray:
                            cv2.imwrite(os.path.join(predef["name"]["path"], predef["name"][sub].name+"_"+sub+".jpg"), sumPic)
                            cv2.imwrite(os.path.join(predef["name"]["path"], predef["name"][sub].name+"_"+sub+"_spec.jpg"), genFace.warpedFaceSpectrum([sumPic])[0])
                            cv2.imwrite(os.path.join(predef["spectrums"]["path"], predef["spectrums"][sub].name+"_"+sub+".jpg"), sumSpec)

                    bwnorm = predef["name"][sub].getAllImages()
                    bwspec = predef["spectrums"][sub].getAllImages()
                    for i in range(len(bwnorm)):
                        cv2.imwrite(os.path.join(predef["name"][sub].path, bwnorm[i]["name"]), bwnorm[i]["img"])
                        cv2.imwrite(os.path.join(predef["spectrums"][sub].path, bwspec[i]["name"]), bwspec[i]["img"])



if __name__ == "__main__":
    f = FaceBucketMaster()

    # f.takeFolder("/home/rovian/Desktop/aligned_images_DB/Abel_Pacheco")
    # f.addImage("/home/rovian/Documents/GitHub/head-pose-estimation/yourface1.jpg")
    
    # /home/rovian/Desktop/aligned_images_DB/
    # ./sets/

    f.getPeople("./sets/", affine=True, avg=True)