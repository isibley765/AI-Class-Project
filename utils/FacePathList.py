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
        if len(self.images) != 0:
            avg = np.zeros(self.shape, np.float32)

            N = len(self.images)
            for image in self.images:
                avg += image["img"].astype(np.float32) / N
            
            return np.round(avg).astype(np.uint8)
        
        else:
            return None