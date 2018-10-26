from pprint import pprint as pp
import numpy as np
import argparse
# import imutils
import dlib
import cv2

class getFace:
    def __init__(self, image_face=None):
        if type(image_face)==np.ndarray:
            self.image_face = image_face
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.grey = cv2.cvtColor(self.image_face, cv2.COLOR_BGR2GRAY)
    
    def getFeatures(self):
        rects = self.face_detector(self.grey, 1)

        pp(rects)
        return rects



if __name__ == "__main__":
    try:
        face = getFace("/home/rovian/Documents/GitHub/head-pose-estimation/yourface0.jpg")
    except ValueError as v:
        pp(v)
        face = getFace(cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/yourface0.jpg"))

    pp(face.getFeatures())