from pprint import pprint as pp
from copy import deepcopy
import numpy as np
import argparse
# import imutils
import dlib
import cv2

class getFace:
    def __init__(self, image_face=None, predict="./utils/face_landmarks_68.dat"):
        if type(image_face)==np.ndarray:
            self.image_face = image_face
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predict)

        self.grey = cv2.cvtColor(self.image_face, cv2.COLOR_BGR2GRAY)
    
    def getFeatures(self):
        rects = self.face_detector(self.grey, 2)
        image = deepcopy(self.image_face)
        for rect in rects:
            # (x, y, w, h) = (rec.left(), rec.top(), rec.right() - rec.left(), rec.bottom() - rec.top())
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            points = [[point.x, point.y] for point in self.predictor(self.grey, rect).parts()]

            for point in points:
                x = point[0]
                y = point[1]
                cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

            self.show(image)

        pp(rects)
        return points
    
    def show(self, image):
            cv2.imshow("Output", image)
            cv2.waitKey(50000)
            cv2.destroyWindow("Output")



if __name__ == "__main__":
    try:
        face = getFace("/home/rovian/Documents/GitHub/head-pose-estimation/self/yourface0.jpg")
    except ValueError as v:
        pp(type(cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/twoface.jpg")))
        face = getFace(cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/twoface.jpg"))

    pp(face.getFeatures())