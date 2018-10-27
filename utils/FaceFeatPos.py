from pprint import pprint as pp
from copy import deepcopy
import numpy as np
import argparse
# import imutils
import dlib
import cv2

class getFace:
    def __init__(self, image_face=None, predict="./utils/face_landmarks_68.dat", model="./assets/model.txt"):
        if type(image_face)==np.ndarray:
            self.image_face = deepcopy(image_face)
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        # pp(image_face)
        
        self.fullmodel = np.loadtxt(model, delimiter=",", dtype=np.float32)
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predict)

        self.grey = cv2.cvtColor(self.image_face, cv2.COLOR_BGR2GRAY)
        self.feats = None

        self.rvec = None
        self.tvec = None
    
    def getFeatures(self):  # For each face found (rect), find the designated points and return them
        if not self.feats:
            self.feats = []

            self.rects = self.face_detector(self.grey, 2)
            
            # image = deepcopy(self.image_face)

            for rect in self.rects:
                # (x, y, w, h) = (rec.left(), rec.top(), rec.right() - rec.left(), rec.bottom() - rec.top())
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                points = np.asarray([[point.x, point.y] for point in self.predictor(self.grey, rect).parts()])
                
                """
                for point in points:
                    x = point[0]
                    y = point[1]
                    cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

                self.show(image)
                """

                self.feats.append(points)

        return self.feats
    
    def findAngle(self):
        size = (len(self.image_face), len(self.image_face[0]))
        # pp((len(self.image_face), len(self.image_face[0])))

        focus = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera = np.array(
            [[focus, 0, center[0]],
            [0, focus, center[1]],
            [0, 0, 1]], dtype="double")
        

        if self.rvec is None:
            pp(self.image_face)
            (_, rotvec, transvec) = cv2.solvePnP(
                self.fullmodel, self.image_face, camera, np.zeros((4, 1)))
            self.rvec = rotvec
            self.tvec = transvec
        
        (_, rvec, tvec) = cv2.solvePnPRansac(
            self.fullmodel,
            self.image_face,
            camera,
            np.zeros((4, 1)),
            rvec=self.rvec,
            tvec=self.tvec,
            useExtrinsicGuess=True)
        
        pp(rvec, self.r_vec)

        return (rvec, tvec)
    
    def show(self, image):
        
        cv2.imshow("Output", image)
        cv2.waitKey(20000)
        cv2.destroyWindow("Output")



if __name__ == "__main__":
    img = np.float32(cv2.imread("/home/rovian/Documents/GitHub/neural/Faith/therock.jpg"))
    try:
        face = getFace(img)

        """
        feats = face.getFeatures()[0]

        for point in feats:
            x = point[0]
            y = point[1]
            cv2.circle(img, (x, y), 1, (0, 0, 255), 2)

        face.show(img)
        cv2.imwrite("rockout.jpg", img)
        pp(feats)
        """
        pp(face.findAngle())
    except ValueError as v:
        # pp(type(cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/twoface.jpg")))
        pp(v)
    except Exception as e:
        pp(e)