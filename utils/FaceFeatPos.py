from pprint import pprint as pp
from copy import deepcopy
# import face_recognition
import numpy as np
import argparse
import math
import dlib
import cv2


CNN_INPUT_SIZE = 128

class GetFace:  # Presets based on operating in the parent folder of utils, despite this file being located in it
    # Points based on the outter eye corners and bottom chin point for the default predictor model, please update both together
    def __init__(self, image_face=None, predict="./assets/shape_predictor_68_face_landmarks.dat", model="./assets/model.txt", points=[8, 36, 45]):
        """
            Very accurate but slow:
                ./assets/shape_predictor_68_face_landmarks.dat
                
            Faster, relatively accurate points, not accurate enough for angle finding:
                ./assets/face_landmarks_68.dat
        """
        
        if type(image_face)==np.ndarray:
            self.orig_size = (len(image_face[0]), len(image_face))
            self.image_face = image_face # cv2.resize(image_face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        
        self.fullmodel = np.loadtxt(model, delimiter=",", dtype=np.float32)
        self.indexTrio = points
        self.cropScale = 256/180.0  # for later, using hard values below for now
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predict)

        self.grey = cv2.cvtColor(self.image_face, cv2.COLOR_BGR2GRAY)

        self.rvec = None
        self.tvec = None
        
        """
            Some quick calculations for a "camera array"
            The basic presets based on image dimensions
            solvePnPRansic also fine tunes for accuracy
            Most camera matrixes are negligible for solvePnP processes
        """
        size = (len(self.image_face), len(self.image_face[0]))
        # pp((len(self.image_face), len(self.image_face[0])))
        focus = size[1]
        center = (size[1] / 2, size[0] / 2)
        self.camera = np.array(
            [[focus, 0, center[0]],
            [0, focus, center[1]],
            [0, 0, 1]], dtype="double")

        self.feats = self.getFeatures()
    
    def getFeatures(self):  # For each face found (rect), find the designated points and return them
        feats = []

        self.rects = self.face_detector(self.grey, 2)
        
        """
        image = deepcopy(self.image_face)
        for point in self.fullmodel:
            x = point[0]
            y = point[1]
            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)
        cv2.rectangle(image, (64, 64), (570, 380), (0, 255, 0), 2)
        """

        for rect in self.rects:

            points = np.asarray([[point.x, point.y] for point in self.predictor(self.grey, rect).parts()], dtype=np.float32)
            feats.append(points)
            
            """
            image = deepcopy(self.image_face)
            (x, y, w, h) = (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for point in points:
                x = point[0]
                y = point[1]
                cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

            self.show(image)
            """


        return feats
    
    def findAngle(self):
        self.rvec = []
        self.tvec = []

        for i in range(0, len(self.rects)):
            model = self.fullmodel # self.modelAdjust(self.rects[i])
            rvec = []
            tvec = []
            _, rvec, tvec = cv2.solvePnP(
                model, self.feats[i], self.camera, np.zeros((4, 1)))


            _, rvec, tvec, _ = cv2.solvePnPRansac(
                model, self.feats[i], self.camera, np.zeros((4, 1)),
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True)
            
            self.rvec.append(np.asarray([angle[0] for angle in rvec]))
            self.tvec.append(np.asarray([angle[0] for angle in tvec]))

        return self.rvec
    
    def modelAdjust(self, size=256): #   Only aimed to center the model on the image, with linear transforms might be useless
        model = deepcopy(self.fullmodel)
        # left, top, width, height
        # (l, t, w, h) = (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

        model[:, 0] = model[:, 0] * size/180 + size/2
        model[:, 1] = model[:, 1] * size/180 + size/2
        
        image = np.zeros([size, size, 3], dtype=np.uint8)
        for point in model:
            x = point[0]
            y = point[1]
            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)
        #cv2.rectangle(image, (l, t), (l+w, t+h), (0, 255, 0), 2)

        for index in self.indexTrio:
            cv2.circle(image, tuple(model[index][:2]), 1, (0, 0, 255), 2)

        self.show(image)

        return model
    
    def trioAdjust(self, size=256): #   Only aimed to center the model on the image, with linear transforms might be useless
        model = self.fullmodel[self.indexTrio]

        model[:, 0] = model[:, 0] * size/180 + size/2
        model[:, 1] = model[:, 1] * size/180 + size/2

        return model
    
    def warpFaceFront(self, size=256):
        res = []
        
        trio = np.array([[x[0], x[1]] for x in self.trioAdjust(size)], dtype=np.float32)
                
        for i in range(0, len(self.feats)):
            found3 = self.feats[i][self.indexTrio]
                
            affineM = cv2.getAffineTransform(found3, trio)  # maps found features to square image of lengths = size
            
            dst = cv2.warpAffine(self.grey, affineM, (size, size))

            res.append(dst)
        
        return res

    def warpedFaceSpectrum(self, images=None):
        if images is None:
            images = self.warpFaceFront()
        
        spectrum = []
        for image in images:
            s = np.asarray(20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))), dtype=np.uint8)
            spectrum.append(s)

        return spectrum

    
    def show(self, image=None):
        try:    # Just in case made equal to undefined value?
            image
        except NameError:
            image = None
        
        
        if type(image) == None:
            image = self.image_face
        
        cv2.imshow("Output", image)
        cv2.waitKey(20000)
        cv2.destroyWindow("Output")



if __name__ == "__main__":
    """
    img = cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/twoface.jpg")
    face = GetFace(img)
    for img in face.warpedFaceSpectrum():
        face.show(img)

    # cv2.circle(img, tuple(face.feats[0][33]), 1, (0, 0, 255), 2) # 33'd index is the tip of the nose
    face.show(face.draw_annotation_line(img, color=(255, 0, 0)))
    face.show(img)
    feats = face.getFeatures()[0]

    for point in feats:
        x = point[0]
        y = point[1]
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)

    face.show(img)
    cv2.imwrite("rockout.jpg", img)
    pp(feats)
    

    for i in range(0, 9):
        img = cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/Ian_Sibley/yourface{}.jpg".format(i))
        face = GetFace(img)
        pose = face.findAngle()[0]
        print("u/d: {}\nl/r: {}\n".format(pose[0], pose[1]))
        face.show(face.draw_annotation_line(img, color=(255, 0, 0)))
    """
    
    for i in range(0, 9):
        img = cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/Ian_Sibley/yourface{}.jpg".format(i))
        face = GetFace(img)
        for img in face.warpedFaceSpectrum():
            face.show(img)