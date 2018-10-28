from pprint import pprint as pp
from copy import deepcopy
import numpy as np
import argparse
# import imutils
import math
import dlib
import cv2


CNN_INPUT_SIZE = 128

class getFace:  # Presets based on operating in the parent folder of utils, despite this file being located in it
    # Points based on the outter eye corners and bottom chin point for the default predictor model, please update both together
    def __init__(self, image_face=None, predict="./utils/shape_predictor_68_face_landmarks.dat", model="./assets/model.txt", points=[8, 36, 45]):
        if type(image_face)==np.ndarray:
            self.orig_size = (len(image_face[0]), len(image_face))
            self.image_face = image_face # cv2.resize(image_face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        # pp(image_face)
        
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
        
        """
        image = np.zeros([size, size, 3], dtype=np.uint8)
        for point in model:
            x = point[0]
            y = point[1]
            cv2.circle(image, (x, y), 1, (255, 0, 0), 2)
        #cv2.rectangle(image, (l, t), (l+w, t+h), (0, 255, 0), 2)

        self.show(image)
        """

        return model
    
    def warpFaceFront(self):
        trio = np.array([[x[0], x[1]] for x in self.trioAdjust()], dtype=np.float32)
        
        for i in range(0, len(self.feats)):
            w = self.rects[i].width()
            h = self.rects[i].height()
            center = self.rects[i].center()
            pp(self.image_face.shape)


            wExtend = (w * 256/180) / 2
            hExtend = (h * 256/180) / 2

            pt1 = (center.x - wExtend, center.y - hExtend)
            pt2 = (center.x + wExtend, center.y + hExtend)

            chunk = self.grey[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            chank = self.image_face[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            """
            image = deepcopy(self.image_face)
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
            for point in self.feats[i]:
                x = point[0]
                y = point[1]
                cv2.circle(image, (x, y), 1, (255, 0, 0), 2)
            
            self.show(image)
            """
            self.show(chunk)


            found3 = self.feats[i][self.indexTrio]
            pp(found3)
            pp(trio)

            affineM = cv2.getAffineTransform(found3, trio)
            pp(affineM)
            dst = cv2.warpAffine(chunk, affineM, (pt2[1]-pt1[1], pt2[0]-pt1[0]))

            self.show(dst)

    


    # TODO: Delete this method
    # Line variant of box method, by Ian Sibley, baed on the head-pose-estimation github library
    # Used only for visual appraisal during testing, not for algorithm completion
    def draw_annotation_line(self, image, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        image = deepcopy(image)

        point_3d = []
        rear_size = 0
        rear_depth = 0
        point_3d.append((0, 0, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((0, 0, front_depth))

        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        if not self.rvec:
            self.findAngle()
            pp("Finding angle")

        for i in range(0, len(self.rvec)):
            # Map to 2d image points
            (point_2d, _) = cv2.projectPoints(point_3d,
                                            self.rvec[i],
                                            self.tvec[i],
                                            self.camera,
                                            np.zeros((4, 1)))
            point_2d = np.int32(point_2d.reshape(-1, 2))

            # Draw all the lines
            cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)

        return image

        # Line variant of box method, by Ian Sibley, baed on the head-pose-estimation github library
    def draw_angle_line(self, image, angle, start, color=(255, 255, 255), line_width=2):
        rad = angle * 100 # rad = math.radians(angle*100)        
        pointa = start
        pointb = [pointa[0]+50*math.sin(rad), pointa[0]+50*math.cos(rad)]
        point_2d = np.asarray([pointa, pointb], dtype=np.int32)
        pp(point_2d)

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)

        return image
    
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
    img = cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/twoface.jpg")
    face = getFace(img)
    face.warpFaceFront()
    """

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
        face = getFace(img)
        pose = face.findAngle()[0]
        print("u/d: {}\nl/r: {}\n".format(pose[0], pose[1]))
        face.show(face.draw_annotation_line(img, color=(255, 0, 0)))
    """