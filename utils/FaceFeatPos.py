from pprint import pprint as pp
from copy import deepcopy
# import face_recognition
import numpy as np
import argparse
import math
import dlib
import cv2

import os
import json
import random


CNN_INPUT_SIZE = 128

class GetFace:  # Presets based on operating in the parent folder of utils, despite this file being located in it
    # Points based on the outter eye corners and bottom chin point for the default predictor model, please update both together
    def __init__(self, image_face=None, predict="./assets/shape_predictor_68_face_landmarks.dat", model="./assets/model.txt", points=[8, 36, 45], mpoints=[8, 36, 45]):
        """
            Very accurate but slow:
                predict="./assets/shape_predictor_68_face_landmarks.dat"
                points=[8, 36, 45]
                
            Faster, relatively accurate points, not accurate enough for angle finding:
                predict="./assets/face_landmarks_68.dat"
                points=[66, 30, 40]

            default model:
                model="./assets/model.txt"
                mpoints=[8, 36, 45]
        """
        
        if type(image_face)==np.ndarray:
            self.orig_size = (len(image_face[0]), len(image_face))
            self.image_face = image_face # cv2.resize(image_face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        else:
            raise ValueError("Please provide an image in the form of a numpy array find its face")
        
        self.feats = None
        self.rects = None
        self.fullmodel = np.loadtxt(model, delimiter=",", dtype=np.float32)
        self.indexTrio = points
        self.modelTrio = mpoints
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
        
        self.getFeatures()
    
    def getFeatures(self):  # For each face found (rect), find the designated points and return them
        if self.feats is None:
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

                model = self.modelAdjust(size=image.shape[0])
                for point in points:
                    x = point[0]
                    y = point[1]
                    cv2.circle(image, (x, y), 1, (0, 0, 255), 2)
                for point in model:
                    x = point[0]
                    y = point[1]
                    cv2.circle(image, (x, y), 1, (255, 0, 0), 2)
                
                for i in range(len(self.indexTrio)):
                    cv2.circle(image, tuple(points[self.indexTrio[i]][:2]), 1, (0, 255, 0), 2)
                    self.show(image)
                    
                    cv2.circle(image, tuple(model[self.modelTrio[i]][:2]), 1, (0, 255, 0), 2)
                    self.show(image)

                """
            self.feats = feats

        return self.feats
    
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
        
        """
        image = np.zeros([size, size, 3], dtype=np.uint8)
        for point in model:
            x = point[0]
            y = point[1]
            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)
        #cv2.rectangle(image, (l, t), (l+w, t+h), (0, 255, 0), 2)

        for index in self.modelTrio:
            cv2.circle(image, tuple(model[index][:2]), 1, (0, 0, 255), 2)

        self.show(image)
        """

        return model
    
    def trioAdjust(self, size=256): #   Only aimed to center the model on the image, with linear transforms might be useless
        model = self.fullmodel[self.modelTrio]

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
    
    def faceFront(self, size=256):
        res = []
        
        for i in range(len(self.rects)):
            w = self.rects[i].right() - self.rects[i].left()
            h = self.rects[i].bottom() - self.rects[i].top()

            wn = (w * size/180)/2
            hn = (h * size/180)/2

            center = [self.rects[i].center().x, self.rects[i].center().y]
            tlcorner = [center[0]-wn, center[1]-hn]
            brcorner = [center[0]+wn, center[1]+hn]

            image = self.image_face[tlcorner[1]:brcorner[1],tlcorner[0]:brcorner[0]]
            if type(image) is np.ndarray and image.size > 0:
                image = cv2.resize(image, (256, 256))

                res.append(image)
        
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
        cv2.waitKey(2000)
        cv2.destroyWindow("Output")




def mapImages(path="/home/rovian/Documents/GitHub/neural/sets/Barbara_Walters"):
    
    
    out = []
    pp(path)

    a = os.listdir(path)
    if len(a) > 40:
        a = [a[i] for i in sorted(random.sample(range(len(a)), 40))]

    for name in a:
        tryName = os.path.join(path, name)
        if os.path.isdir(tryName) and name != "spectrums":
            out.extend(mapImages(tryName))
        else:
            if name.endswith("_spec.jpg"):
                continue
            elif name.endswith(".jpg"):
                image = cv2.imread(tryName)
                face = GetFace(image)
                
                template = face.faceFront()
                if len(template) == 1:
                    template = template[0]
                    face.image_face = deepcopy(template)
                    face.feats = None
                    face.rects = None
                    face.grey = cv2.cvtColor(face.image_face, cv2.COLOR_BGR2GRAY)

                    feats = face.getFeatures()
                    if len(feats) == 1:
                        feats = feats[0][face.indexTrio]

                        t = [ [int(feat[0]), int(feat[1])] for feat in feats]

                        out.append({"img": template, "trio": t})
                    else:
                        pp("Cropped {} found {} faces?".format(tryName, len(feats)))
                else:
                    pp("{} found {} faces?".format(tryName, len(template)))

                """
                feats = face.getFeatures()
                if len(feats) == 1:
                    feats = feats[0]
                    feats = [[ int(feat) for feat in feats[x]] for x in face.indexTrio]

                    out.append({"image": tryName, "pointTrio": feats})
                else:
                    pp("{} found {} faces?".format(tryName, len(feats)))
                """
    return out

def getFolderImageTrios(folder="/home/rovian/Documents/GitHub/neural/trial/", name="anon", end="train", mode='w'):
    out = None
    tpath = "./train/"
    if not os.path.exists(tpath):
        os.makedirs(tpath)

    out = mapImages(path=folder)


    for i in range(len(out)):
        imgName = os.path.join(tpath, name+"image{}".format(i)+end+".jpg")
        cv2.imwrite(imgName, out[i]["img"])
        out[i]["img"] = imgName

    if mode == 'a':
        with open(os.path.join(tpath, end+".json"), "r") as file:
            x = json.loads(file.read())
        out = out + x

    with open(os.path.join(tpath, end+".json"), "w") as file:
        file.write(json.dumps(out, indent=4))

if __name__ == "__main__":
    start = "/home/rovian/Downloads/frame_images_DB/"
    a = os.listdir(start)
    if len(a) > 40:
        a = [a[i] for i in sorted(random.sample(range(len(a)), 40))]
    outType = 'w'

    for name in a:
        getFolderImageTrios(folder=os.path.join(start, name), name=name, end="train_big_norm", mode=outType)
        outType = 'a'
    """
    im = cv2.imread("./smallset/Ian_Sibley/C/yourface0.jpg")
    face = GetFace(im)
    
    feats = face.faceFront()

    for feat in feats:
        pp
        for point in feat["trio"]:
            cv2.circle(feat["img"], tuple(point), 1, (0, 0, 255), 2) # 33'd index is the tip of the nose
        face.show(feat["img"])

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
    
    img = cv2.imread("/home/rovian/Documents/GitHub/neural/bigset/Aaron_Guiel/HL/aligned_detect_5.1979.jpg")
    face = GetFace(img, predict="./assets/face_landmarks_68.dat", points=[66, 30, 40])
    
    for img in face.warpFaceFront():
        face.show(img)
    
    for i in range(0, 9):
        img = cv2.imread("/home/rovian/Documents/GitHub/head-pose-estimation/self/Ian_Sibley/yourface{}.jpg".format(i))
        face = GetFace(img, predict="./assets/face_landmarks_68.dat", points=[66, 30, 40])
        
        for img in face.warpFaceFront():
           face.show(img)
    """