import cv2
import numpy as np
from scipy import spatial
import cPickle as pickle
import random
import os
import matplotlib.pyplot as plt

# Image key point location generator
def makeImageCircles(kps, imgpath, img):
        imcpy = np.copy(img)
        print(imgpath)
        for kp in kps:
            pt = list(kp.pt)
            for i in range(len(pt)):
                pt[i] = int(round(pt[i]))
            cv2.circle(imcpy, tuple(pt), 2, (0,0,255), thickness=8, lineType=8, shift=0)
        cv2.imwrite("./features/"+imgpath, imcpy)

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = cv2.imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Finding image keypoints
        kps = alg.detect(image)
        print(kps[0].pt) #'angle', 'class_id', 'convert', 'octave', 'overlap', 'pt', 'response', 'size'
        
        makeImageCircles(kps, image_path, image) # Identifying the location of each keypoint
        # Possible to search against alternative keypoints on other image and find most likely match?


        # Getting first 32 of them. 
        # Number of keypoints varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        #kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
    except cv2.error as e:
        print 'Error: ', e
        return None

    return image, kps,dsc


if __name__ == "__main__":
    img1, kps1, dsc1 = extract_features("./textures/starrynight.jpg")
    img2, kps2, dsc2 = extract_features("./images/white_tiger.jpg")

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(dsc1, dsc2)

    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, kps1, img2, kps2, matches, img_matches)

    cv2.imwrite("./features/starrytiger.jpg", img_matches)