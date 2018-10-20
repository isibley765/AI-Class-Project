# https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774


import cv2
import numpy as np
from scipy import spatial
import cPickle as pickle
import random
import os
import matplotlib.pyplot as plt

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
        imcpy = np.copy(image)
        print(image_path)
        for kp in kps:
            pt = list(kp.pt)
            for i in range(len(pt)):
                pt[i] = int(round(pt[i]))
            cv2.circle(imcpy, tuple(pt), 2, (0,0,255), thickness=8, lineType=8, shift=0)
        cv2.imwrite("./features/"+image_path, imcpy)
        # Getting first 32 of them. 
        # Number of keypoints varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less than 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print 'Error: ', e
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print 'Extracting features from image %s' % f
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'w') as fp:
        pickle.dump(result, fp)

class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path) as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.iteritems():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

def show_img(path):
    img = cv2.imread(path)
    plt.imshow(img)
    plt.show()

def run():
    images_path = './images/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    textures_path = './textures/'
    textures = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    sample = random.sample(files, 1)
    # getting 3 random images 
    sample = random.sample(textures, 1)
    
    batch_extractor(images_path)
    batch_extractor(textures_path, pickled_db_path='textures.pck')

    ma = Matcher('features.pck')
    mt = Matcher('textures.pck')
    """
    for j in range(len(sample)):
        print 'Query image =========================================='
        show_img(sample[j])
        names, match = ma.match(sample[j], topn=3)
        print 'Result images ========================================'
        for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print 'Match %s' % (1-match[i])
            show_img(os.path.join(images_path, names[i]))
    """

run()

print(len(extract_features("./images/white_tiger.jpg")))