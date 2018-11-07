from pprint import pprint as pp
from copy import deepcopy
import numpy as np
from scipy import ndimage
import face_recognition as fr
import math
import cv2
import os

class TrainSet: # Gets the normal images in imageBins, and then does averaging, filtering and Fourier transforms to fill the rest
    def __init__(self, path=None, name=None):
        self.path = path
        self.name = name

        self.buckets = ["SL", "SR", "L", "R", "C", "HL", "HR"]

        self.imageBins = {}
        self.specBins = {}

        self.snormed = {}

        self.averages = {
            "norm": {}, # averages of imageBins
            "nspec": {}, # fourier of norm averages
            "spec": {}, # average of specBins
            "high": {}, # average of highFilter
            "low": {}, # average of lowFilter
            "normhi": {}, # average of normhi
            "normlow": {} # average of normlow
            }

        self.hiFiler = {}
        self.lowFilter = {}

        self.normhi = {}
        self.normlow = {}

        self.refreshBins()

    def refreshBins(self):    
        for buck in self.buckets:
            self.imageBins[buck] = []
            self.specBins[buck] = []

            self.snormed[buck] = []

            for avg in self.averages:
                self.averages[avg][buck] = []

            self.hiFiler[buck] = []
            self.lowFilter[buck] = []


            self.normhi[buck] = []
            self.normlow[buck] = []
        
    def getImageFolder(self, path=None, name=None):  # Expects self.buckets values to be present in presented folder
        if path is None:
            if self.path is None:
                raise ValueError("Can't get images without knowing where they are")
            else:
                path = self.path
        
        if not name is None:
            self.name = name    # Assume user responsible for correction
        elif self.name is None:
            raise ValueError("Name must be associated with the image set")
        
        self.refreshBins()  # Only one set at a time

        for buck in self.buckets:   # Single layer folders, with names corresponding to self.buckets and images inside
            binPath = os.path.join(path, buck)

            for name in os.listdir(binPath):    # Find images in the bucket subfolders
                tryImage = os.path.join(binPath, name)
                if name.endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(tryImage)#, cv2.IMREAD_GRAYSCALE)
                    if not img is None:
                        self.imageBins[buck].append(img)
        """
        for image in os.listdir(path):    # Find average images in the folder itself
            tryImage = os.path.join(path, image)
            if image.endswith((".jpg", ".jpeg", ".png")): # If acceptable image type, expecting .jpg
                parts = image.split('.')
                for el in self.buckets: # Find the appropriate bucket
                    if parts[-2].endswith("_"+el):  # If a bucket fits, add it to the appropriate average
                        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                        if not img is None:
                            self.averages["norm"][el].append(img)
        """
        
        self.prepareRest()
    
    #Fills the rest of the bins
    def prepareRest(self):
        self.genAverage(self.imageBins, self.averages["norm"])

        self.genTft(self.imageBins, self.specBins)
        self.genAverage(self.specBins, self.averages["spec"], real=False)
        self.genTft(self.averages["norm"], self.averages["nspec"])

        self.genLow(self.imageBins, self.normlow)
        self.genAverage(self.normlow, self.averages["normlow"])

        self.genHigh(self.imageBins, self.normhi)
        self.genAverage(self.normhi, self.averages["normhi"])

        self.genTft(self.normlow, self.lowFilter)
        self.genAverage(self.lowFilter, self.averages["low"], real=False)
        
        self.genTft(self.normhi, self.hiFiler)
        self.genAverage(self.hiFiler, self.averages["high"], real=False)

        self.ftTgen(self.averages["spec"], self.snormed)
    
    # Apply Generic Average on A to B
    def genAverage(self, dictSrc, dictDest, real=True):
        for key in dictSrc.keys():
            if real:
                avg = self.getImageAverage(dictSrc[key])
            else:
                avg = self.getComplexAverage(dictSrc[key])
            if not avg is None:
                dictDest[key] = [avg]
    
    # Tested, works for normal inputs
    def getImageAverage(self, imgList):
        N = len(imgList)
        if len(imgList) != 0:
            avg = np.zeros(imgList[0].shape, np.float32)

            for image in imgList:
                avg += image.astype(np.float32) / N
            
            return np.round(avg).astype(np.uint8)
        else:
            return None
    
    # Tested, works for normal inputs
    def getComplexAverage(self, imgList):
        N = len(imgList)
        if len(imgList) != 0:
            avg = np.zeros(imgList[0].shape, imgList[0].dtype)

            for image in imgList:
                avg += image / N
            
            return avg
        else:
            return None
        
    # Apply General Fourier Transform from A to B
    def genTft(self, dictSrc, dictDest):
        for key in dictSrc.keys():
            ftImgs = self.ftImages(dictSrc[key])
            if not ftImgs is None:
                dictDest[key]= ftImgs
    
    # Apply Fourier Transform to list
    def ftImages(self, imageList):
        spectrum = []

        for image in imageList:
            spectrum.append(np.fft.fftshift(np.fft.fft2(image)))
        
        return spectrum

    # Works, I think
    def genLow(self, dictSrc, dictDest):
        for key in dictSrc.keys():
            lowImgs = self.lowPass(dictSrc[key])
            if not lowImgs is None:
                dictDest[key]= lowImgs
    
    # Works, I think
    def genHigh(self, dictSrc, dictDest):
        for key in dictSrc.keys():
            highImgs = self.highPass(dictSrc[key])
            if not highImgs is None:
                dictDest[key]= highImgs

    # gaussian filter on non-spectrum images appears most accurate for now
    def lowPass(self, imageList):
        lows = []
        for image in imageList:
            lows.append(ndimage.gaussian_filter(image, 3))
        
        return lows

    # gaussian filter on non-spectrum images appears most accurate for now
    def highPass(self, imageList):
        highs = []
        for image in imageList:
            low = ndimage.gaussian_filter(image, 3)
            highs.append(image - low)
        cv2.IMREAD_GRAYSCALE
        return highs

    # General Inverse Fourier Transform
    def ftTgen(self, dictSrc, dictDest):
        for key in dictSrc.keys():
            normed = self.iftImages(dictSrc[key])
            if not normed is None:
                dictDest[key] = normed

    # Inverse Fourier Transform
    def iftImages(self, imageList):
        normalized = []

        for image in imageList:
            normalized.append(np.abs(np.fft.ifft2(np.fft.ifftshift(image))))
        
        return normalized

    # Tries to check if image is Spectrum or Normal
    def show(self, image, delay=1000):
        if np.issubdtype(np.uint8, image.dtype):
            self._showImage(image, delay)
        else:
            self._showSpecImg(image, delay)

    # Converts complex Fourier into something plottable
    def _showSpecImg(self, image, delay=1000):
        self._showImage(np.asarray(20*np.log(np.abs(image)), dtype=np.uint8), delay)
        
    # General image display method
    def _showImage(self, image, delay=1000):
        if self.name is None:
            name = "unknown"
        else:
            name = self.name
        
        pp("None?: {}".format(image is None))
        cv2.imshow(name, image)
        cv2.waitKey(delay)
        cv2.destroyWindow(name)

    def compareImageToGroup(self, image, group=None):
        if group is None:
            group = self.imageBins
            pp("HEre")

        if type(image) is str:
            image = cv2.imread(image)
            
        out = []
        featIn = np.asarray(fr.face_encodings(image))
        #pp(featIn)

        for key in group.keys():
            for el in group[key]:
                featEl = np.asarray(fr.face_encodings(el))

                out.append(fr.compare_faces(featIn, featEl))
                pp(out[-1])

        return out

if __name__ == "__main__":
    folderSet = TrainSet(path="./trial/Ian_Sibley", name="Ian_Sibley")

    folderSet.getImageFolder()

    folderSet.compareImageToGroup("/home/rovian/Pictures/profile.jpg", folderSet.averages["norm"])

    """

    folder = "./spectral/"
    i = 0

    # Test that all images came through
    for el in folderSet.buckets:
        pp(el)
        
        for image in folderSet.averages["nspec"][el]:
            cv2.imwrite(os.path.join(folder, "image_nspec{}.jpg".format(i)), 20*np.log(np.abs(image)))
            i += 1
        
        for image in folderSet.specBins[el]:
            cv2.imwrite(os.path.join(folder, "image_specBins{}.jpg".format(i)), 20*np.log(np.abs(image)))
            i += 1
        
        for image in folderSet.averages["spec"][el]:
            cv2.imwrite(os.path.join(folder, "spec{}.jpg".format(i)), 20*np.log(np.abs(image)))
            i += 1
            
        for image in folderSet.snormed[el]:
            folderSet.show(image, 500)
            
        for image in folderSet.lowFilter[el]:
            folderSet.show(image, 500)

        for image in folderSet.hiFiler[el]:
            folderSet.show(image, 500)


        for image in folderSet.hiFiler[el]:
            folderSet.show(image)

        for image in folderSet.lowFilter[el]:
            folderSet.show(image)
            
        for image in folderSet.imageBins[el]:
            folderSet.show(image)
        
        for image in folderSet.averages["norm"][el]:
            pp("Average {}:".format(el))
            folderSet.show(image)
    """