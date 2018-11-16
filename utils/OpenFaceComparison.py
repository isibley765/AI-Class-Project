from pprint import pprint as pp
from copy import deepcopy
import numpy as np
import dlib
import cv2

import os
import json
import random

import openface

class FaceCompariosonGuru:
    def __init__(self, networkModel=None, imgDim=None, align=None):
        self.align = openface