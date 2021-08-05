import os 
import os.path as osp
import matplotlib.pyplot as plt
import random
import xml.etree.cElementTree as ET
import cv2
import torch.utils.data as data 
import numpy as np 
import torch 

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)