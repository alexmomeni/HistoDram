import matplotlib
matplotlib.use('Agg')
from multiprocessing import Pool
from config import Config 
import os 
import numpy as np
import cv2
from openslide import *
from scipy.ndimage.morphology import binary_dilation
import pandas as pd
import shutil
import matplotlib.pyplot as plt


def get_patches(patient_id, config):
    config.patch_size = 4096
    if os.path.isdir("/labs/gevaertlab/data/momena/patches_4096/%s"%patient_id):
        print ("sample already processed")
        return
    if os.path.isdir("/labs/gevaertlab/data/momena/temp/%s"% patient_id):
        shutil.rmtree("/labs/gevaertlab/data/momena/temp/%s"% patient_id)
    os.makedirs("/labs/gevaertlab/data/momena/temp/%s"% patient_id)
    img = OpenSlide("/labs/gevaertlab/data/cedoz/slides/%s.svs"% patient_id)
    width, height = img.dimensions
    idx = 0
    for i in range(int(height/config.patch_size)):
        print ("iteration %d out of %d"%(i+1,int(height/config.patch_size)))
        for j in range(int(width/config.patch_size)):
            patch = img.read_region(location=(j*config.patch_size,i*config.patch_size), level=0,
                                    size=(config.patch_size,config.patch_size)).convert('RGB')
            array = np.array(patch)[:,:,:3]    
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = binary_dilation(thresh, iterations=15)
            ratio = np.mean(thresh)
            if ret < 200 and ratio > 0.90:
                patch.save("/labs/gevaertlab/data/momena/temp/%s/%s.jpg"% (patient_id, idx))
                idx += 1
    shutil.move("/labs/gevaertlab/data/momena/temp/%s"% patient_id, "/labs/gevaertlab/data/momena/patches_4096/%s"% patient_id)

def get_all_patches(config, processes=30):
    
    patient_ids = os.listdir("/labs/gevaertlab/data/cedoz/slides/")
    patient_ids = [patient_id[:-4] for patient_id in patient_ids]    
    p = Pool(processes)
    p.starmap(get_patches, [(patient_id, config) for patient_id in patient_ids])

config = Config()
get_all_patches(config)