#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:29:57 2018

@author: alexmomeni
"""

from multiprocessing import Pool
import time
import os
from openslide import * 
import numpy as np 
from config import Config
import cv2
from skimage import io
from skimage.measure import label, regionprops
import pandas as pd
from tqdm import tqdm
from PIL import Image 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Preprocess(object):
      
    
    def get_ids (self, config):
        
        self.config = config
        patient_ids = os.listdir(self.config.source_path)
        patient_ids = [patient_id[:-4] for patient_id in patient_ids]
        target_ids = set(pd.read_excel(self.config.label_path,header=0,index_col=0).index)
        ids =  list(target_ids.intersection(patient_ids))
            
        return ids
    
    def tissue_segmentation(self, config, patient_id, save_segmentation = True, save_tissues = True, plot = False):
        
        print(patient_id)

        self.config = config
        if os.path.isdir('%s/%s' % (self.config.temp_path,patient_id)):
                print ("Already processed")
                return
        
        os.makedirs('%s/Segmentation' % self.config.temp_path, exist_ok=True)
    

        os.makedirs('%s/%s' % (self.config.temp_path,patient_id), exist_ok=True)
        img, labeled = self.otsu_thresholding(config, patient_id)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        patches = []

        for region_index, region in enumerate(regionprops(labeled)):

            if region.area >  ((img.shape[0]*img.shape[1])//100):   
                if region.area > 1000000:
                    minr, minc, maxr, maxc = region.bbox
                    patches.append(img[minr:maxr, minc:maxc])
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        
        if save_segmentation == True:
            plt.savefig('%s/Segmentation/%s.jpg' % (self.config.temp_path,patient_id))
        
        if plot == True:
            plt.show()

        if save_tissues == True:
            for c, patch in enumerate(patches):
                if c:
                    io.imsave('%s/%s/%s.jpg'% (self.config.temp_path,patient_id, c), patch)  
                else:
                    pass
        return

    
    def otsu_thresholding (self, config, patient_id, plot = False, level = 1):
        
        
        self.config = config
        img = OpenSlide("%s/%s.svs" %(self.config.source_path,patient_id))
        img = img.read_region(location = (0,0), level = level, size = img.level_dimensions[level]).convert('RGB')
        img = np.array(img)[:,:,:3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 100,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_c = np.ones((75,75), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
        kernel_o = np.ones((75,75), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_o)
        labeled = label(opening)
        
        if plot == True:
            fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
            ax0, ax1, ax2, ax3 = ax.ravel()
            plt.tight_layout()
            
            ax0.imshow(img)
            ax0.set_title('Original')
            ax0.axis('off')
            
            ax1.imshow(gray, cmap=plt.cm.gray)
            ax1.set_title('Gray')
            ax1.axis('off')
            
            ax2.imshow(thresh, cmap=plt.cm.gray)
            ax2.set_title('Otsu')
            ax2.axis('off')
            ax3.imshow(closing, cmap=plt.cm.gray)
            ax3.set_title('Opening/Closing')
            ax3.axis('off')
            
            plt.show()
        
        return img, labeled

    
    def get_all_tissue_segments(self, config, processes = 30):
        
        patient_ids = self.get_ids(config)
        p = Pool(processes)
        p.starmap(self.tissue_segmentation, [[config,patient_id] for patient_id in patient_ids])
        print("Preprocessing done")

        
class Resize(object):
    
    def resize_image_with_crop_or_pad(self, image, config):
        
        self.config = config
        
        height, width, channels = image.shape    
    
        cropped = self.crop_to_bounding_box(image,
                                       min(config.input_shape, height),
                                       min(config.input_shape, width))
    
        resized = self.pad_to_bounding_box(cropped, config.input_shape, config.input_shape)

        return resized

    
    def crop_to_bounding_box(self, image, target_height,
                             target_width):
        
        height, width, channels = image.shape
    
        cropped = np.array(image)[(height - target_height)//2: (height + target_height)//2,
                          (width- target_width)//2: (width+ target_width)//2,:3]
    
        return cropped
    
    
    def pad_to_bounding_box(self, image, target_height,
                            target_width):
        
        height, width, channels = image.shape
        img = np.zeros((target_height, target_width,3), dtype= np.uint8) + 255
        
        if (target_height-height) % 2 == 0:
            lb_height = (target_height-height)//2
            ub_height = target_height- lb_height
        else: 
            lb_height = (target_height-height)//2 +1
            ub_height = target_height- lb_height + 1
            
        if (target_width-width) % 2 == 0:
            lb_width = (target_width-width)//2
            ub_width = target_width- lb_width
        else: 
            lb_width = (target_width-width)//2 +1
            ub_width = target_width - lb_width + 1
        
        
        img[lb_height : ub_height ,lb_width : ub_width ] = image[:,:]
        
        padded = img 
        
        return padded