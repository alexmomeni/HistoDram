#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:26:55 2018

@author: alexmomeni
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from preprocess import Resize
from sklearn.model_selection import StratifiedShuffleSplit  

class Dataset(object):
    
    
    def __init__(self, config):
            self.config= config
            self._list_features = self.list_features()
            self._selected_features = self.output_data()
            self._binarized_data, self.le_name_mapping = self.binarized_data()
            self._ids = self.ids()
            self._labels = self.labels()
            self._partition = self.partition()
            

    def list_features(self):
        data = pd.read_excel(self.config.label_path, header=0, index_col=0)
        list_features = data.columns
        return list_features

    def output_data(self):
        data = pd.read_excel(self.config.label_path, header=0, index_col=0)
        selected_features = data[self.config.selected_features].dropna()
        return selected_features

    def binarized_data(self):
        data = self.output_data()
        le = LabelEncoder()
        binarized_data = data.apply(le.fit_transform)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        return binarized_data, le_name_mapping

    def ids(self):
        patient_ids = os.listdir(self.config.temp_path)
        target_ids = set(self.output_data().index)
        ids = list(target_ids.intersection(patient_ids))
        return ids

    def labels(self):
        labels = {}
        samples = self.ids()
        data = self.binarized_data()[0]
        for feature in self.config.selected_features:
            labels[feature] = {}
            for sample in samples:
                labels[feature][sample] = data.loc[sample, feature]

        return labels

    def partition(self):
        data = pd.DataFrame(self._labels)
        ids = data.index
        labels = data[self.config.selected_features].values
        
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size= self.config.test_size)
        sss_test.get_n_splits(ids, labels)

        for train_index, test_index in sss_test.split(ids, labels):
            ids_train, ids_test = ids[train_index], ids[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        
        ids, labels = ids_train,  y_train
        

        sss_val = StratifiedShuffleSplit(n_splits=1, test_size= self.config.val_size)
        sss_val.get_n_splits(ids, labels)
        
        for train_index, test_index in sss_val.split(ids, labels):
            ids_train, ids_val = ids[train_index], ids[test_index]
            y_train, y_val = labels[train_index], labels[test_index]

        print(self.le_name_mapping)
        
        #for i in range(len(np.unique(y_val))):
        #    print(i,len((y_val[y_val == i]))/len(y_val))
        #for i in range(len(np.unique(y_train))):
        #    print(i,len((y_train[y_train == i]))/len(y_train)) 
        
        names = list(self.le_name_mapping)
        y_val_split = []
        y_train_split = []
        y_test_split = []

        for i in range(len(np.unique(y_val))):
            y_val_split.append( len(y_val[y_val == i]))
        
        for i in range(len(np.unique(y_train))):
            y_train_split.append(len(y_train[y_train == i]))
        
        for i in range(len(np.unique(y_train))):
            y_test_split.append(len(y_test[y_test == i]))
        
        split = pd.DataFrame(list(zip(y_train_split,y_val_split, y_test_split)), columns = ['train','val', 'test'] , index = names)
        split.to_csv(self.config.logdir + '/train_val_split.csv')
            
        partition_ids = {'train': list(ids_train), 'val': list(ids_val), 'test': list(ids_test)}
        partition_labels = {'train': list(y_train), 'val': list(y_val), 'test': list(y_test)}
        partition_labels = {'train': list(y_train), 'val': list(y_val), 'test': list(y_test)}

        return partition_ids, partition_labels

    def convert_to_arrays(self, samples, size = None):

        Image.MAX_IMAGE_PIXELS = 1000000000

        X, ids = [], []
        y = []
        

        for sample in samples:
            try:
                patches = os.listdir("%s/%s" % (self.config.temp_path, sample))
                
                if size != None:
                    patches = np.random.choice(patches, size= size, replace=True)
                
                for patch in patches:
                    ID = "%s/%s/%s" % (self.config.temp_path, sample, patch)
                    ids.append(ID)
                    img = Image.open(ID)
                    img = np.array(img)[:, :, :3]
                    X.append(img)
            except Exception as e:
                print(e)
                pass
        
        X = np.asarray(X)
        
        for label in self.labels().keys():
            y_label = []    
            for ID in ids:
                sample = ID.split('/')[-2]
                y_label.append(self.labels()[label][sample])
            y_label = np.asarray(y_label)
            y.append(y_label)
        y = np.asarray(y)
        return X, y
    
    def next_batch(self, images, labels, batch_size, step):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        
        start = step * batch_size
        end = (step+1) * batch_size
        if end > self._num_examples:
            pass
        else: 
            return self._images[start:end], self._labels[start:end]
            
        