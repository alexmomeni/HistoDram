import numpy as np
from keras.preprocessing.image import ImageDataGenerator
            
            
class Generator(object):
    
    def __init__(self, config, dataset):
        
        self.config = config
        self.dataset = dataset
        self.training_ids = self.dataset._partition[0]['train']
        self.training_labels = np.array(self.dataset._partition[1]['train']).flatten()
        print('Training set has %s patients' % len(self.training_ids))
   
    def generate(self):
        'Generates batches of samples'
        self.t_indexes = {}
        self.t_batches = {}
        
        for i in range(self.config.num_classes):
            self.t_indexes[i]  = np.argwhere(self.training_labels == i).flatten()
            self.t_batches[i] = np.array(self.training_labels)[self.t_indexes[i]].flatten()
            imax = int(len(self.training_labels)/ self.config.batch_size)

        for j in range(imax): 
            indexes = np.concatenate([np.random.choice(self.t_indexes[i], int(self.config.batch_size/self.config.num_classes), replace = False) for i in range(self.config.num_classes)])
            self.batch_labels = self.training_labels[indexes]
            self.batch_ids = [self.training_ids[i] for i in indexes]
            print(self.batch_ids)
            print(self.batch_labels)
            X, y = self.__data_generation(self.batch_ids)
            X = self.__data_augmentation(X) 
            return X, y
        

    def __data_generation(self, list_IDs_temp):
        X, y = self.dataset.convert_to_arrays(samples = list_IDs_temp, size = 1)
        return X, y
    

    def __data_augmentation(self, X):    
        augmented_images = []
        for i in range(len(X)):
            x = X[i] 
            switch = np.random.randint(2)
            k = np.random.randint(4)
            if switch == 0:
                x_rotated = np.rot90(x, k = k)
                X[i] = x_rotated 
            else:
                if k == 0 :
                    x_flipped = x
                if k ==1:
                    x_flipped = np.fliplr(x)
                if k == 2:
                    x_flipped = np.fliplr(x)
                if k ==3:
                    x_flipped = x
                X[i] = x_flipped                
        return X 