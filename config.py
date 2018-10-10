import os 
import shutil 
class Config(object):

    def __init__(self, source_path="/labs/gevaertlab/data/cedoz/slides",
                 temp_path="/labs/gevaertlab/data/momena/patches_4096",
                 label_path="/home/aamomeni/research/histo_dram/TCGA_Supp_Data.xlsx",
                 selected_features=['Study'],
                 array_path="/labs/gevaertlab/data/momena/patches_4096", logdir="/home/aamomeni/research/histo_dram/LGGGBM_2/",
                 val_size=0.125, test_size=0.20, num_glimpses = 16,  eval_batch_size= 24, test_batch_size = 24 , batch_size= 24, input_shape=4096, gpu="1", glimpse_size= 256, patch_size = 4096, sampling_size_train = 1, sampling_size_val = 1, sampling_size_test = 1):
        

                
        self.temp_path = temp_path
        self.label_path = label_path
        self.selected_features = selected_features
        self.val_size = val_size
        self.test_size = test_size
        self.input_shape = input_shape
        
        self.source_path = source_path
        self.array_path = array_path
        self.logdir = logdir
        
        self.patch_size = patch_size

        self.gpu = gpu

        self.num_glimpses = num_glimpses
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        
        # glimpse network
        self.glimpse_size = glimpse_size
        self.bandwidth = glimpse_size ** 2
        self.sensor_size = glimpse_size ** 2 * 3

        self.hg_size = 1024
        self.hl_size = 1024
        self.loc_dim = 2
        self.g_size = 2048
        self.regularizer = 0.01

        # training
        self.steps = 500
        self.lr_start = 0.01
        self.lr_min = 0.0001
        
        self.loc_std = 0.25
        self.max_grad_norm = 5.
        self.n_verbose = 1

        # lstm
        self.cell_output_size = 3072
        self.cell_size = 3072
        #self.cell_output_size = 2048
        #self.cell_size = 2048
        self.cell_out_size = self.cell_size

        # task
        self.num_classes = 2
        
        
        self.sampling_size_train  = sampling_size_train 
        self.sampling_size_val = sampling_size_val
        self.sampling_size_test = sampling_size_test
        
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
            os.makedirs(self.logdir)
        else:
            os.makedirs(self.logdir)

        return
