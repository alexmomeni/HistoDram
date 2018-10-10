from DRAM  import DRAM
from config import Config
import tensorflow as tf

config  = Config()

#tf.reset_default_graph()
model = DRAM(config)
#model.load('/home/aamomeni/research/histo_dram/Subtype/')
model.count_params()
model.train()