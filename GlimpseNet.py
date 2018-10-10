import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim


class ConvGlimpseNetwork(object):
    """ Takes image and previous glimpse location and outputs feature vector."""

    def __init__(self, config, images_ph):
        self.config = config
        self.images_ph = images_ph
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.regularizer)
        self.initializer = tf.truncated_normal_initializer()    


    def get_glimpse(self, loc):
        loc = tf.stop_gradient(loc)
        glimpse_size = self.config.glimpse_size
        input_shape = self.config.input_shape
        self.glimpse = tf.image.extract_glimpse(self.images_ph, [glimpse_size, glimpse_size], loc, centered=True,
                                           normalized=True, name='extract_glimpse')
        return self.glimpse
    
     

    def __call__(self, loc):
        glimpse_input = self.get_glimpse(loc)

        with tf.variable_scope('glimpse_sensor', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('convolutions', reuse=tf.AUTO_REUSE): 
               # h, _ = resnet_v1.resnet_v1_101(glimpse_input, self.config.g_size, is_training=True)
               # h = tf.nn.avg_pool(h, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
                
                h = tf.layers.conv2d(inputs=glimpse_input, filters= 16, kernel_size= 3, padding='SAME',
                                     kernel_initializer=self.initializer, name="conv1", activation = tf.nn.relu)
               
                h = tf.layers.conv2d(h, filters= 16, kernel_size=3, padding='SAME', kernel_initializer=self.initializer,
                                     name="conv2")
                
                h = tf.layers.batch_normalization(h, training = True)
                
                h = tf.nn.relu(h)

                h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
                

                h = tf.layers.conv2d(h, filters= 32, kernel_size=3, padding='SAME', kernel_initializer=self.initializer,
                                     name="conv3", activation=tf.nn.relu)

                h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
                
                 
            h = tf.contrib.layers.flatten(h)

        with tf.variable_scope('glimpse_sensor', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('combine', reuse=tf.AUTO_REUSE):
                    h = tf.layers.dense(inputs=h, units=self.config.g_size, kernel_initializer=self.initializer)
                    h = tf.layers.batch_normalization(h, training = True)
                    h = tf.nn.relu(h)

        with tf.variable_scope('location_encoder', reuse=tf.AUTO_REUSE):
            l = tf.layers.dense(loc, units=self.config.hl_size, kernel_initializer=self.initializer)

        with tf.variable_scope('combined_where_and_what', reuse=tf.AUTO_REUSE):
            l = tf.layers.dense(l, units=self.config.g_size, kernel_initializer=self.initializer)


        # combine
        g = tf.nn.relu(h * l)

        return g


class LocNet(object):
    """ Location network.
        Takes RNN output and produces the 2D mean for next location.
    """

    def __init__(self, config):
        self.config = config
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.regularizer)
        self.initializer = tf.contrib.layers.xavier_initializer()

    def __call__(self, input):

        with tf.variable_scope('loc'):
            mean = tf.layers.dense(input, units=self.config.loc_dim, kernel_initializer=self.initializer, activation = tf.nn.tanh)
            
        loc = mean + tf.random_normal((tf.shape(input)[0], 2), stddev= 0.001)
       # loc = mean 
       # loc = tf.clip_by_value(loc, -1., 1.)

        loc = tf.stop_gradient(loc)

        return loc, mean