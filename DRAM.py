import tensorflow as tf
import pandas as pd
from GlimpseNet import ConvGlimpseNetwork
from GlimpseNet import LocNet
from Datasets import Dataset
from utils.Logger import Logger
from utils.fig import *
from utils.generator import Generator
import shutil
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score,average_precision_score, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import label_binarize
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle


class DRAM(object):

    def __init__(self, config):
        self.config = config
        self.data_init()
        self.model_init()

    def data_init(self):
        print("\nData init")
        self.dataset = Dataset(self.config)
        self.generator = Generator(self.config, self.dataset)
   
    def model_init(self):

        self.rnn_cell = tf.contrib.rnn
        self.config = config
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.regularizer)
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.images_ph = tf.placeholder(tf.float32, [None, self.config.input_shape, self.config.input_shape, 3])
        self.labels_ph = tf.placeholder(tf.int64, [None])
        self.N = tf.shape(self.images_ph)[0]

        # ------- GlimpseNet / LocNet -------

        with tf.variable_scope('glimpse_net'):
            self.gl = ConvGlimpseNetwork(self.config, self.images_ph)

        with tf.variable_scope('loc_net'):
            self.loc_net = LocNet(self.config)

        self.init_loc = tf.zeros(shape=[self.N, 2], dtype=tf.float32)
        with tf.variable_scope("rnn_decoder/loop_function", reuse=tf.AUTO_REUSE):
            self.init_glimpse = self.gl(self.init_loc)

        self.inputs = [self.init_glimpse]
        self.inputs.extend([0] * (self.config.num_glimpses - 1))

        # ------- Recurrent network -------

        def get_next_input(output, i):

            loc, loc_mean = self.loc_net(output)
            gl_next = self.gl(loc)

            self.loc_mean_arr.append(loc_mean)
            self.sampled_loc_arr.append(loc)
            self.glimpses.append(self.gl.glimpse)

            return gl_next

        def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None):

            with tf.variable_scope("rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None

                for i, inp in enumerate(decoder_inputs):
                    if loop_function is not None and prev is not None:
                        with tf.variable_scope("loop_function", reuse=tf.AUTO_REUSE):
                            inp = loop_function(prev, i)

                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    output, state = cell(inp, state)
                    outputs.append(output)

                    if loop_function is not None:
                        prev = output

            return outputs, state

        self.loc_mean_arr = [self.init_loc]
        self.sampled_loc_arr = [self.init_loc]
        self.glimpses = [self.gl.glimpse]

        self.lstm_cell = self.rnn_cell.LSTMCell(self.config.cell_size, state_is_tuple=True, activation=tf.nn.tanh, forget_bias=1.)
        self.init_state = self.lstm_cell.zero_state(self.N, tf.float32)
        self.outputs, self.rnn_state = rnn_decoder(self.inputs, self.init_state, self.lstm_cell,
                                                   loop_function=get_next_input)

        # ------- Classification -------
        
        baselines = []
        for t, output in enumerate(self.outputs):
            with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
                baseline_t = tf.layers.dense(inputs=output, units=2, kernel_initializer=self.initializer)
            baseline_t = tf.squeeze(baseline_t)
            baselines.append(baseline_t)

        baselines = tf.stack(baselines) 
        self.baselines = tf.transpose(baselines) 


            
        with tf.variable_scope('classification', reuse= tf.AUTO_REUSE):
            self.class_prob_arr = []
            for t, op in enumerate(self.outputs):
                self.glimpse_logit = tf.layers.dense(inputs=op, units=self.config.num_classes, kernel_initializer=self.initializer,
                                                     name='FCCN', reuse=tf.AUTO_REUSE)
                self.glimpse_logit = tf.stop_gradient(self.glimpse_logit)
                self.glimpse_logit = tf.nn.softmax(self.glimpse_logit)
                self.class_prob_arr.append(self.glimpse_logit)
            self.class_prob_arr = tf.stack(self.class_prob_arr, axis=1)
            
        self.output = self.outputs[-1]  
        with tf.variable_scope('classification', reuse=tf.AUTO_REUSE):
            self.logits = tf.layers.dense(inputs=self.output, units=self.config.num_classes, kernel_initializer=self.initializer, name='FCCN',
                                      reuse=tf.AUTO_REUSE)

          
            self.softmax = tf.nn.softmax(self.logits)
 
        self.sampled_locations = tf.concat(self.sampled_loc_arr, axis=0)
        self.mean_locations = tf.concat(self.loc_mean_arr, axis=0)
        self.sampled_locations = tf.reshape(self.sampled_locations, (self.config.num_glimpses, self.N, 2))
        self.sampled_locations = tf.transpose(self.sampled_locations, [1, 0, 2])
        self.mean_locations = tf.reshape(self.mean_locations, (self.config.num_glimpses, self.N, 2))
        self.mean_locations = tf.transpose(self.mean_locations, [1, 0, 2])
        prefix = tf.expand_dims(self.init_loc, 1)
        self.sampled_locations = tf.concat([prefix, self.sampled_locations], axis=1)
        self.mean_locations = tf.concat([prefix, self.mean_locations], axis=1)
        self.glimpses = tf.stack(self.glimpses, axis=1)
            

        # Losses/reward

        def loglikelihood(mean_arr, sampled_arr, sigma):
            mu = tf.stack(mean_arr) 
            sampled = tf.stack(sampled_arr) 
            gaussian = tf.contrib.distributions.Normal(mu, sigma)
            logll = gaussian.log_prob(sampled)
            logll = tf.reduce_sum(logll, 2)
            logll = tf.transpose(logll)  
            return logll
        
        
        self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph)
        self.xent = tf.reduce_mean(self.xent)

        self.pred_labels = tf.argmax(self.logits, 1)
        self.reward = tf.cast(tf.equal(self.pred_labels, self.labels_ph), tf.float32)   
        self.rewards = tf.expand_dims(self.reward, 1)
        self.rewards = tf.tile(self.rewards, [1, self.config.num_glimpses])        
        self.logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, self.config.loc_std)
        self.advs = self.rewards - tf.stop_gradient(self.baselines)
        self.logllratio = tf.reduce_mean(self.logll * self.advs)
                
        self.reward = tf.reduce_mean(self.reward)

        
        self.baselines_mse = tf.reduce_mean(tf.square((self.rewards - self.baselines)))
        self.var_list = tf.trainable_variables()

        self.loss = -self.logllratio + self.xent + self.baselines_mse
        self.grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(self.grads, self.config.max_grad_norm)

        self.setup_optimization()

        # session
        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.visible_device_list = self.config.gpu
        self.session_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.session_config)
        self.session.run(tf.global_variables_initializer())

    def setup_optimization(self):

        # learning rate
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        self.training_steps_per_epoch = int(len(self.generator.training_ids) // self.config.batch_size)
        print('Training Step Per Epoch:',self.training_steps_per_epoch)
       
        self.starter_learning_rate = self.config.lr_start
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, self.training_steps_per_epoch, 0.70, staircase=False)
        self.learning_rate = tf.maximum(self.learning_rate, self.config.lr_min)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.90, use_nesterov=True)
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.var_list), global_step=self.global_step)

    def setup_logger(self):
        """Creates log directory and initializes logger."""

        self.summary_ops = {'reward': tf.summary.scalar('reward', self.reward),
            'hybrid_loss': tf.summary.scalar('hybrid_loss', self.loss),
            'cross_entropy': tf.summary.scalar('cross_entropy', self.xent),
            'baseline_mse': tf.summary.scalar('baseline_mse', self.baselines_mse),
            'logllratio': tf.summary.scalar('logllratio', self.logllratio),
            'lr': tf.summary.scalar('lr', self.learning_rate)}
           # 'glimpses': tf.summary.image('glimpses',tf.reshape(self.glimpses,[-1,self.config.glimpse_size,
            #                                                                  self.config.glimpse_size,
             #                                                                 3]),max_outputs=8)}
        
        self.eval_ops = {'labels': self.labels_ph, 'pred_labels': self.pred_labels, 'reward': self.reward, 
                         'hybrid_loss': self.loss, 'cross_entropy': self.xent, 'baseline_mse': self.baselines_mse, 
                         'logllratio': self.logllratio, 'lr': self.learning_rate}
        
        self.logger = Logger(self.config.logdir, sess=self.session, summary_ops=self.summary_ops,
                             global_step=self.global_step, eval_ops=self.eval_ops, n_verbose=self.config.n_verbose,
                             var_list=self.var_list)

    def train(self):

        print('\n\n\n------------ Starting training ------------  \nT -- %s x %s \n' \
              'Model:  %s glimpses, glimpse size %s x %s \n\n\n' % (
                  self.config.input_shape, self.config.input_shape, self.config.num_glimpses, self.config.glimpse_size,
                  self.config.glimpse_size))

        self.setup_logger()

        for i in range(self.config.steps  + 1) :
            
            loc_dir_name = self.config.logdir +'/image/locations'
            traj_dir_name = self.config.logdir +'/image/trajectories'
            ROCs_dir_name = self.config.logdir + '/metrics/ROCs_AUCs/'
            PRs_dir_name = self.config.logdir + '/metrics/PRs/'
            
            if i == 0:
                if os.path.exists(loc_dir_name):
                    shutil.rmtree(loc_dir_name)
                    os.makedirs(loc_dir_name)
                else:
                    os.makedirs(loc_dir_name)

                if os.path.exists(traj_dir_name):
                    shutil.rmtree(traj_dir_name)
                    os.makedirs(traj_dir_name)
                else:
                    os.makedirs(traj_dir_name)
                
                if os.path.exists(ROCs_dir_name):
                    shutil.rmtree(ROCs_dir_name)
                    os.makedirs(ROCs_dir_name)
                else:
                    os.makedirs(ROCs_dir_name)
                
                if os.path.exists(PRs_dir_name):
                    shutil.rmtree(PRs_dir_name)
                    os.makedirs(PRs_dir_name)
                else:
                    os.makedirs(PRs_dir_name)
            
            self.logger.step = i
            
            images, labels = self.generator.generate()
            images = images.reshape((-1, self.config.input_shape, self.config.input_shape, 3))
            labels = labels[0]
            feed_dict = {self.images_ph: images, self.labels_ph: labels}
            

            fetches = [self.output, self.rewards, self.reward, self.labels_ph, self.pred_labels, self.logits, self.train_op, self.loss, self.xent, self.baselines_mse, self.logllratio, self.learning_rate, self.loc_mean_arr]
            output, rewards, reward, real_labels, pred_labels, logits, _ , hybrid_loss, cross_entropy, baselines_mse, logllratio, lr, locations = self.session.run(fetches, feed_dict)
                
            if i % 1 ==0:

                print ('\n------ Step %s ------' % (i))
                print('reward', reward)
                print('labels', real_labels)
                print('pred_labels', pred_labels)
                print('hybrid_loss', hybrid_loss)
                print('cross_entropy', cross_entropy)
                print('baseline_mse',baselines_mse)
                print('logllratio', logllratio)
                print('lr',lr)
                print('locations', locations[-1])
                print('logits',logits)
                self.logger.log('train', feed_dict=feed_dict)
                
            #if  i > 0 and i % 100 == 0:
                
             #   self.eval(i)
              #  self.logger.log('val', feed_dict=feed_dict)
                
            
            if  i == self.config.steps:
                
                self.test(i)
      

            #if i == self.config.steps:
           # if i > 0 and i % 100 == 0:

               
            #    glimpse_images = self.session.run(self.glimpses, feed_dict)
             #   mean_locations = self.session.run(self.mean_locations, feed_dict)
              #  probs = self.session.run(self.class_prob_arr, feed_dict)

               # plot_glimpses(config=self.config, glimpse_images=glimpse_images, pred_labels=pred_labels, probs=probs,
                           #   sampled_loc=mean_locations, X=images, labels=real_labels, file_name=loc_dir_name, step=i)

                #plot_trajectories(config=self.config, locations=mean_locations, X=images, labels=real_labels,
                               #   pred_labels=pred_labels, file_name=traj_dir_name, step=i)
                
                #self.logger.save()

    
    
    def eval(self, step):
        return self.evaluate(self.session, self.images_ph, self.labels_ph, self.softmax, step)

    def evaluate(self, sess, images_ph, labels_ph, softmax, step):
        print('Evaluating (%s x %s) using %s glimpses' % (
            self.config.input_shape, self.config.input_shape, self.config.num_glimpses))
        self.X_val, self.y_val = self.dataset.convert_to_arrays(self.dataset._partition[0]['val'], size = self.config.sampling_size_val)
        print('Validation set has %s patients' % len(self.y_val))

        X_val, y_val = self.X_val, self.y_val
        
        _num_examples = X_val.shape[0]
        steps_per_epoch = _num_examples // self.config.eval_batch_size

        y_scores = []
        y_trues = []

        for i in tqdm(iter(range(steps_per_epoch))):

            images, labels_val = self.dataset.next_batch(X_val, y_val[0], self.config.eval_batch_size, i)
            #images = images.reshape((-1, self.config.input_shape, self.config.input_shape, 3))
            
            softmax_val = sess.run(softmax, feed_dict={images_ph: images, labels_ph: labels_val})
            y_trues.extend(labels_val)
            y_scores.extend(softmax_val)        
        
        y_preds = np.argmax(y_scores, 1)
        y_scores = np.array(y_scores)

        self.metrics_ROCs(y_trues, y_preds, y_scores, step)
        self.metrics(y_trues, y_preds, step)
        return 
    
    
    
    def count_params(self):
        return self.count_parameters(self.session)

    def count_parameters(self, sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        n_params = 0

        for k, v in zip(variables_names, values):
            print('-'.center(140, '-'))
            print('%s \t Shape: %s \t %s parameters' % (k, v.shape, v.size))
            n_params += v.size

        print('-'.center(140, '-'))
        print('Total # parameters:\t\t %s \n\n' % (n_params))
        return n_params
    
    
    def metrics_ROCs(self, y_trues, y_preds, y_scores, step, stage = None):
        
        y_trues_binary= label_binarize(y_trues, classes=list(self.dataset.le_name_mapping.values()))
        y_preds_binary= label_binarize(y_preds, classes=list(self.dataset.le_name_mapping.values()))
        n_classes = y_preds_binary.shape[1]
        if stage == 'test':
            fpr, tpr, _ = roc_curve(y_trues, y_scores)
        else:
            fpr, tpr, _ = roc_curve(y_trues, y_scores[:,1])
                        
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        
        plt.plot(fpr, tpr,label='ROC curve (AUC = {0:0.2f})'''.format(roc_auc),color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiving Operating Characteristic Curves')
        plt.legend(loc="lower right")
        plt.savefig(self.config.logdir + '/metrics/ROCs_AUCs/%i' %step)
        return 

    def metrics(self, y_trues, y_preds, step):
#        y_trues_binary= label_binarize(y_trues, classes=list(self.dataset.le_name_mapping.values()))
 #       y_preds_binary= label_binarize(y_preds, classes=list(self.dataset.le_name_mapping.values()))
        
        accuracy = accuracy_score(y_trues, y_preds)
        f1score = f1_score(y_trues, y_preds)
        recall = recall_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds)
        names = ['accuracy','f1_score','recall','precision']
        pd.DataFrame(data=np.array([accuracy,f1score,recall,precision]), index=names).to_csv(self.config.logdir + '/metrics/metrics_%i.csv' % step)
        return
    
    def load(self, checkpoint_dir):
        folder = os.path.join(checkpoint_dir,'checkpoints')
        print ('\nLoading model from <<{}>>.\n'.format(folder))

        self.saver       = tf.train.Saver(self.var_list)
        ckpt = tf.train.get_checkpoint_state(folder)

        if ckpt and ckpt.model_checkpoint_path:
            print (ckpt)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            
    def patch_to_image(self, y_patches, proba=True):
        
        if proba == True:
            y_image = np.array([np.mean(y_patches[i*self.config.sampling_size_test:(i+1)*self.config.sampling_size_test], axis = 0) for i in range(int(len(y_patches)/self.config.sampling_size_test))])
                        
        else:
            y_image = np.array([np.mean(y_patches[i*self.config.sampling_size_test:(i+1)*self.config.sampling_size_test])>0.5 for i in range(int(len(y_patches)/self.config.sampling_size_test))]).reshape((-1,1)).astype(int)
            y_image = np.asarray(y_image.flatten())
        return y_image

                
    def test(self, step):
        return self.testing(self.session, self.images_ph, self.labels_ph, self.softmax, step)

    
    def testing(self, sess, images_ph, labels_ph, softmax, step):
        print('Testing (%s x %s) using %s glimpses' % (
            self.config.input_shape, self.config.input_shape, self.config.num_glimpses))
        print(self.dataset._partition[0]['test'])
        self.X_test, self.y_test = self.dataset.convert_to_arrays(self.dataset._partition[0]['test'], size = self.config.sampling_size_test)
        X_test, y_test = self.X_test, self.y_test
        print('y_test', y_test)
        _num_examples = X_test.shape[0]
        steps_per_epoch = _num_examples // self.config.test_batch_size

        y_scores = []
        y_trues = []

        for i in tqdm(iter(range(steps_per_epoch))):

            images, labels_test = self.dataset.next_batch(X_test, y_test[0], self.config.test_batch_size, i)
            
            print(labels_test)
            #images = images.reshape((-1, self.config.input_shape, self.config.input_shape, 3))
            
            softmax_test = sess.run(softmax, feed_dict={images_ph: images, labels_ph: labels_test})
            y_trues.extend(labels_test)
            y_scores.extend(softmax_test) 
            
     
        y_trues = self.patch_to_image(y_trues, proba=False)
        y_scores = self.patch_to_image(y_scores, proba=True)
        
        y_preds = np.argmax(y_scores, 1)
        
        print('Test Set', self.dataset._partition[0]['test'])
        print(y_trues)
        print(y_preds)
        
        self.metrics_ROCs(y_trues, y_preds, y_scores, step) 
        self.metrics(y_trues, y_preds, step)
        return
    

    