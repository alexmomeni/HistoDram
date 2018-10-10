""" Implements Logger class for tensorflow experiments. """

import tensorflow as tf
import os

class Logger(object):

    
    def __init__(self, log_dir='/home/aamomeni/research/momena/tests/experiments', sess=None, summary_ops={}, var_list=[],
               global_step=None, eval_ops={}, n_verbose=10):
        
        self.session                            = sess

        # folders
        self.log_dir                            = log_dir

        self.checkpoint_path, self.summary_path = self.create_directories(log_dir)

        # file writers
        self.writers = {
            'train':  tf.summary.FileWriter(os.path.join(self.summary_path,'train'),self.session.graph, flush_secs= 120),
            'test':   tf.summary.FileWriter(os.path.join(self.summary_path,'test')),
            'val':    tf.summary.FileWriter(os.path.join(self.summary_path,'val'))
        }

        # saver
        self.global_step = global_step
        if var_list == []:
            self.saver       = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        else:
            self.saver       = tf.train.Saver(var_list, keep_checkpoint_every_n_hours=1)

        # summaries
        self.summary_ops = summary_ops
        self.eval_ops    = eval_ops
        self.merged_op   = tf.summary.merge_all()

        # step counter
        self.step        = 0
        self.n_verbose   = n_verbose
   
    
    
    def log(self, writer_id, feed_dict):
        """ Logs performance using either 'train', 'test' or 'val' writer"""
        summaries = self.session.run(self.merged_op, feed_dict=feed_dict)
        self.writers[writer_id].add_summary(summaries,self.step)
        
       # print ('\n------ Step %s ------' % (self.step))
 
       # for key in self.eval_ops.keys():
        #    val = self.session.run(self.eval_ops[key],feed_dict)
         #   print ('%s \t %s' %(key,val))
            
  
        
            
    def create_directories(self, log_dir):
        
        checkpoint_path = os.path.join(log_dir, 'checkpoints')
        summary_path    = os.path.join(log_dir, 'summaries')
    
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            os.mkdir(checkpoint_path)
            os.mkdir(summary_path)

        print ('\n\nLogging to <<%s>>.\n\n' % log_dir )
        
        return checkpoint_path, summary_path
       
    def save(self):
        self.saver.save(self.session, os.path.join(self.checkpoint_path,'checkpoint'),
                        global_step=self.global_step)
        
    def restore(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print (ckpt)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
