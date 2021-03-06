import tensorflow as tf
import numpy as np
from utils.scope_decorator import define_scope

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.build_model() # common build procedure: set seed, step, epoch, build graph and initialize parameters
        self.loaded = False # check whether loading model successfully


    def build_model(self):
        """All the variable you want to save should be under the `self.graph`."""
        # set global seed
        self.set_random_seed()

        # init the global step
        self.global_step

        # init the epoch counter
        self.cur_epoch

        # build graph, override by subclass
        self.build_graph()

        # init saver, must be put after the build_graph
        self.saver


    # set random seed both for tensorflow and numpy
    def set_random_seed(self):
        tf.set_random_seed(seed=self.config.seed)
        np.random.seed(seed=self.config.seed)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        '''make sure you can't call the global variables init after calling this method'''
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            self.loaded = True


    # just initialize a tensorflow variable to use it as epoch counter
    @define_scope(scope='epoch')
    def cur_epoch(self):
        return tf.Variable(0, trainable=False, name='cur_epoch')

    @define_scope(scope='epoch')
    def incr_cur_epoch_op(self):
        return tf.assign(self.cur_epoch, self.cur_epoch + 1, name='incr_cur_epoch_op')


    # just initialize a tensorflow variable to use it as global step counter
    @define_scope
    def global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        return tf.Variable(0, trainable=False, name='global_step')


    @define_scope
    def saver(self):
        '''
        The saver must be define under the `self.graph` and init at the last of the graph,
            otherwise it can't find all the variables under the graph.
        '''
        return tf.train.Saver(max_to_keep=self.config.max_to_keep)



    def build_graph(self):
        '''
        override by subclass, should define
            placeholder node, operation node and variable node to construct the graph jointly in decorator style

            Input & output: Placeholder, one-hot
            Model Parameters: Variable, embedding layer, dense layer, etc...
            Inference Op (Forward): Operation, Network Structure
            Optimize Op: (Backward): Optimizer, single-batch-iteration

        '''
        raise NotImplementedError
