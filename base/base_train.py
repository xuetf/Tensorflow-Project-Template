import tensorflow as tf
from utils.logger import Logger

class BaseTrain:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.logger = Logger(sess, config)

        if self.config.load and model.loaded:
            pass
        else:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)


    def train(self):
        for cur_epoch in range(self.model.cur_epoch.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.incr_cur_epoch_op)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
