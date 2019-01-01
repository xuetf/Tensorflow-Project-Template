from base.base_model import BaseModel
import tensorflow as tf
from utils.scope_decorator import *


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)


    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size, name='input')
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name='label')

        self.loss
        self.optimize
        self.accuracy




    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="mlp_layer_1")


    @define_scope
    def mlp_layer_2(self):
        return tf.layers.dense(self.mlp_layer_1, 10, name="mlp_layer_2")

    @define_scope
    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.mlp_layer_2))


    @define_scope
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self.config.learning_rate)\
                .minimize(self.loss,global_step=self.global_step)


    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.mlp_layer_2, 1), tf.argmax(self.y, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

