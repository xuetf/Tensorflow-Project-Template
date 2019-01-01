from base.base_model import BaseModel
import tensorflow as tf


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)



    def build_graph(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass


    def set_placeholder(self):

        pass