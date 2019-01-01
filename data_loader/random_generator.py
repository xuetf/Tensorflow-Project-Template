import numpy as np
from base.base_generator import BaseGenerator

class RandomGenerator(BaseGenerator):

    def __init__(self, config):
        super(RandomGenerator, self).__init__(config)

        self.total_sample_num = 500
        # load data here
        self.input = np.random.rand(self.total_sample_num, self.config.state_size)
        self.y = np.zeros((self.total_sample_num, self.config.label_num))
        for i in range(self.total_sample_num): # set label randomly
            self.y[i][np.random.randint(self.config.label_num)] = 1



    def next_batch(self):
        idx = np.random.choice(self.total_sample_num, self.batch_size)
        yield self.input[idx], self.y[idx]


    def close(self):
        pass