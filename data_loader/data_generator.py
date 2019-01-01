import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.random.rand(500, 784)
        self.y = np.zeros((500, 10))
        for i in range(500): # set label randomly
            self.y[i][np.random.randint(10)] = 1



    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
