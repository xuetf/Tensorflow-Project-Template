
class BaseGenerator:

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size



    def next_batch(self):
        raise NotImplementedError



    def close(self):
        raise NotImplementedError