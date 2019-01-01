from multiprocessing import Process, Queue
from base.base_generator import BaseGenerator
import numpy as np


def generate_func(*args, **kwargs):
    # generate a batch, multi-thread can jointly call this func
    config = kwargs['config']
    queue = kwargs['queue']
    while True:  # important
        # mimic the process of generating data
        input = np.random.rand(config.batch_size, config.state_size)
        y = np.zeros((config.batch_size, config.label_num))
        for i in range(config.batch_size):  # set label randomly
            y[i][np.random.randint(config.label_num)] = 1

        queue.put((input, y))



class WarpGenerator(BaseGenerator):
    """
    A generator that, in parallel

    """

    def __init__(self, config, *args):
        super(WarpGenerator, self).__init__(config)

        self.n_workers = self.config.n_workers

        self.result_queue = Queue(maxsize=self.n_workers*2)
        self.processors = []
        for i in range(self.n_workers):
            self.processors.append(
                Process(target=generate_func, args=args, kwargs={'config':config, 'queue':self.result_queue}))
            self.processors[-1].start()


    def next_batch(self):
        yield self.result_queue.get()



    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

        self.result_queue.close()



