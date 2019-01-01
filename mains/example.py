import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.argser import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session

    # create your data generator
    data = DataGenerator(config)

    graph = tf.Graph()
    # create an instance of the model you want
    with graph.as_default():
        sess = tf.Session(graph=graph)
        model = ExampleModel(config)
        # create tensorboard logger
        logger = Logger(sess, config)
        # create trainer and pass all the previous components to it
        trainer = ExampleTrainer(sess, model, data, config, logger)
        #load model if exists
        model.load(sess)
        # here you train your model
        trainer.train()


if __name__ == '__main__':
    main()
