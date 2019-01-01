import tensorflow as tf

from data_loader.random_generator import RandomGenerator
from data_loader.warp_generator import WarpGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
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


    # create your data generator
    #generator = WarpGenerator(config)
    generator = RandomGenerator(config)

    graph = tf.Graph()
    # create an instance of the model you want
    with graph.as_default():
        # create tensorflow session
        sess = tf.Session(graph=graph)

        # create model
        model = ExampleModel(config)
        # load model if exists
        if config.load: model.load(sess)

        # create trainer and pass all the previous components to it
        trainer = ExampleTrainer(sess, model, generator, config)
        # here you train your model
        trainer.train()

        sess.close()

    generator.close()


if __name__ == '__main__':
    main()
