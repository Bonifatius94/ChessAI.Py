
import sys
import chessai


def main():

    # parse script args
    startup_config = sys.argv[1] if len(sys.argv) >= 2 else 'all'

    # launch the training according to the specified startup config
    if startup_config == 'pretrain_ratings': launch_pretrain_ratings()
    elif startup_config == 'train_deepq': launch_train_deepq()
    # register future trainings here ...
    elif startup_config == 'all': launch_train_all()
    else: raise ValueError('Invalid args! Unknown training startup configuration {}!'.format(startup_config))


def launch_pretrain_ratings():

    # define training parameters
    params = {

        # define dataset batch size and training epochs
        'batch_size': 32,
        'epochs': 30,
        'train_data_split': 1.0,

        'rating_classes': 3,
        'dropout_rate': 0.0,

        # define the learning rate (exp. decay)
        'learn_rate': 0.005,
        'total_train_batches': 2496,
        'lr_decay_epochs': 5,
        'lr_decay_rate': 0.5,
        'lr_decay_staircase': False,

        # define regularization loss penalties
        'l1_penalty': 1e-4,
        'l2_penalty': 1e-5,

        # make the feature extractor variables trainable
        'is_fx_trainable': True,
    }

    # create a new training session and launch the training
    session = chessai.train.RatingTrainingSession(params)
    session.run_training()


def launch_train_deepq():

    # define training parameters
    params = {

        # define dataset batch size and training epochs
        'batch_size': 32,
        'epochs': 1000,
        'batches_per_epoch': 200,
        'fit_epochs': 1,

        # Q-learning params
        'expl_rate': 0.1,
        'gamma': 0.99,

        # define the learning rate (exp. decay)
        'learn_rate': 0.01,
        'momentum': 0.8,

        # define model regularization parameters
        'dropout_rate': 0.0,
        'l1_penalty': 1e-4,
        'l2_penalty': 1e-5,

        # make the feature extractor variables trainable
        'is_fx_trainable': True,

        # define the logging settings
        'stockfish_level': 7,
        'sample_interval': 100,
    }

    # create a new training session and launch the training
    session = chessai.train.DeepQTrainingSession(params)
    session.run_training()


def launch_train_all():

    # TODO: implement a useful multi-stage train config
    pass


if __name__ == '__main__':
    main()