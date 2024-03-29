
import sys
import chessai


def main():

    # parse script args
    startup_config = sys.argv[1] if len(sys.argv) >= 2 else 'all'

    # launch the training according to the specified startup config
    if startup_config == 'pretrain_fx': launch_pretrain_fx()
    if startup_config == 'pretrain_ratings': launch_pretrain_ratings()

    # TODO: add launching single other trainings here ...

    elif startup_config == 'all':
        launch_pretrain()
        launch_pretrain_ratings()
        # TODO: add launching other trainings here ...
    else:
        raise ValueError('Invalid args! Unknown training startup configuration {}!'.format(startup_config))


def launch_pretrain():

    # launch entire pre-training
    launch_pretrain_fx()


def launch_pretrain_fx():

    params = {
        'batch_size': 32,
        'learn_rate': 0.2,
        'epochs': 30,
        'lr_decay_epochs': 3,
        'lr_decay_rate': 0.5,

        'log_interval': 100,
        'total_train_batches': 2400, # TODO: find out the exact value
    }

    # create a new training session and launch the training
    session = chessai.pretrain.DrawGenTrainingSession(params)
    session.run_training()

    # TODO: launch all other pre-train sessions here, too ...


def launch_pretrain_ratings():

    # frozen fx settings
    params = {
        'batch_size': 32,
        'learn_rate': 0.01,
        'epochs': 30,
        'lr_decay_epochs': 3,
        'lr_decay_rate': 0.1,
        'is_fx_trainable': False,

        'log_interval': 100,
        'total_train_batches': 2400, # TODO: find out the exact value
    }

    # create a new training session and launch the training
    session = chessai.pretrain.RatingTrainingSession(params)
    session.run_training()


# def get_instance_by_name(fq_classname: str):
#     parts = kls.split('.')
#     module = ".".join(parts[:-1])
#     m = __import__( module )
#     for comp in parts[1:]:
#         m = getattr(m, comp)            
#     return m


if __name__ == '__main__':
    main()