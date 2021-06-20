import os


ROOT_DIR = os.environ['RLROOT']

DEFAULT_PARAMS = {
    'episodes': 1000000,
    'training_steps': 1000000,
    'batch_size': 32,
    'mem_size': 300000,
    'dry_size': 10000,
    'input_shape': (1, 90),
    'history': 1,
}
