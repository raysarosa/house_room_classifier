import random
import numpy as np
import tensorflow as tf

SEED = 42  # Global seed for reproducibility

def set_seed(seed=SEED):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)