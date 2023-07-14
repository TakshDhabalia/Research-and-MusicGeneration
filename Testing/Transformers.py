#implementing some transformers 
import time

import numpy as np
from matplotlib import pyplot as plt 

import tensorflow_datasets as tfds
import tensorflow as tf
import torch  
import tensorflow_text

#making the self attention layer 
class Relative_multi_head_attention (tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, name='mha'):
        super(Relative_multi_head_attention, self).__init__(name=name)

        assert d_model % n_heads == 0#basically saying that this is alsways the case 