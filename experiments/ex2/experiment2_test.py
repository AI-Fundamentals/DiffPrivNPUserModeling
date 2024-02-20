# Load packages

import neuralprocesses.tensorflow as nps
import argparse
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import json
import matplotlib.pyplot as plt


from dppum.data import hdf_to_tf_dataset
from dppum.loss import np_elbo_tf_cat
from dppum.util import print_dictionary
from dppum.train import train_model_dp_tf

print("Finished importing packages.")
