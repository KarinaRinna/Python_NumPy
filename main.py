import numpy
import matplotlib
import numpy as np

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_input_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

bias_input_to_hidden = np.zeros((20, 1))
bias_input_to_output = np.zeros((10, 1))

