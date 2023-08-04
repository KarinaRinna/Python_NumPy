import numpy
import matplotlib
import numpy as np
import utils

images, labels = utils.load_dataset()


weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_input_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_input_to_output = np.zeros((10, 1))

epochs = 3
e_loss = 0
e_correct = 0

for epoch in range(epochs):
    print(f"Epoch №{epoch}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

    # прямое распространение hidden слой
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

    # прямое распространение output слой
    output_raw = bias_input_to_output + weights_input_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    # ошибка расчета