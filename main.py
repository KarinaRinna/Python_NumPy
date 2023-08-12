import matplotlib.pyplot as plt
import numpy
import matplotlib
import numpy as np
import utils

images, labels = utils.load_dataset()


weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 3
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Epoch №{epoch}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # прямое распространение hidden слой
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

        # прямое распространение output слой
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        # ошибка расчета
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # обратное распространение (выходной слой)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # обратное распространение (скрытый  слой)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    print(f"Loss:{round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy:{round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# проверка
test_image = plt.imread("custom.jpg", format="jpeg")

gray = lambda rgb : np.dot(rgb[... , :3] , [0.299, 0.587, 0.144])
test_image = 1 - (gray(test_image).astype("float32") / 255)

test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

image = np.reshape(test_image, (-1, 1))

hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) #сигмойд

output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"Нейросеть предпологает что это цифра: {output.argmax()}")
plt.show()

#https://www.youtube.com/watch?v=tihq_bLfk08
# 16:16 nymPy.save, nymPy.load