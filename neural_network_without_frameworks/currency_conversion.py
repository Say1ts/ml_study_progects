import random
import numpy as np


def relu(t):
    return np.maximum(t, 0)


def relu_deriv(t):
    return (t >= 0).astype(float)


class NeuralNetwork:
    def __init__(self, alpha):
        # Инициализация начальных весов
        self.weight = random.random()
        self.bias_weight = random.random()
        self.ALPHA = alpha

        self.loss_array = []
        self.loss_array_epoch = []

    def train(self, training_inputs, training_outputs, num_epochs):
        print("Средние ошибки по эпохам")
        for epoch in range(num_epochs):
            # рассчет массива выходных значений НС для каждого из входных значений
            output = [self.predict(training_inputs[j]) for j in range(len(training_outputs))]

            # Расчет средней ошибки
            if epoch % 10 == 0 or epoch < 10:
                error = [(training_outputs[j] - output[j])**2 for j in range(len(training_outputs))]
                error_mean = 0
                for i in range(len(error)):
                    error_mean += error[i]
                error_mean /= len(error)
                self.loss_array.append(error_mean)
                self.loss_array_epoch.append(epoch)
                print("Эпоха №", epoch, ":", error_mean)

            # коррекция весов
            for i in range(len(output)):
                # вычисление ошибки
                error = [(training_outputs[j] - output[j]) ** 2 for j in range(len(training_outputs))]
                self.weight += self.ALPHA * error[i] * output[i]
                self.bias_weight += self.ALPHA * error[i] * output[i]

    def predict(self, dollars):
        euro = self.weight * dollars + self.bias_weight
        return euro


ALPHA = 0.00000001
NUM_EPOCHS = 600
TRAINING_SIZE = 200

TEST_DOLLARS = 118
COURSE_RATIO = 1.18

TRAINING_INPUTS = np.random.randint(100, size=TRAINING_SIZE)
TRAINING_OUTPUTS = np.array(tuple(map(lambda c: c / COURSE_RATIO,  TRAINING_INPUTS.copy())))

neural_network = NeuralNetwork(ALPHA)

print("Начальные веса нейронной сети:", neural_network.weight, neural_network.bias_weight, "", sep="\n")

neural_network.train(TRAINING_INPUTS, TRAINING_OUTPUTS, NUM_EPOCHS)

print(f"Веса нейронной сети после обучения: \n {neural_network.weight} , {neural_network.bias_weight}", "\n")

print(f"Тестирование сети: {TEST_DOLLARS} долларов",
      f"Значение, полученное на выходе сети: {round(neural_network.predict(TEST_DOLLARS))} евро",
      sep="\n")

import matplotlib.pyplot as plt
plt.xlabel("Эпоха") # ось абсцисс
plt.ylabel("Средняя ошибка") # ось ординат
plt.plot(neural_network.loss_array_epoch, neural_network.loss_array)
plt.show()