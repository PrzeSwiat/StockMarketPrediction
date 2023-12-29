import numpy as np
from matplotlib import pyplot as plt

import helper


class MLP:
    def __init__(self, learning_rate=0.01, epochs=10000, momentum=0.25):
        # Inicjalizacja wag i biasów dla dwóch warstw sieci
        self.input_size = 2
        self.hidden_size = 64
        self.output_size = 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))

        self.output_layer_input = 0
        self.hidden_layer_output = 0
        self.hidden_layer_input = 0
        self.predicted_output = 0

        # Inicjalizacja zmiennych momentum
        self.momentum_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.momentum_bias_input_hidden = np.zeros_like(self.bias_input_hidden)
        self.momentum_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.momentum_bias_hidden_output = np.zeros_like(self.bias_hidden_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Propagacja sygnału do przodu
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

        return self.predicted_output

    def backward_propagation(self, inputs, target):
        # Obliczanie błędu na wyjściu
        output_error = target - self.predicted_output
        d_output = output_error * self.sigmoid_derivative(self.predicted_output)

        # Obliczanie błędu w warstwie ukrytej
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Aktualizacja wag i biasów z uwzględnieniem momentum
        self.momentum_weights_hidden_output = (self.momentum * self.momentum_weights_hidden_output +
                                               self.hidden_layer_output.T.dot(d_output) * self.learning_rate)
        self.weights_hidden_output += self.momentum_weights_hidden_output

        self.momentum_bias_hidden_output = (self.momentum * self.momentum_bias_hidden_output +
                                            np.sum(d_output) * self.learning_rate)
        self.bias_hidden_output += self.momentum_bias_hidden_output

        self.momentum_weights_input_hidden = (self.momentum * self.momentum_weights_input_hidden +
                                              inputs.T.dot(d_hidden) * self.learning_rate)
        self.weights_input_hidden += self.momentum_weights_input_hidden

        self.momentum_bias_input_hidden = (self.momentum * self.momentum_bias_input_hidden +
                                           np.sum(d_hidden) * self.learning_rate)
        self.bias_input_hidden += self.momentum_bias_input_hidden

    def train(self, inputs, targets):
        losses = []
        for epoch in range(self.epochs):
            for i in range(len(inputs)):
                input_data = np.array(inputs[i], ndmin=2)
                target = np.array(targets[i], ndmin=2)

                output = self.forward_propagation(input_data)
                self.backward_propagation(input_data, target)

                if i == len(inputs)-1:
                    loss = np.mean(np.square(target - output))
                    losses.append(loss)


                #print(f"Epoch: {epoch}, Loss: {loss}")

        plt.plot(losses, linestyle='-', marker='', color='b')
        plt.title('loss during epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def predict_next(self, next_input, last_n_prices, min_change, max_change, min_price, max_price):
        nextChange = self.forward_propagation(next_input)
        nextChange = helper.inverse_min_max_scaling(nextChange, min_change, max_change)
        #print(nextChange)
        next_rsi = helper.calculate_rsi(last_n_prices, 14)
        last_price = helper.denormalize_data(last_n_prices[-1], min_price, max_price)
        print(last_price)
        next_price = last_price + nextChange
        #print(next_rsi[-1])
        output = np.hstack((nextChange[0], next_price[0], next_rsi[-1], last_n_prices[-1]))
        return output

    def predict_for_days(self, first_input, days_to_predict, prices, min_change, max_change, min_price, max_price):
        outputs = []
        new_rsis = []
        for i in range(days_to_predict):
            output = self.predict_next(first_input, prices, min_change, max_change, min_price, max_price)
            prices = np.append(prices, output[3])
            new_rsis.append(output[2])
            outputs.append(output[1])
            first_input = [output[0], output[2]]
        return outputs, new_rsis

