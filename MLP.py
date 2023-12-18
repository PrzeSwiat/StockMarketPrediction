import numpy as np

import helper


class MLP:
    def __init__(self, learning_rate=0.01, epochs=10000):
        # Inicjalizacja wag i biasów dla dwóch warstw sieci
        self.input_size = 3
        self.hidden_size = 6
        self.output_size = 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        self.output_layer_input = 0
        self.hidden_layer_output = 0
        self.hidden_layer_input = 0
        self.predicted_output = 0

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

        # Aktualizacja wag i biasów
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.bias_hidden_output += np.sum(d_output) * self.learning_rate

        self.weights_input_hidden += inputs.T.dot(d_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(d_hidden) * self.learning_rate

    def train(self, inputs, targets):
        for epoch in range(self.epochs):
            for i in range(len(inputs)):
                input_data = np.array(inputs[i], ndmin=2)
                target = np.array(targets[i], ndmin=2)

                output = self.forward_propagation(input_data)
                self.backward_propagation(input_data, target)

                if epoch % 1000 == 0:
                    loss = np.mean(np.square(target - self.predicted_output))
                    #print(f"Epoch: {epoch}, Loss: {loss}")

    def predict_next(self, next_input):
        nextPrice = self.forward_propagation(next_input)
        nextChange = helper.calculate_change(nextPrice, next_input[0])
        nextRoi = helper.calculate_roi(nextPrice, next_input[0])
        output = np.hstack((nextPrice[0], nextChange[0], nextRoi[0]))
        return output

    def predict_for_days(self, first_input, days_to_predict):
        outputs = []
        for i in range(days_to_predict):
            output = self.predict_next(first_input)
            outputs.append(output[0])
            first_input = output
        return outputs

