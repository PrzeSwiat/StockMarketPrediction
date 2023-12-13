import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def tanh(x):
    return np.tanh(x)


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1, epochs=10):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicjalizacja wag
        self.weights_input_hidden1 = np.random.normal(0, 0.1, size=(input_size, hidden_size1))
        self.weights_hidden1_hidden2 = np.random.normal(0, 0.1, size=(hidden_size1, hidden_size2))
        self.weights_hidden2_output = np.random.normal(0, 0.1, size=(hidden_size2, output_size))

        # Parametry normalizacji
        self.min_date = None
        self.ptp_date = None
        self.min_price = None
        self.ptp_price = None

    def normalize_data(self, dates, prices):
        dates_numeric = [(date - datetime(1970, 1, 1)).total_seconds() for date in dates]
        self.min_date = np.min(dates_numeric)
        self.ptp_date = np.ptp(dates_numeric)

        self.min_price = np.min(prices)
        self.ptp_price = np.ptp(prices)

        dates_normalized = (dates_numeric - self.min_date) / self.ptp_date
        prices_normalized = (prices - self.min_price) / self.ptp_price

        return dates_normalized, prices_normalized

    def train(self, dates, prices, min_error=0.0001):
        dates_normalized, prices_normalized = self.normalize_data(dates, prices)
        # Convert datetime to numerical values (timestamps)
        dates_numeric = [(date - datetime(1970, 1, 1)).total_seconds() for date in dates]
        previous_error = float('inf')
        # Normalization and scaling
        dates_scaled = (np.array(dates_numeric) - np.min(dates_numeric)) / np.ptp(dates_numeric)
        dates_scaled = 2 * dates_scaled - 1  # Scale to the range [-1, 1]

        prices_normalized = (prices - np.min(prices)) / np.ptp(prices)

        for epoch in range(self.epochs):
            # Propagacja do przodu (Forward pass)
            hidden_layer1_input = np.dot(dates_normalized.reshape(-1, 1), self.weights_input_hidden1)
            hidden_layer1_output = tanh(hidden_layer1_input)

            hidden_layer2_input = np.dot(hidden_layer1_output, self.weights_hidden1_hidden2)
            hidden_layer2_output = tanh(hidden_layer2_input)

            output_layer_input = np.dot(hidden_layer2_output, self.weights_hidden2_output)
            predicted_prices = tanh(output_layer_input)

            # Obliczanie błędu
            error = prices_normalized.reshape(-1, 1) - predicted_prices
            if np.abs(np.mean(error)) < min_error:
                break
            if np.abs(np.mean(error) - previous_error) < min_error:
                break

            previous_error = np.mean(error)
            # Propagacja wsteczna (Backpropagation)
            output_error = error * tanh_derivative(predicted_prices)

            hidden_layer2_error = output_error.dot(self.weights_hidden2_output.T) * tanh_derivative(
                hidden_layer2_output)

            hidden_layer1_error = hidden_layer2_error.dot(self.weights_hidden1_hidden2.T) * tanh_derivative(
                hidden_layer1_output)

            # Aktualizacja wag
            self.weights_hidden2_output += hidden_layer2_output.T.dot(output_error) * self.learning_rate
            self.weights_hidden1_hidden2 += hidden_layer1_output.T.dot(hidden_layer2_error) * self.learning_rate
            self.weights_input_hidden1 += dates_scaled.reshape(-1, 1).T.dot(hidden_layer1_error) * self.learning_rate

    def predict(self, dates, original_prices):
        dates_normalized, _ = self.normalize_data(dates, original_prices)

        # Forward pass
        hidden_layer1_input = np.dot(dates_normalized.reshape(-1, 1), self.weights_input_hidden1)
        hidden_layer1_output = tanh(hidden_layer1_input)

        hidden_layer2_input = np.dot(hidden_layer1_output, self.weights_hidden1_hidden2)
        hidden_layer2_output = tanh(hidden_layer2_input)

        output_layer_input = np.dot(hidden_layer2_output, self.weights_hidden2_output)
        predicted_prices_normalized = tanh(output_layer_input)

        # Denormalization
        predicted_prices_denormalized = predicted_prices_normalized * self.ptp_price + self.min_price

        return predicted_prices_denormalized
