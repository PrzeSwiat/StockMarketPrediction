import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

import helper


class MLPModel:
    def __init__(self):
        # Standardize the data
        self.scaler = StandardScaler()
        self.mlp = MLPRegressor(hidden_layer_sizes=(264,), activation='logistic', max_iter=10000, random_state=42)

    def train(self, mas, norm_changes, rsis, prices):
        inputsArray, firstToPredict = helper.prepareInputMLP(mas, norm_changes, rsis)
        outputsArray = helper.prepareOutputMLP(prices)
        # Split the data into training and testing sets

        X_train, X_test, y_train, y_test = train_test_split(inputsArray, outputsArray, test_size=0.2, random_state=42)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # Create and train the MLP regressor model
        self.mlp.fit(X_train, y_train)
        # Evaluate the model using Mean Squared Error (MSE)
        y_pred = self.mlp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error (MSE): {mse}')
        accuracy = self.mlp.score(X_test, y_test)
        print(f'Model Accuracy: {accuracy}')

        return firstToPredict

    def predict_next(self, next_to_predict, last_n_prices):
        next_day_price = self.mlp.predict([next_to_predict])
        nextChange = helper.calculate_change(next_day_price, last_n_prices[-1])
        next_rsi = helper.calculate_rsi(last_n_prices, 14)
        next_ma = helper.calculate_moving_average(last_n_prices, 14)
        output = np.hstack((nextChange[0], next_day_price[0], next_rsi[-1], next_ma[-1]))
        return output

    def predict_for_days(self, first_input, days_to_predict, prices, min_price, max_price):
        outputs = []
        for i in range(days_to_predict):
            output = self.predict_next(first_input, prices)
            prices = np.append(prices, output[1])
            outputs.append(helper.denormalize_data(output[1], min_price, max_price))
            first_input = [output[3], output[0], output[2]]
        return outputs
