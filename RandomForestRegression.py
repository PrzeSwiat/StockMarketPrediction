import numpy as np
from sklearn.ensemble import RandomForestRegressor

import helper


class RandomForestRegression:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=100, min_samples_split=10, min_samples_leaf=10, min_weight_fraction_leaf=0.00001, max_features=1.0, max_leaf_nodes=10)

    def train(self, prices):
        inputsArray, firstToPredict = helper.prepare_input_prices(prices)
        outputsArray = helper.prepare_output_prices(prices)
        self.rf.fit(inputsArray, outputsArray)
        # score = self.linreg.score(inputsArray, outputsArray)
        # print(score)
        return firstToPredict

    def predict_next(self, next_to_predict, last_n_prices):
        next_day_price = self.rf.predict([next_to_predict])
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
            first_input = [prices[-3], prices[-2], prices[-1]]
        return outputs