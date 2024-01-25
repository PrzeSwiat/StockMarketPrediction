import numpy as np
from sklearn.linear_model import Lasso

import helper


class LassoRegression:
    def __init__(self):
        self.lasso = Lasso(alpha=0.0001, tol=0.0001, max_iter=1000, selection='cyclic')

    def train(self, prices):
        inputsArray, firstToPredict = helper.prepareInputPrices(prices)
        outputsArray = helper.prepareOutputPrices(prices)
        self.lasso.fit(inputsArray, outputsArray)
        # score = self.linreg.score(inputsArray, outputsArray)
        # print(score)
        return firstToPredict

    def predict_next(self, next_to_predict, last_n_prices):
        next_day_price = self.lasso.predict([next_to_predict])
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

    def train6(self, prices, mas, changes, rsis):
        inputsArray, firstToPredict = helper.prepareInputPrices6(mas, changes, rsis, prices)
        outputsArray = helper.prepareOutputPrices6(prices)
        self.lasso.fit(inputsArray, outputsArray)
        # score = self.linreg.score(inputsArray, outputsArray)
        # print(score)
        return firstToPredict

    def predict_for_days_6(self, first_input, days_to_predict, prices, min_price, max_price):
        outputs = []
        for i in range(days_to_predict):
            output = self.predict_next6(first_input, prices, min_price, max_price)
            prices = np.append(prices, output[1])
            outputs.append(helper.denormalize_data(output[1], min_price, max_price))
            first_input = [prices[-3], prices[-2], prices[-1], output[2]]
        return outputs

    def predict_next6(self, next_to_predict, last_n_prices, min_price, max_price):
        next_day_price = self.lasso.predict([next_to_predict])
        denormprice = helper.denormalize_data(next_day_price, min_price, max_price)
        denormprices = helper.denormalize_data(last_n_prices, min_price, max_price)

        nextChange = helper.calculate_change(denormprice, denormprices[-1])
        next_rsi = helper.calculate_rsi(denormprices, 14)
        next_ma = helper.calculate_moving_average(denormprices, 14)
        output = np.hstack((nextChange[0], next_day_price[0], next_rsi[-1], next_ma[-1]))
        return output

    def train_by_rest(self, mas, norm_changes, rsis, prices):
        inputsArray, firstToPredict = helper.prepareInputRest(mas, norm_changes, rsis, prices)
        outputsArray = helper.prepareOutputRest(prices)
        self.lasso.fit(inputsArray, outputsArray)
        return firstToPredict

    def predict_next_by_rest(self, next_to_predict, last_n_prices):
        next_day_price = self.lasso.predict([next_to_predict])
        nextChange = helper.calculate_change(next_day_price, last_n_prices[-1])
        next_rsi = helper.calculate_rsi(last_n_prices, 14)
        next_ma = helper.calculate_moving_average(last_n_prices, 14)
        output = np.hstack((nextChange[0], next_day_price[0], next_rsi[-1], next_ma[-1]))
        return output

    def predict_for_days_by_rest(self, first_input, days_to_predict, prices, min_price, max_price):
        outputs = []
        for i in range(days_to_predict):
            output = self.predict_next_by_rest(first_input, prices)
            last_price = prices[-1]
            prices = np.append(prices, output[1])
            outputs.append(helper.denormalize_data(output[1], min_price, max_price))
            first_input = [output[3], output[0], output[2], last_price]
        return outputs
