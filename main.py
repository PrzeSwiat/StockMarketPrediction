import time
from statistics import LinearRegression

import numpy as np
import sklearn
from pyparsing import helpers

import MLPModel
import NetworkController
import csvController
import helper
import plotDrawer
from datetime import datetime

from BayesianRidgeRegression import BayesianRidgeRegression
from LassoRegression import LassoRegression
from LinearRegresion import LinearRegressor
from SupportVectorRegression import SupportVectorRegression
from MLP import MLP

url1 = "https://steamcommunity.com/market/listings/730/Sticker%20Capsule"
url2 = "https://steamcommunity.com/market/listings/730/Falchion%20Case"

# csvController.CreateCsvFromUrl(url1)

# '''
file_path = 'output.csv'
loaded_data = csvController.load_csv_data(file_path)

#       -------------------   defaults --------------------
number_of_days_to_predict = 60  # optimal = 2
subset_size = 500
thresholding_value = 0.01
all_accuracies = []
all_RMSE = []
all_MAPE = []
rounds_of_training = 1
print("Start")
start_time = time.time()

for i in range(rounds_of_training):
    #       -------------------   preparing data --------------------
    sub_set, original_data = helper.select_last_subset(loaded_data, subset_size, number_of_days_to_predict)
    dates, prices, changes, rois = helper.split_data(sub_set)
    origin_dates, origin_prices, origin_changes, origin_rois = helper.split_data(original_data)

    prices = np.array(prices)
    changes = np.array(changes)
    rois = np.array(rois)
    origin_prices = np.array(origin_prices)

    norm_prices, min_price_value, max_price_value = helper.normalize_data(prices)
    norm_changes, min_change_value, max_change_value = helper.min_max_scaling(changes)
    rsis = helper.calculate_rsi(prices, 14)  # liczenie wartości RSI dla cen
    mas = helper.calculate_moving_average(prices, 14)
    rsis = [0] * (len(prices) - len(rsis)) + rsis
    mas = [0] * (len(prices) - len(mas)) + mas
    norm_mas, min_mas, max_mas = helper.normalize_data(mas)
    norm_rsis, min_rsi_value, max_rsi_value = helper.normalize_data(rsis)

    #       -------------------   MLP--------------------
    '''
    predicted_prices, original_prices = NetworkController.trainMLP(origin_prices, origin_dates, original_data, mas,
                                                                   norm_changes, rsis, prices, norm_prices,
                                                                   min_change_value, max_change_value, min_price_value,
                                                                   max_price_value, dates,
                                                                   number_of_days_to_predict, subset_size)
    
    # calculate accuracy
    accuracy = helper.calculate_accuracy(original_prices, predicted_prices, thresholding_value)
    all_accuracies.append(accuracy)
    '''

    # print("accuracy", accuracy)

    #       -------------------   MLPModel--------------------
    '''
    mlpmodel = MLPModel.MLPModel()
    first_input = mlpmodel.train(norm_prices)
    predicted_prices = mlpmodel.predict_for_days(first_input, number_of_days_to_predict, norm_prices, min_price_value, max_price_value)
    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    #plotDrawer.plot_two_datasets(original_data, merged, 1)
    '''
    #       -------------------   Linear Regression--------------------
    '''
    linreg = LinearRegressor()
    first_input = linreg.train(norm_prices)
    predicted_prices = linreg.predict_for_days(first_input, number_of_days_to_predict, norm_prices, min_price_value,
                                               max_price_value)
    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    # plotDrawer.plot_two_datasets(original_data, merged, 1)
    '''
    #       -------------------  Lasso Regression --------------------
    '''
    lasso = LassoRegression()
    first_input = lasso.train(norm_prices)
    predicted_prices = lasso.predict_for_days(first_input, number_of_days_to_predict, norm_prices, min_price_value,
                                               max_price_value)
    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    #plotDrawer.plot_two_datasets(original_data, merged, 1)
    '''
    #       -------------------  SupportVectorRegression --------------------
    '''
    svr = SupportVectorRegression()
    first_input = svr.train(norm_prices)
    predicted_prices = svr.predict_for_days(first_input, number_of_days_to_predict, norm_prices, min_price_value,
                                              max_price_value)
    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    # plotDrawer.plot_two_datasets(original_data, merged, 1)
    '''
    #       -------------------  BayesianRidge --------------------

    brr = BayesianRidgeRegression()
    first_input = brr.train(norm_prices)
    predicted_prices = brr.predict_for_days(first_input, number_of_days_to_predict, norm_prices, min_price_value,
                                            max_price_value)
    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    plotDrawer.plot_two_datasets(original_data, merged, 0.3)

    RMSE = sklearn.metrics.mean_squared_error(origin_prices[:number_of_days_to_predict], predicted_prices) # Root Mean Square Error
    MAPE = sklearn.metrics.mean_absolute_percentage_error(origin_prices[:number_of_days_to_predict], predicted_prices) # Mean Absolute Percentage Error
    if MAPE > 1:    # mean_absolute_percentage_error function in sklearn library returns in range of [0,100]. if not, 1e2 notation occurred (creator's joke?)
        MAPE = 0
    accuracy = helper.calculate_accuracy(original_data[-number_of_days_to_predict:],
                                         merged[-number_of_days_to_predict:], thresholding_value)

    all_accuracies.append(accuracy)
    all_RMSE.append(RMSE)
    all_MAPE.append(MAPE)

end_time = time.time()
execution_time = end_time - start_time
hms_time = helper.seconds_to_hms(execution_time)
print("End")
print(f"Czas testowania sieci: {hms_time}")

total_accuracy = helper.calculate_average(all_accuracies)
total_RMSE = helper.calculate_average(all_RMSE)
total_MAPE = helper.calculate_average(all_MAPE)

print("Total accuracy", total_accuracy)
#print("All accuracies", all_accuracies)
print("Total RMSE", total_RMSE)
print("Total MAPE", total_MAPE)
