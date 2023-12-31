import time

import numpy as np
from pyparsing import helpers

import csvController
import helper
import plotDrawer
from datetime import datetime

from MLP import MLP

url1 = "https://steamcommunity.com/market/listings/730/Sticker%20Capsule"
url2 = "https://steamcommunity.com/market/listings/730/Falchion%20Case"

# csvController.CreateCsvFromUrl(url1)

# '''
file_path = 'output.csv'
loaded_data = csvController.load_csv_data(file_path)
number_of_days_to_predict = 10  # optimal = 3
sub_set, original_data = helper.select_random_subset(loaded_data, 100, number_of_days_to_predict)

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

#           -----    MLP   -----
start_time = time.time()
# Utworzenie i trening modelu MLP
mlp = MLP(0.01, 5000, 0.9)
inputsArray, firstToPredict = helper.prepareInputMLP(mas, norm_changes, rsis)
outputsArray = helper.prepareOutputMLP(prices)
plotDrawer.draw_plot(norm_changes)
plotDrawer.draw_plot(rsis)
plotDrawer.draw_plot(mas)
mlp.train(inputsArray, outputsArray)

end_time = time.time()
execution_time = end_time - start_time
print(prices[-1], changes[-1], rsis[-1], dates[-1])
print(f"Czas uczenia sieci MLP: {execution_time} sekundy")

# Przewidywanie dla kolejnych 'N' dni
predicted_prices, new_rsis = mlp.predict_for_days(firstToPredict, number_of_days_to_predict, norm_prices, min_change_value, max_change_value, min_price_value, max_price_value)

# Tworzenie dat dla 'N' kolejnych dni
new_dates = helper.get_next_days(dates[-1], number_of_days_to_predict)

# Wyświetlanie przewidywanych cen i kolejnych
print("prediction: ", predicted_prices)
print("original: ", original_data[-3], original_data[-2], original_data[-1])
merged = helper.merge_data(new_dates, predicted_prices)
plotDrawer.plot_two_datasets(original_data, merged)

