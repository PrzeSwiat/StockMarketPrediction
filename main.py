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
number_of_days_to_predict = 15  # optimal = 10
sub_set, original_data = helper.select_random_subset(loaded_data, 100, number_of_days_to_predict)

dates, prices, changes, rois = helper.split_data(sub_set)
origin_dates, origin_prices, origin_changes, origin_rois = helper.split_data(original_data)

prices = np.array(prices)
changes = np.array(changes)
rois = np.array(rois)
origin_prices = np.array(origin_prices)

norm_prices, min_price_value, max_price_value = helper.normalize_data(prices)
norm_changes, min_change_value, max_change_value = helper.normalize_data(changes)
norm_rois, min_roi_value, max_roi_value = helper.normalize_data(rois)
rsis = helper.calculate_rsi(norm_prices, 10)  # liczenie wartości RSI dla cen
rsis = [0] * (len(prices) - len(rsis)) + rsis
norm_rsis, min_rsi_value, max_rsi_value = helper.normalize_data(rsis)

#           -----    MLP   -----
start_time = time.time()
# Utworzenie i trening modelu MLP
mlp = MLP(0.01, 5000, 0.9)
inputsArray, firstToPredict = helper.prepareInputMLP(norm_prices, norm_changes, norm_rsis)
outputsArray = helper.prepareOutputMLP(norm_prices)
mlp.train(inputsArray, outputsArray)

end_time = time.time()
execution_time = end_time - start_time
print(f"Czas uczenia sieci MLP: {execution_time} sekundy")

# Przewidywanie dla kolejnych 'N' dni

predicted_prices, new_rsis = mlp.predict_for_days(firstToPredict, number_of_days_to_predict, norm_prices)
denormalized_prices = helper.denormalize_data(predicted_prices, min_price_value, max_price_value)

# Tworzenie dat dla 'N' kolejnych dni
last_date = dates[-1]
next_day = helper.get_next_day(last_date)
new_dates = helper.get_next_days(next_day, number_of_days_to_predict)

# Wyświetlanie przewidywanych cen i kolejnych dat
print("Predicted Prices for New Dates:", denormalized_prices)
print("new datas:", new_dates)

merged = helper.merge_data(new_dates, denormalized_prices)
plotDrawer.plot_two_datasets(original_data, merged)

plotDrawer.draw_plot(changes)
plotDrawer.draw_plot(rsis)
plotDrawer.draw_plot(new_rsis)
