import time

import numpy as np
from pyparsing import helpers

import NetworkController
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

#       -------------------   defaults --------------------
number_of_days_to_predict = 60  # optimal = 3
subset_size = 500
thresholding_value = 0.01
#       -------------------   preparing data --------------------
sub_set, original_data = helper.select_random_subset(loaded_data, subset_size, number_of_days_to_predict)
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
print("Start")
NetworkController.trainMLP(origin_prices, origin_dates, original_data, mas, norm_changes, rsis, prices, norm_prices,
                           min_change_value, max_change_value, min_price_value, max_price_value, dates,
                           number_of_days_to_predict, subset_size)
print("End")
# calculate accuracy
