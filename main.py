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

#csvController.CreateCsvFromUrl(url1)

#'''
file_path = 'output.csv'
loaded_data = csvController.load_csv_data(file_path)

sub_set = helper.select_random_subset(loaded_data, 100)

dates, prices, changes, rois = helper.split_data(sub_set)

prices = np.array(prices)
changes = np.array(changes)
rois = np.array(rois)

norm_prices, min_price_value, max_price_value = helper.normalize_data(prices)
norm_changes, min_change_value, max_change_value = helper.normalize_data(changes)
norm_rois, min_roi_value, max_roi_value = helper.normalize_data(rois)

#           -----    MLP   -----
start_time = time.time()
# Utworzenie i trening modelu MLP
mlp = MLP(0.01, 10000, 0.9)
inputsArray, firstToPredict = helper.prepareInputMLP(norm_prices, norm_changes, norm_rois)
outputsArray = helper.prepareOutputMLP(norm_prices)
mlp.train(inputsArray, outputsArray)

end_time = time.time()
execution_time = end_time - start_time
print(f"Czas uczenia sieci MLP: {execution_time} sekundy")
# Przewidywanie dla kolejnych 'N' dni
number_of_days = 17
predicted_prices = mlp.predict_for_days(firstToPredict, number_of_days)
denormalized_prices = helper.denormalize_data(predicted_prices, min_price_value, max_price_value)

# Tworzenie dat dla 'N' kolejnych dni
last_date = dates[-1]
next_day = helper.get_next_day(last_date)
new_dates = helper.get_next_days(next_day, number_of_days)

# Wyświetlanie przewidywanych cen i kolejnych dat
print("Predicted Prices for New Dates:", denormalized_prices)
print("new datas:", new_dates)

merged = helper.merge_data(new_dates, denormalized_prices)
plotDrawer.plot_two_datasets(sub_set, merged)
