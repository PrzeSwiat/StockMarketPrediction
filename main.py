import numpy as np

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

dates, prices = helper.split_data(sub_set)

dates = np.array(dates)
prices = np.array(prices)


# Utworzenie i trening modelu MLP
mlp = MLP(input_size=1, hidden_size1=10, hidden_size2=5, output_size=1, learning_rate=0.01, epochs=100)
mlp.train(dates, prices)

last_date = dates[-1]
next_day = helper.get_next_day(last_date)
# Przewidywanie cen dla nowych danych
new_dates = helper.get_next_60_days(next_day)

# Przewidywanie cen dla nowych dat
predicted_prices = mlp.predict(new_dates, prices)

print("dates by def", dates)
print("prices by def:", prices)
# Wyświetlanie przewidywanych cen
print("Predicted Prices for New Dates:", predicted_prices)
print(new_dates)


merged = helper.merge_data(new_dates, predicted_prices)

plotDrawer.plot_two_datasets(sub_set, merged)

#plotDrawer.plot_one_dataset(sub_set)
#'''