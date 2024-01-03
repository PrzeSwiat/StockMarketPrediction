import random
from datetime import timedelta

import numpy as np


def calculate_moving_average(data, window_size=14):
    moving_avg = []

    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        moving_avg.append(avg)

    return moving_avg


def cut_first_rows(array, number):
    return array[number:]


def prepareInputMLP(array1, array2, array3):
    combined_tab = []
    for i in range(len(array1) - 1):
        # Tworzenie wiersza z trzech elementów
        row = [array1[i], array2[i], array3[i]]

        # Dodawanie wiersza do combined_tab
        combined_tab.append(row)
    last_row = [array1[-1], array2[-1], array3[-1]]
    return combined_tab, last_row


def prepareOutputMLP(array):
    return array[1:]


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def min_max_scaling(data, min_range=-1, max_range=1):
    data_array = np.array(data)

    min_val = np.min(data_array)
    max_val = np.max(data_array)

    scaled_data = min_range + (max_range - min_range) * (data_array - min_val) / (max_val - min_val)

    return scaled_data, min_val, max_val


def inverse_min_max_scaling(scaled_data, min_val, max_val, min_range=-1, max_range=1):
    original_data = min_val + (scaled_data - min_range) * (max_val - min_val) / (max_range - min_range)
    return original_data


def normalize_data(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    normalized_data = (data - min_values) / (max_values - min_values)

    return normalized_data, min_values, max_values


def normalize_data_on_given_minmax(data, mina, maxa):
    normalized_data = (data - mina) / (maxa - mina)
    return normalized_data


def denormalize_data(normalized_data, min_values, max_values):
    normalized_data = np.array(normalized_data)
    min_values = np.array(min_values)
    max_values = np.array(max_values)

    denormalized_data = normalized_data * (max_values - min_values) + min_values
    return denormalized_data


def calculate_roi(current_value, cost):
    if cost == 0:
        return 0
    roi = (current_value - cost) / cost
    return roi


def calculate_rsi(prices, period=14):
    # period - number of days to calculate RSI. Optimal value = 14 days
    rsi_values = []

    for i in range(period, len(prices)):
        gains = losses = 0

        for j in range(i - period, i):
            price_diff = prices[j + 1] - prices[j]
            if price_diff > 0:
                gains += price_diff
            elif price_diff < 0:
                losses += abs(price_diff)

        average_gain = gains / period
        average_loss = losses / period

        if average_loss == 0:
            rs = 100
        else:
            rs = average_gain / average_loss

        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return rsi_values


def calculate_change(current_value, cost):
    change = current_value - cost
    return change


def select_random_subset(sample_data, subset_size=100, number_of_dates=10):
    if not sample_data or subset_size + number_of_dates <= 0:
        return []

    max_start_index = max(0, len(sample_data) - subset_size + number_of_dates)
    start_index = random.randint(0, max_start_index)

    training_subset = sample_data[start_index:start_index + subset_size]
    output_subset = sample_data[start_index:start_index + subset_size + number_of_dates]
    return training_subset, output_subset


def select_last_subset(sample_data, subset_size=100, number_of_dates=10):
    if not sample_data or subset_size <= 0:
        return []

    # Wyłonienie ostatnich 100 kolejnych wierszy
    selected_subset = sample_data[-subset_size:]
    output_subset = sample_data[-subset_size - number_of_dates:]
    return selected_subset, output_subset


def split_data(data):
    dates = [entry['Date'] for entry in data]
    prices = [entry['Price'] for entry in data]
    changes = [entry['Change'] for entry in data]
    rois = [entry['ROI'] for entry in data]
    return dates, prices, changes, rois


def get_next_day(last_date):
    next_day = last_date + timedelta(days=1)
    return next_day


def get_next_days(start_date, days):
    next_days = [start_date + timedelta(days=i) for i in range(1, days + 1)]
    return next_days


def merge_data(dates, prices):
    merged_data = [{'Date': date, 'Price': price} for date, price in zip(dates, prices)]
    return merged_data


def calculate_accuracy(original_prices, predicted_prices, thresholding):
    correct_predictions = 0
    length = len(predicted_prices)
    for i in range(len(original_prices)):
        if abs(original_prices[i]['Price'] - predicted_prices[i]['Price']) <= thresholding:
            correct_predictions += 1
    return correct_predictions / length


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_string = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return time_string
