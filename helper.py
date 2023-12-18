import random
from datetime import timedelta

import numpy as np


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


def normalize_data(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    normalized_data = (data - min_values) / (max_values - min_values)

    return normalized_data, min_values, max_values


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


def calculate_change(current_value, cost):
    change = current_value - cost
    return change


def select_random_subset(sample_data, subset_size=100):
    if not sample_data or subset_size <= 0:
        return []

    max_start_index = max(0, len(sample_data) - subset_size)
    start_index = random.randint(0, max_start_index)

    training_subset = sample_data[start_index:start_index + subset_size]

    return training_subset


def select_last_subset(sample_data, subset_size=100):
    if not sample_data or subset_size <= 0:
        return []

    # Wyłonienie ostatnich 100 kolejnych wierszy
    selected_subset = sample_data[-subset_size:]

    return selected_subset


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
