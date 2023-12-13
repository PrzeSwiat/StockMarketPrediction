import random
from datetime import timedelta


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
    return dates, prices


def get_next_day(last_date):
    next_day = last_date + timedelta(days=1)
    return next_day


def get_next_60_days(start_date):
    next_days = [start_date + timedelta(days=i) for i in range(1, 61)]
    return next_days


def merge_data(dates, prices):
    merged_data = [{'Date': date, 'Price': price} for date, price in zip(dates, prices)]
    return merged_data
