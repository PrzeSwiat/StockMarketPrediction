import random


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
