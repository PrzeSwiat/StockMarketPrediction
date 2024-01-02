import time

import numpy as np

import helper
import plotDrawer
from MLP import MLP


def trainMLP(origin_prices, origin_dates, original_data, mas, norm_changes, rsis, prices, norm_prices, min_change_value, max_change_value, min_price_value, max_price_value, dates, number_of_days_to_predict=10, subset_size=300):
    start_time = time.time()
    # Utworzenie i trening modelu MLP
    mlp = MLP(0.01, 5000, 0.9)
    inputsArray, firstToPredict = helper.prepareInputMLP(mas, norm_changes, rsis)
    outputsArray = helper.prepareOutputMLP(norm_prices)
    # plotDrawer.draw_plot(norm_changes)
    # plotDrawer.draw_plot(rsis)
    # plotDrawer.draw_plot(mas)
    mlp.train(inputsArray, outputsArray)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas uczenia sieci MLP: {execution_time} sekundy")

    # Przewidywanie dla kolejnych 'N' dni
    predicted_prices, new_rsis = mlp.predict_for_days(firstToPredict, number_of_days_to_predict, norm_prices,
                                                      min_change_value, max_change_value, min_price_value,
                                                      max_price_value)

    # Tworzenie dat dla 'N' kolejnych dni
    new_dates = helper.get_next_days(dates[-1], number_of_days_to_predict)

    # Wyświetlanie przewidywanych cen i kolejnych

    predicted_prices = np.array(predicted_prices)
    predicted_with_origin = np.concatenate((origin_prices[:subset_size], predicted_prices))
    merged = helper.merge_data(origin_dates, predicted_with_origin)
    plotDrawer.plot_two_datasets(original_data, merged, 1)
    print("prediction: ", merged[-number_of_days_to_predict:])
    print("original: ", original_data[-number_of_days_to_predict:])
    return merged, original_data
