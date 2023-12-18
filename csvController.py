import json
import requests
from bs4 import BeautifulSoup
import re
import csv
from datetime import datetime

import helper


def parse_date(date_str):
    return datetime.strptime(date_str, '%b %d %Y %H')


def load_csv_data(file_path):
    data_array = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        next(csv_reader)
        for row in csv_reader:
            date_str, price, number, roi, change = row
            date = parse_date(date_str)

            data_array.append({
                'Date': date,
                'Price': float(price),
                'ROI': float(roi),
                'Change': float(change)
            })

    return data_array


def CreateCsvFromUrl(url):
    result = requests.get(url)

    if result.status_code != 200:
        print("Error occurred. Status Code: " + str(result.status_code))
    else:
        print("Access permitted")

    soup = BeautifulSoup(result.text, "html.parser")
    match = re.search(r'var line1=(\[.*?\]);', soup.prettify(), re.DOTALL)

    if match:
        line1_data = match.group(1)

        line1_list = json.loads(line1_data)
        csv_file_path = "output.csv"
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(["Date", "Price", "Number", "ROI", "change"])

            previous_price = 0
            for i in range(len(line1_list)):
                item = line1_list[i]
                date_str = item[0].replace(': +0', '')
                item_date = datetime.strptime(date_str, '%b %d %Y %H')

                # Check if the data was collected at 01:00
                if item_date.hour == 1:
                    formatted_date_str = item_date.strftime('%b %d %Y %H')
                    current_price = item[1]
                    roi_value = helper.calculate_roi(current_price, previous_price) if i > 0 else 0
                    change_value = helper.calculate_change(current_price, previous_price) if i > 0 else 0
                    csv_writer.writerow([formatted_date_str, current_price, i + 1, roi_value, change_value])
                    previous_price = current_price

            print(f"CSV file created successfully: {csv_file_path}")
    else:
        print("No match found.")
