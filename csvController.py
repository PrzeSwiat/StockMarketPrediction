import json
import requests
from bs4 import BeautifulSoup
import re
import csv
from datetime import datetime


def parse_date(date_str):
    return datetime.strptime(date_str, '%b %d %Y %H')


def load_csv_data(file_path):
    data_array = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        next(csv_reader)
        for row in csv_reader:
            date_str, value1 = row
            date = parse_date(date_str)

            data_array.append({
                'Date': date,
                'Price': float(value1)
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

            csv_writer.writerow(["Date", "Price", "Number"])

            for item in line1_list:
                date_str = item[0].replace(': +0', '')
                item_date = datetime.strptime(date_str, '%b %d %Y %H')

                # Check if the data was collected at 01:00
                if item_date.hour == 1:
                    formatted_date_str = item_date.strftime('%b %d %Y %H')
                    csv_writer.writerow([formatted_date_str, item[1]])
            print(f"CSV file created successfully: {csv_file_path}")
    else:
        print("No match found.")
