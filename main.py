import csvController
import helper
import plotDrawer

url1 = "https://steamcommunity.com/market/listings/730/Sticker%20Capsule"
url2 = "https://steamcommunity.com/market/listings/730/Falchion%20Case"

#csvController.CreateCsvFromUrl(url1)

#'''
file_path = 'output.csv'
loaded_data = csvController.load_csv_data(file_path)

sub_set = helper.select_random_subset(loaded_data, 100)

#plotDrawer.plot_two_datasets(sub_set[0],sub_set[1])
plotDrawer.plot_one_dataset(sub_set)
#'''