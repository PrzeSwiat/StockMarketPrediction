import matplotlib.pyplot as plt


def plot_one_dataset(data):
    dates = [item['Date'] for item in data]
    prices = [item['Price'] for item in data]
    plt.plot(dates, prices, linestyle='-', marker='', color='b')
    plt.title('Wykres Ceny w Czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_two_datasets(data1, data2):
    dates1 = [item['Date'] for item in data1]
    prices1 = [item['Price'] for item in data1]

    dates2 = [item['Date'] for item in data2]
    prices2 = [item['Price'] for item in data2]

    # Rysowanie pierwszego wykresu w niebieskim kolorze
    plt.plot(dates1, prices1, linestyle='-', marker='', color='b', label='Training data')

    # Rysowanie drugiego wykresu w czerwonym kolorze
    plt.plot(dates2, prices2, linestyle='-', marker='', color='r', label='Output data')

    plt.title('Wykres Ceny w Czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()  # Dodanie legendy
    plt.tight_layout()  # Dostosowanie marginesów, aby uniknąć obcięcia tekstu osi daty
    plt.show()
