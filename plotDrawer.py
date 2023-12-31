import matplotlib.pyplot as plt


def draw_plot(data):
    plt.plot(data, linestyle='-', marker='', color='b')
    # plt.ylim(bottom=0, top=100)
    plt.title('data on count')
    plt.xlabel('count')
    plt.ylabel('data')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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


def plot_two_datasets(data1, data2, linewidth):
    dates1 = [item['Date'] for item in data1]
    prices1 = [item['Price'] for item in data1]

    dates2 = [item['Date'] for item in data2]
    prices2 = [item['Price'] for item in data2]

    # Tworzenie obiektu rysunku i obszaru rysowania
    fig, ax = plt.subplots()

    # Rysowanie drugiego wykresu w czerwonym kolorze
    ax.plot(dates2, prices2, linestyle='-', marker='', color='r', label='Predicted data', linewidth=linewidth)
    # Rysowanie pierwszego wykresu w niebieskim kolorze
    ax.plot(dates1, prices1, linestyle='-', marker='', color='b', label='Original data', linewidth=linewidth)

    ax.set_title('Wykres Ceny w Czasie')
    ax.set_xlabel('Data')
    ax.set_ylabel('Cena')
    ax.tick_params(rotation=45)
    ax.grid(True)
    ax.legend()  # Dodanie legendy
    fig.tight_layout()  # Dostosowanie marginesów, aby uniknąć obcięcia tekstu osi daty
    plt.savefig('Wykres Ceny w Czasie.png', dpi=1000)
    plt.show()
