import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

# the csv file is at the following url: https://ici.radio-canada.ca/info/2020/coronavirus-covid-19-pandemie-cas-carte-maladie-symptomes-propagation/index-en.html

# We read our CSV data file skipping the first 6 rows using pandas

data = pd.read_csv('quebec.csv', skiprows=6)

# We then transform our input into a 2D array

data_matrix = data.to_numpy()

montreal_rows=[]

# Here, we get all the rows whose location indicate Montreal and add them to a list

for row in range(len(data_matrix)):
    if data_matrix[row][1] == "Montréal":
        montreal_rows.append(data_matrix[row])


# We create a file containing the information specifically for Montreal

header = ['Date', 'Geo', 'Confirmed', 'Deaths', 'Recovered', 'Hospitalizations', 'ICU', 'Tested', 'Positivity',
            'Vaccinated', 'Vaccinated 2', 'Vac Distributed', 'Total variants', 'Variant B.1.1.7', 'Variant B.1.351',
            'Variant P.1', 'Variant Nigeria', 'Variant unknown', 'Variant B.1.617', 'Variant B.1.1.529', 'Vaccinated 1',
            'Vaccinated 3']

with open('montreal_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(montreal_rows)






