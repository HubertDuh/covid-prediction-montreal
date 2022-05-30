import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

# the csv file is at the following url: https://ici.radio-canada.ca/info/2020/coronavirus-covid-19-pandemie-cas-carte-maladie-symptomes-propagation/index-en.html

data = pd.read_csv("quebec.csv", skiprows=6)
print(data)

col_list = ['Date', 'Geo', 'Confirmed', 'Deaths', 'Recovered', 'Hospitalizations', 'ICU', 'Tested', 'Positivity',
            'Vaccinated', 'Vaccinated 2', 'Vac Distributed', 'Total variants', 'Variant B.1.1.7', 'Variant B.1.351',
            'Variant P.1', 'Variant Nigeria', 'Variant unknown', 'Variant B.1.617', 'Variant B.1.1.529', 'Vaccinated 1',
            'Vaccinated 3']
df = pd.read_csv("quebec.csv", skiprows=6, usecols=col_list)

for i in range(len(data.index)):



