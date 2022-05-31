import lmfit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import odeint
from lmfit import Parameters, minimize
from copy import deepcopy
from tqdm.auto import tqdm

# the csv file is at the following url: https://ici.radio-canada.ca/info/2020/coronavirus-covid-19-pandemie-cas-carte-maladie-symptomes-propagation/index-en.html

# We read our CSV data file skipping the first 6 rows using pandas

data = pd.read_csv('quebec.csv', skiprows=6)

# We then transform our input into a 2D array

data_matrix = data.to_numpy()

montreal_rows = []

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

montreal_data = pd.read_csv('montreal_data.csv')
montreal_data_matrix = montreal_data.to_numpy()

# It is to note here that pands has a built-in way (.diff()) to get the difference between two rows. However,
# since our data set has problems, I found it easier to just do it manually and use lists.

recovered_daily = [0]

days_rec = 1
while days_rec < len(montreal_data_matrix):
    difference_in_hosp = montreal_data_matrix[days_rec][4] - montreal_data_matrix[days_rec - 1][4]
    if difference_in_hosp < 0:  # the data has mistakes where the number of recovered diminishes for some reason
        recovered_daily.append(0)  # this condition tries to solve that. A better data set is needed
    else:
        recovered_daily.append(difference_in_hosp)
    days_rec += 1

montreal_data['Daily_Recovery'] = recovered_daily

cases_daily = [1]

days_case = 1
while days_case < len(montreal_data_matrix):
    difference_in_cases = montreal_data_matrix[days_case][2] - montreal_data_matrix[days_case - 1][2]
    if difference_in_cases < 0:  # the data has mistakes where the number of recovered diminishes for some reason
        cases_daily.append(0)  # this condition tries to solve that. A better data set is needed
    else:
        cases_daily.append(difference_in_cases)
    days_case += 1

montreal_data['Daily_Cases'] = cases_daily

deaths_daily = [0]

days_deaths = 1
while days_deaths < len(montreal_data_matrix):
    difference_in_deaths = montreal_data_matrix[days_deaths][3] - montreal_data_matrix[days_deaths - 1][3]
    if difference_in_deaths < 0:  # the data has mistakes where the number of recovered diminishes for some reason
        deaths_daily.append(0)  # this condition tries to solve that. A better data set is needed
    else:
        deaths_daily.append(difference_in_deaths)
    days_deaths += 1

montreal_data['Daily_Deaths'] = deaths_daily

# Here, we will clean up our dataframe so that it is cleaner and easier to work with to do SEIRD

montreal_data.drop(['Hospitalizations', 'ICU', 'Tested', 'Positivity', 'Vaccinated', 'Vaccinated 2', 'Vac Distributed',
                    'Total variants', 'Variant B.1.1.7', 'Variant B.1.351', 'Variant P.1', 'Variant Nigeria',
                    'Variant unknown', 'Variant B.1.617', 'Variant B.1.1.529', 'Vaccinated 1', 'Vaccinated 3'],
                   axis=1, inplace=True)


# Now, we build the barebone SEIR model

