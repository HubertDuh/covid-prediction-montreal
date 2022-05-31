import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import operator
from sklearn.neural_network import MLPRegressor

# We will first create 3 matrix with the information we want: confirmed, deaths and recovered

confirmed_df = pd.read_csv(r'/Users/hubert/PycharmProjects/covid-prediction-mtl/data/clean_montreal_data.csv',
                           usecols=['Date', 'Days', 'Confirmed'])
confirmed_df.insert(2, 'Confirmed', confirmed_df.pop('Confirmed'))

deaths_df = pd.read_csv(r'/Users/hubert/PycharmProjects/covid-prediction-mtl/data/clean_montreal_data.csv',
                        usecols=['Date', 'Days', 'Deaths'])
deaths_df.insert(2, 'Deaths', deaths_df.pop('Deaths'))

recovered_df = pd.read_csv(r'/Users/hubert/PycharmProjects/covid-prediction-mtl/data/clean_montreal_data.csv',
                           usecols=['Date', 'Days', 'Recovered'])
recovered_df.insert(2, 'Recovered', recovered_df.pop('Recovered'))


class PolynomialRegressionModel:

    def __init__(self, model_name, polynomial_degree):
        self.__model_name = model_name
        self.__polynomial_degree = polynomial_degree
        self.__model = None

    def train(self, x, y):
        polynomial_features = PolynomialFeatures(degree=self.__polynomial_degree)
        x_poly = polynomial_features.fit_transform(x)
        self.__model = LinearRegression()
        self.__model.fit(x_poly, y)

    def get_predictions(self, x):
        polynomial_features = PolynomialFeatures(degree=self.__polynomial_degree)
        x_poly = polynomial_features.fit_transform(x)
        return np.round(self.__model.predict(x_poly), 0).astype(np.int32)

    def get_model_polynomial_str(self):
        coef = self.__model.coef_
        intercept = self.__model.intercept_
        poly = "{0:.3f}".format(intercept)

        for i in range(1, len(coef)):
            if coef[i] >= 0:
                poly += " + "
            else:
                poly += " - "
            poly += "{0:.3f}".format(coef[i]).replace("-", "") + "X^" + str(i)

        return poly


training_set = confirmed_df
x = np.array(training_set["Days"]).reshape(-1, 1)
y = training_set["Confirmed"]

training_set_deaths = deaths_df
x_deaths = np.array(training_set_deaths["Days"]).reshape(-1, 1)
y_deaths = training_set_deaths["Deaths"]

regression_model = PolynomialRegressionModel("Cases using Polynomial Regression", 2)
regression_model.train(x, y)

y_pred = regression_model.get_predictions(x)


def print_forecast(model_name, model, beginning_day=0, limit=10):
    next_days_x = np.array(range(beginning_day, beginning_day + limit)).reshape(-1, 1)
    next_days_pred = model.get_predictions(next_days_x)

    print("The forecast for " + model_name + " in the following " + str(limit) + " days is:")
    for i in range(0, limit):
        print("Day " + str(i + 1) + ": " + str(next_days_pred[i]))


def plot_graph(model_name, x, y, y_pred):
    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_pred), key=sort_axis)
    x, y_pred = zip(*sorted_zip)

    plt.plot(x, y_pred, color='m')
    plt.title("Amount of " + model_name + " in each day")
    plt.xlabel("Day")
    plt.ylabel(model_name)
    plt.show()


class NeuralNetModel:

    def __init__(self, model_name):
        self.__model_name = model_name
        self.__model = None

    def train(self, x, y, hidden_layer_sizes=[10, ], learning_rate=0.001, max_iter=2000):
        self.__model = MLPRegressor(solver="adam", activation="relu", alpha=1e-5, random_state=0,
                                    hidden_layer_sizes=hidden_layer_sizes, verbose=False, tol=1e-5,
                                    learning_rate_init=learning_rate, max_iter=max_iter)
        self.__model.fit(x, y)

    def get_predictions(self, x):
        return np.round(self.__model.predict(x), 0).astype(np.int32)


