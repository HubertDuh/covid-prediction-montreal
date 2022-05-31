import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import timedelta, datetime

import Regression_and_ML.reg_and_ML

START_DATE = {'Montreal': '02/27/20'}
rho = 1


class Learner(object):
    """constructs an SIR model learner to load training set, train the model,
       and make predictions at country level.
    """

    def __init__(self, loss, start_date, predict_range, s_0, i_0, r_0):
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0

    def load_confirmed(self):
        """
        Load confirmed cases downloaded from pre-made dataframe
        """

        confirmed_series = pd.Series(Regression_and_ML.reg_and_ML.confirmed_df['Confirmed'].to_numpy(),
                                     index=Regression_and_ML.reg_and_ML.confirmed_df.Date)
        return confirmed_series

    def load_dead(self):
        """
        Load deaths downloaded from pre-made dataframe
        """

        deaths_series = pd.Series(Regression_and_ML.reg_and_ML.deaths_df['Deaths'].to_numpy(),
                                  index=Regression_and_ML.reg_and_ML.deaths_df.Date)
        return deaths_series

    def load_recovered(self):
        """
        Load recovered cases downloaded from pre-made dataframe
        """
        recovered_series = pd.Series(Regression_and_ML.reg_and_ML.recovered_df['Recovered'].to_numpy(),
                                     index=Regression_and_ML.reg_and_ML.recovered_df.Date)
        return recovered_series

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict_0(self, beta, gamma, data):
        """
        Simplifield version.
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        predict_range = 150
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-rho * beta * S * I, rho * beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        return new_index, extended_actual, solve_ivp(SIR, [0, size], [S_0, I_0, R_0], t_eval=np.arange(0, size, 1))

    def predict(self, beta, gamma, data, recovered, death, s_0, i_0, r_0):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-rho * beta * S * I, rho * beta * S * I - gamma * I, gamma * I]

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        solved = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1))
        return new_index, extended_actual, extended_recovered, extended_death, solved

    def train_0(self):
        """
        Simplified version.
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        data = self.load_confirmed()
        optimal = minimize(
            loss,
            [0.001, 0.001],
            args=(data),
            method='L-BFGS-B',
            bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
        )
        beta, gamma = optimal.x
        new_index, extended_actual, prediction = self.predict(beta, gamma, data)
        df = pd.DataFrame({
            'Atual': extended_actual,
            'S': prediction.y[0],
            'I': prediction.y[1],
            'R': prediction.y[2]
        }, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Montreal")
        df.plot(ax=ax)
        fig.savefig(f"montreal.png")

    def train(self):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        recovered = self.load_recovered()
        death = self.load_dead()
        data = (self.load_confirmed() - recovered - death)

        optimal = minimize(loss, [0.001, 0.001],
                           args=(data, recovered, self.s_0, self.i_0, self.r_0),
                           method='L-BFGS-B',
                           bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)

        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data,
                                                                                                  recovered, death,
                                                                                                  self.s_0,
                                                                                                  self.i_0, self.r_0)
        df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered,
                           'Death data': extended_death, 'Susceptible': prediction.y[0],
                           'Infected': prediction.y[1], 'Recovered': prediction.y[2]}, index=new_index)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Montreal")
        df.plot(ax=ax)
        print(f"city=Montreal, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta / gamma):.8f}")
        fig.savefig(f"Montreal.png")


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-rho * beta * S * I, rho * beta * S * I - gamma * I, gamma * I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data) ** 2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered) ** 2))
    alpha = 0.1

    return alpha * l1 + (1 - alpha) * l2


predict_range = 250
s_0 = 100000
i_0 = 2
r_0 = 10

start_date = START_DATE['Montreal']
learner = Learner(loss, start_date, predict_range, s_0, i_0, r_0)
learner.train()
