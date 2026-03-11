import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

class LogitRegression(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        p[p ==1] = 0.999
        p[p == 0] = 0.001
        y = np.log(p / (1.0 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

class LogitXGBRegression(XGBRegressor):

    def fit(self, x, p):
        p = np.asarray(p)
        p[p ==1] = 0.999
        p[p == 0] = 0.001
        y = np.log(p / (1.0 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

class LogitRFRegression(RandomForestRegressor):

    def fit(self, x, p):
        p = np.asarray(p)
        p[p ==1] = 0.999
        p[p == 0] = 0.001
        y = np.log(p / (1.0 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)