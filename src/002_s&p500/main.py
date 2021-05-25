import pandas as pd
import seaborn as sns
import os
from sktime.forecasting.base import ForecastingHorizon
from sklearn.neighbors import KNeighborsRegressor
from sktime.utils.plotting import plot_series
from sktime.datasets import load_airline
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
)
from sktime.forecasting.compose import (
    EnsembleForecaster,
    make_reduction,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation import evaluate


cwdir = os.getcwd()
data = cwdir + "/../../data/S&P500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv"
data = pd.read_csv(data)
data = data["open"]
#plot_series(data);

y_train, y_test = temporal_train_test_split(data, test_size=300)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
print(y_train.shape[0], y_test.shape[0])
plt.show()

fh = ForecastingHorizon(y_test.index, is_relative=False)
regressor = KNeighborsRegressor(n_neighbors=1)
forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
#mean_absolute_percentage_error(y_pred, y_test)
plt.show()
