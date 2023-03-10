from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_dataset = []
y_dataset = []

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.3, random_state=None)

RFR = RandomForestRegressor(n_estimators=200, random_state=0)
RFR.fit(x_train, y_train)
y_pred = RFR.predict(x_train)
