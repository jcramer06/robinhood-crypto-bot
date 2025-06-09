from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from trading import cost_over_time
import matplotlib.pyplot as plt
import time
import numpy as np
from trading import xlist, ylist

def PolyRegression(x, y):
    x = np.array(x).reshape(-1, 1)  # Ensure 2D input
    y = np.array(y)

    features = PolynomialFeatures(degree=3)
    x_poly = features.fit_transform(x)

    poly_regression_model = LinearRegression()
    poly_regression_model.fit(x_poly, y)
    plt.scatter(x, y, color="red")
    plt.plot(x, poly_regression_model.predict(x_poly), color="blue")
    plt.show()
    return poly_regression_model

cost_over_time()


while True:
    time.sleep(1)
    PolyRegression(xlist, ylist)