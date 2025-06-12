from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from trading import cost_over_time
import matplotlib.pyplot as plt
import time
import numpy as np
from trading import xlist, ylist
from sklearn.tree import DecisionTreeRegressor

def PolyRegression(x, y):
    x = np.array(x).reshape(-1, 1)  # Ensure 2D input
    y = np.array(y)

    features = PolynomialFeatures(degree=3)
    x_poly = features.fit_transform(x)

    poly_regression_model = LinearRegression()
    poly_regression_model.fit(x_poly, y)
    #plt.scatter(x, y, color="red")
    #plt.plot(x, poly_regression_model.predict(x_poly), color="blue")
    #plt.show()
    x_new = np.array(len(xlist)+30).reshape(-1, 1)
    x_new_poly = features.transform(x_new)
    return poly_regression_model.predict(x_new_poly)

cost_over_time()


#while True:
#    time.sleep(1)
#    result = PolyRegression(xlist, ylist)
#    while len(xlist) > 500:
#       xlist.pop(0)
#    while len(ylist) > 500:
#      ylist.pop(0)
#        print(len(xlist))s
#    print(result)