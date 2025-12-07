# import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes 
# from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error

X,y = load_diabetes(return_X_y = True)
X_train, X_test , y_train , y_test  = train_test_split(X , y , test_size = 0.2 , random_state = 42)
# model = LinearRegression()
# model.fit(X_train , y_train)

# y_predict = model.predict(X_test)

# print("rr_Score: " ,r2_score(y_test , y_predict))

# print(model.coef_)
# print(model.intercept_)
# print(load_diabetes().feature_names)

class MeraLR:
    def __init__(self):
        self.coef_=None  #b1 , b2 , b3
        self.intercept = None  #b0

    def fit(self,X_train,y_train):
        X_train = np.insert(X_train , 0 , 1 ,axis = 1)
    #kaun sa array , kaun se index , kaun se operatins , kaun si axis pr 
    # calculate coefficients
        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        print(betas)
        self.intercept = betas[0]
        self.coef_ = betas[1:]

    def predict(self,X_test):
        y_predict = np.dot(X_test , self.coef_) + self.intercept_
        return y_predict

Lr = MeraLR()
Lr.fit(X_train,y_train)
print(X_train)
print(np.insert(X_train , 0 , 1 ,axis = 1))
Lr.predict(X_test)