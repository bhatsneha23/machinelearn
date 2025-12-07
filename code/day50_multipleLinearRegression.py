# run this code on google colab 
import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

import plotly.express as px  
import plotly.graph_objects as go 

from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

X , y = make_regression(n_samples = 100 , n_features = 2 , n_informative = 2 , n_targets = 1 , noise = 50)
df = pd.DataFrame({'feature1' : X[:,0] , 'feature2' : X[:,1] , 'target' : y})
print(df.head())

fig = px.scatter_3d(df , x = 'feature1' , y = 'feature2' , z = 'target')
fig.show()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)
model = LinearRegression()

model.fit(X_train , y_train)
y_predict = model.predict(X_test)
print("mse: " , mean_squared_error(y_test,y_predict))
print("mae: ", mean_absolute_error(y_test,y_predict))
print("r2 score: ",r2_score(y_test , y_predict))


class MeraLR:
    def __init__(self):
        self.coef_=None 
        self.intercept = None 

    def fit(self,X_train,y_train):
        np.insert(X_train , 0 , 1 ,axis = 1)
    #kaun sa array , kaun se index , kaun se operatins , kaun si axis pr 

    def predict(self,X_test):
        pass 

Lr = MeraLR()
Lr.fit(X_train,y_train)
print(X_train)
print(np.insert(X_train , 0 , 1 ,axis = 1))
