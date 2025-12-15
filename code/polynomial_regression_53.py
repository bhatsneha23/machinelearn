import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression , SGDRegressor
from sklearn.preprocessing import PolynomialFeatures , StandardScaler 
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline  

X = 6 * np.random.rand(200,1) - 3 
# np.random.rand(200,1) → 200 random numbers between 0 and 1
# 6 * ... → unhe 0 se 6 range mein convert karta hai
# -3 → phir range ko -3 se +3 shift kar deta hai
# So final X values lie between -3 and +3
# Ye tum input feature bana rahi ho.
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.rand(200,1)
# 0.8 * X**2 → quadratic term
# + X → linear term
# + 2 → constant
# + np.random.rand(200,1) → thoda random noise add karta hai
# Ye ek quadratic curve ke around thoda noisy data generate karta hai.
# So final y ek curve ko follow karta hai:

# plt.plot(X,y,'b.')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=2)
lr = LinearRegression()
lr.fit(X_train , y_train)
LinearRegression()

y_pred = lr.predict(X_test)
print("old" ,r2_score(y_test , y_pred))

plt.plot(X_test , lr.predict(X_test) , color = 'red')
plt.plot(X,y,'b.')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

poly = PolynomialFeatures(degree = 2 , include_bias = True) # degree mei 1 input se 3 column bnenge
# include_bias intercept term include nhi krega 
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

print(X_train[0])
print(X_train_trans[0])

lrp = LinearRegression()
lrp.fit(X_train_trans ,y_train)
y_pred_poly = lrp.predict(X_test_trans)
# print("NEW" ,r2_score(y_test , y_pred_poly))

print(lrp.coef_)
print(lrp.intercept_)