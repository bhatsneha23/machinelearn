# RUN THIS CODE IN GOOGLE COLAB 
from sklearn.datasets import make_regression 
import numpy as np 
import matplotlib.pyplot as plt 


X , y = make_regression(n_samples = 4 , n_feature = 1 , n_informative = 1 , n_targets = 1 , noise = 80 , random_state = 13)
plt.scatter(X , y)