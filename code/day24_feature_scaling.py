import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\NARUTO\\datasets\\Social_Network_Ads.csv")
df = df.iloc[:,2:]
df.sample(5)

X_train , X_test , y_train , y_test = train_test_split(df.drop('Purchased', axis = 1), df['Purchased'] , test_size = 0.3 , random_state = 0)

