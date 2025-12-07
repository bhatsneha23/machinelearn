import seaborn as sns  
import pandas as pd   
import matplotlib.pyplot as plt  

titanic = sns.load_dataset('titanic')
print(titanic.head())

df = titanic[['age','fare']]
print(df.head())

df.columns