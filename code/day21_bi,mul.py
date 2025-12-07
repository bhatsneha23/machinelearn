import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
titanic =  sns.load_dataset('titanic')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')

# print(tips.head())
# print(titanic.head())
# print(iris.head())
print(flights.head())

# #SCATTER PLOT
# sns.scatterplot(x = 'total_bill' , y = 'tip' , data = tips , hue = tips['sex'] , style = tips['smoker'], size = tips['size'])  #scatter plot for bi variate analysis , by adding sex it became multivariate

# hue → adds color grouping (different colors for categories or values).
# style → adds marker style grouping (different shapes for categories).
# size → adds size grouping (how much amount of people came to restraunt )

#BAR PLOT
#one will be numerical and other will be categorical
#sns.barplot(x = 'pclass' , y = 'fare' , data = titanic  , hue = 'sex')

#BOX PLOT
# sns.boxplot(x ='sex' , y ='age' ,data = titanic , hue = 'survived')  #it will show fare distribution

# sns.distplot(titanic[titanic['survived']==0]['age'],hist= False , color = 'red')
# sns.distplot(titanic[titanic['survived']==1]['age'] , hist= False ,color = 'green')  #it will show age distribution of survived and not survived passengers

# CATEGORICAL VS CATEGORICAL
# HEATMAP (used for relationship between 2 categorical variables)
# sns.heatmap(pd.crosstab(titanic['pclass'],titanic['survived']))

# titanic.groupby('embarked')['survived'].mean() * 100

# CLUSTERMAP 
# sns.clustermap(pd.crosstab(titanic['parch'],titanic['survived']))  #it will cluster similar values together

# PAIRPLOT(it will give pairwise relationship between all numerical columns)
# har col ka scatter plot bnaega dusre col se sath

# sns.pairplot(iris , hue= 'species')

# LINEPLOT :- if we join all the points of scatter plot line plot is formed
new = flights.groupby('year').sum(numeric_only = True).reset_index()
sns.lineplot(x = 'year' , y = 'passenger' , data = new , marker = 'o')
plt.title()