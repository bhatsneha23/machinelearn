import seaborn as sns
import pandas as pd   
import matplotlib.pyplot as plt

print(sns.get_dataset_names())
df = sns.load_dataset("tips")   # restaurant bills dataset
# print(df.sample(7)) # will bring random colmns from dataset
print(df.head(7)) # will bring first 7 colmns from dataset

#CATEGORICAL DATA
sns.countplot(df['sex'])  #frequency count of each category
# df['sex'].value_counts().plot(kind = 'bar') #will give same answer as above line
# df['sex'].value_counts().plot(kind = 'pie' , autopct = '%.2f')

#NUMERICAL DATA
# plt.hist(df['total_bill'], bins = 30)  #bins = no of bars (HISTOGRAM)

# sns.distplot(df['total_bill'], bins = 30 , kde = True)  #KDE = KERNEL DENSITY ESTIMATION (smoothed version of histogram)
# it is showing data distribution so we can call this probability distribution curve , on y axis it is showing probability density function
#also shows skewnness of data
# if we take eg of cos curve , keeping salaries on y axis and people on x axis we can say most people have average salary and very few have very high 
# df['tip'].skew()  #skewness of data
# if it is perfectly skew it will be 0 , if it is positive skew it will be >0 (means right tail is longer) , if negative skew it will be <0 (means left tail is longer)


#(BOXPLOT) it gives 5 point summary - min(q1-1.5*iqr), Q1 (25 percentile log peeche hai), median, Q3 (75 percentile log peeche hai), max(q3+1.5*iqr)
#how much data is noisy , how many outliers are there
# sns.boxplot(df['total_bill'])

# df['tip'].min()
# df['tip'].max()
# df['tip'].mean()
# df['tip'].median()