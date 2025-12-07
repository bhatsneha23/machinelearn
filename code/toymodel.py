#preprocess + eda + feature selection
#extract input and output cols
#scale the values
#train test split
#train the model
#evaluate the model
#deploy the model


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pickle

# it is important to do scaling because some of our model will be biased to big numbers in dataset 
# so in order toprevent this we do scaling in which it brings all the value in one range  like 0 and 1
# scaling bhi different types ki hoti hai toh range bhi differ krega like min-max(0 se 1) , standardization(z-score scaling -3 to 3), maxAbs(-1 se 1)

df = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\NARUTO\\datasets\\placement.csv")
# print(df.head(5))
# df = df.iloc[:,1:]
# print(df.head(6))
df.head(5)
df = df.dropna(subset=['cgpa', 'iq','placement'])
# here placement is in yes or no form so fiest convert strings to numeric
# df['placement_num'] = df['placement'].map({'Yes':1,'No':0})

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
# plt.show()
df = df.iloc[:,1:]
X = df.iloc[:,0:2] #independent variables
y = df.iloc[:,-1]  #dependent variable
X_train , X_test , y_train , y_test =train_test_split(X,y,test_size = 0.1) #random data allot hota hai mtlb beech beec mei se
print(X_train)
# print(y)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train) #yeh numpy array bn chuka hai
# print(X_train)
X_test = scaler.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train , y_train) #model training

y_pred = clf.predict(X_test)
# print(y_test)

print(accuracy_score(y_test, y_pred))
plot_decision_regions(X_train,y_train.values,clf= clf , legend =2) #y_train ko numpy array mei convert kiya hai
# plt.show()

pickle.dump(clf,open('model.pkl','wb'))



# Import libraries — tools you’ll use.
# import pandas as pd, numpy as np, matplotlib.pyplot as plt, ...

# Load data — read the CSV into a DataFrame.
# df = pd.read_csv(".../placement.csv")

# Quick inspect — peek at rows & info so you know data shape & types.
# df.head() , df.info() , df.describe()

# Clean missing values — drop or fill NaNs before anything else.
# df = df.dropna(subset=['cgpa','iq','placement'])

# Encode labels (if needed) — convert target from text → numbers before training/plotting.
# df['placement'] = df['placement'].map({'Yes':1,'No':0})
# (Important: do this before train_test_split and before plotting that relies on numeric labels.)

# Visualize (optional quick check) — plot to understand separation (works best after encoding).
# plt.scatter(df['cgpa'], df['iq'], c=df['placement'])

# Select features & target — pick X (independent) and y (dependent).
# X = df[['cgpa','iq']]
# y = df['placement']

# Split into train & test — always split before scaling. Use random_state for reproducibility.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale features — fit scaler on train only, transform both train & test.
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Train the model — fit your classifier on scaled training data.
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# Predict on test set — get model outputs.
# y_pred = clf.predict(X_test)

# Evaluate — check accuracy (and preferably other metrics).
# print(accuracy_score(y_test, y_pred))
# (Tip: also use confusion_matrix and classification_report for more insight.)

# Visualize decision boundary (optional) — only works for 2 features and numpy arrays.
# plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)

# Save the model (and scaler) — persist what you need for later predictions.
# pickle.dump(clf, open('model.pkl','wb'))
# pickle.dump(scaler, open('scaler.pkl','wb')) (so you can scale new data the same way)

# (Optional) Load & predict later — load both scaler + model and follow scale → predict.
# scaler = pickle.load(open('scaler.pkl','rb'))
# clf = pickle.load(open('model.pkl','rb'))
# clf.predict(scaler.transform(new_X))