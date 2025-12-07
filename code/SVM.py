import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:,:2] #independent features 
y = iris.target
# print(iris.target_name)
# print(X)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.25)

#when our data is in higher dimension it is easy to classify 
#kernel helps us to convert lower dimension problem to higher dimension ... jb hmara data mushkil ho classi
#if the value of c is larger then there is less margin
C = 1.0

# svc support vector classifier ek aise variable hai jisme hmne model store krke rkha hai
svc = svm.SVC(kernel = 'linear' , C = 10).fit(X_train , y_train)
#svc = svm.SVC(kernel='rbf', C=10, gamma='auto').fit(X_train, y_train)
# svc = svm.SVC(kernel='poly', degree=3, C=10).fit(X_train, y_train)
#svc = svm.SVC(kernel='rbf', C=10, gamma='auto').fit(X_train, y_train)


y_pred = svc.predict(X_test)
print(svc.score(X_test,y_test))

#create a mesh to plot in
x_min = X[:,0].min()-1
x_max = X[:,0].max()+1

y_min= X[:,1].min() -1 
y_max=X[:,1].max() + 1
h = (x_max/x_min)/100 #this will show scale means ek point ko chodd ke next point kb aayega
h

# this numpy meshgrid function is used to create a rectangular grid out of two given one dimensional
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

plt.subplot(1,1,1)
Z = svc.predict(np.c_[xx.ravel(),yy.ravel()]) #ravel converts it into 1D and np.c_stacks them into column wise

Z = Z.reshape(xx.shape) #reshpe it into 2d so that we can colour 
plt.contourf(xx,yy,Z)
plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.autumn)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(),xx.max()) #Sets x-axis limits to match the meshgrid range.
plt.title('SVC with linear kernel')
plt.show()


# svm ek line bnata hai yaan curves jo ki classes ko classify krta hai
#us boundary ko plot krne ke liye hme 2d surface ke har ek jgah pr prediction check krna hoga
# mesh grid se hm 2d area ka hr ek point pr prediction check kr sakte hai
