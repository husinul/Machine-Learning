from sklearn.datasets import load_iris
iris = load_iris()

X= iris.data
y= iris.target


print X[0],y[0]


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X,y, test_size = 0.5)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(X_train,y_train)

predictions = clf.predict(X_test)


from sklearn.metrics import accuracy_score

print accuracy_score(y_test, predictions)

