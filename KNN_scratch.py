import random
from scipy.spatial import distance 

def euc(a,b):
	return distance.euclidean(a,b)


class KNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for i in X_test:
			#label = random.choice(self.y_train)
			label = self.closest(i)
			predictions.append(label)
		return predictions
	
	def closest(self,i):
		best_dist = euc(i,self.X_train[0])
		best_index =0
		for j in range(1, len(self.X_train)):
			dist = euc(i,self.X_train[j])
			if dist < best_dist:
				best_dist = dist
				best_index = j
		return self.y_train[best_index]




from sklearn.datasets import load_iris
iris = load_iris()

X= iris.data
y= iris.target


print X[0],y[0]


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X,y, test_size = 0.5)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
clf = KNN()
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)


from sklearn.metrics import accuracy_score

print accuracy_score(y_test, predictions)

