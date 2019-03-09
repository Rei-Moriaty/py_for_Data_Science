from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

def dtc():
	clf = tree.DecisionTreeClassifier()

	model = clf.fit(X,Y)
	predict_accuracy(model, 'Decision Tree')
	
def svc():
	clf = svm.SVC(gamma='scale')
	model = clf.fit(X,Y)
	predict_accuracy(model, 'Support Vector Machine')
	
def ncc():
	clf = NearestCentroid()
	model = clf.fit(X,Y)
	predict_accuracy(model, 'Nearest Centroid Classifier')

def sgdc():
	clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200,tol=0.19)
	model = clf.fit(X,Y)
	predict_accuracy(model, 'Stochastic Gradient Descent Classifier')

def predict_accuracy(model,s):
	prediction=model.predict(X)
	accuracy=accuracy_score(Y, prediction)
	accuracy=accuracy*100
	
	print("Accuracy for "+s+" : " + str(accuracy) + "%")

dtc()
svc()
ncc()
sgdc()