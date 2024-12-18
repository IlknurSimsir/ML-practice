from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


iris=load_iris()

X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=42)


#KNN
knn=KNeighborsClassifier()
knn_param_grid={"n_neighbors": np.arange(2, 31)}

knn_grid_search =GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train,y_train)
print("KNN grid search best params: ",knn_grid_search.best_params_)
print("KNN grid search best accuracy: ",knn_grid_search.best_score_)

knn_random_search =RandomizedSearchCV(knn, knn_param_grid,n_iter=10)
knn_random_search.fit(X_train,y_train)
print("KNN random search best params: ",knn_random_search.best_params_)
print("KNN random search best accuracy: ",knn_random_search.best_score_)

#DT
tree=DecisionTreeClassifier()
tree_param_grid={"max_depth":[3,5,7],"max_leaf_nodes":[None,5,10,20,30,50]}

tree_grid_search =GridSearchCV(tree, tree_param_grid)#include random cross validation
tree_grid_search.fit(X_train,y_train)
print("Tree grid search best params: ",tree_grid_search.best_params_)
print("Tree grid search best accuracy: ",tree_grid_search.best_score_)

tree_random_search =RandomizedSearchCV(tree, tree_param_grid,n_iter=10)
tree_random_search.fit(X_train,y_train)
print("Tree random search best params: ",tree_random_search.best_params_)
print("Tree random search best accuracy: ",tree_random_search.best_score_)

#SVM
svm=SVC()
svm_param_grid={"C":[0.1,10,100],"gamma":[0.1,0.01,0.001,0.0001]}

svm_grid_search =GridSearchCV(svm, svm_param_grid)#include random cross validation
svm_grid_search.fit(X_train,y_train)
print("SVM grid search best params: ",svm_grid_search.best_params_)
print("SVM grid search best accuracy: ",svm_grid_search.best_score_)

svm_random_search =RandomizedSearchCV(svm, svm_param_grid,n_iter=10)
svm_random_search.fit(X_train,y_train)
print("SVM random search best params: ",svm_random_search.best_params_)
print("SVM random search best accuracy: ",svm_random_search.best_score_)









