from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt



cancer = load_breast_cancer()
df=pd.DataFrame(data =cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

X=cancer.data #features
y=cancer.target #target

#TRAIN-TEST Split
X_train,X_test,y_train,y_test=  train_test_split(X, y,test_size=0.3,random_state=42)

#STANDARTIZATION olceklendirme
scaler=StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

#TRAIN 
knn=KNeighborsClassifier(n_neighbors=3)#obje oluşturmayı unutma

knn.fit(X_train,y_train)# fit fonk bizim verimizi(sample+target) kullanarak knn algoritmasını eğitir

y_predict=knn.predict(X_test)

accuracy=accuracy_score(y_test,y_predict)
print(f"doğruluk: {accuracy}")

conf=confusion_matrix(y_test, y_predict)#kaç tane doğru tahmin ettiğini döndürür
print(f"karmaşıklık: {conf}")


#Hiperparametre ayarlması
"""
KNN :Hyperparameter = K
K:1,2,3,...N
Accuracy:%A,%B,%C
"""
accuracy_values=[]
k_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_predict=knn.predict(X_test)
    accuracy=accuracy_score(y_test, y_predict)
    accuracy_values.append(accuracy)
    k_values.append(k)


plt.figure()
plt.plot(k_values,accuracy_values,marker="o",linestyle="-")
plt.title("k değerine göre doğruluk")
plt.xlabel("k değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X=np.sort(5*np.random.rand(40, 1),axis=0) 
y=np.sin(X).ravel()

#plt.scatter(X, y)

y[::5]+=1*(0.5-np.random.rand(8))
#plt.scatter(X,y)
T=np.linspace(0,5,500)[:,np.newaxis]
for i,weight in enumerate(["uniform","distance"]):
    
    knn=KNeighborsRegressor(n_neighbors=5,weights=weight)
    y_predict=knn.fit(X, y).predict(T)
    plt.subplot(2,1,i+1)
    plt.scatter(X,y,color="green",label="data")
    plt.plot(T, y_predict,color="blue",label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor= {}".format(weight))
plt.tight_layout()
plt.show()











