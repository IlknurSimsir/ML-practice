from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X,_ = make_circles(n_samples=1000, factor=0.5,noise=0.05,random_state=42)
plt.figure()
plt.scatter(X[:,0], X[:,1])

dbscan= DBSCAN(eps=0.1,min_samples=5)#eps küme olarak alabilmesi için gerekn uzaklık 
cluster_labels=dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1],c=cluster_labels,cmap="viridis")
plt.title("DBSCAN results")
