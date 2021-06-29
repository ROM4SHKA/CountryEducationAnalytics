#міста села ієрархічна
import  pandas as pd
import  matplotlib.pyplot as plt
import  scipy.cluster.hierarchy as sch
from  sklearn.cluster import AgglomerativeClustering
data = pd.read_csv('1.csv', sep=';')
print(data.head())

X = data.iloc[:, [0, 3, 4]].values
#дендограма
dendogram = sch.dendrogram(sch.linkage(X[:,[1,2]], method='ward'))
plt.title('Dendogram')
plt.xlabel('Clusters')
plt.ylabel('Euclid dist')
plt.show()
# розбиттся на кластери
hierc = AgglomerativeClustering(n_clusters= 3, affinity= 'euclidean', linkage='ward')
y_hierc = hierc.fit_predict(X[:,[1,2]])
# виводимо отримані дані на графік
plt.scatter(X[y_hierc == 0, 1], X[y_hierc == 0,2], s = 100,c = 'y', label = 'Average')
plt.scatter(X[y_hierc == 1, 1], X[y_hierc == 1,2], s = 100,c = 'c', label = 'Worst')
plt.scatter(X[y_hierc == 2, 1], X[y_hierc == 2,2], s = 100,c = 'b', label =  'The Best')
plt.title('Clusters of countries Hierarhial')
plt.xlabel('Urban')
plt.ylabel('Rural')
plt.legend()
plt.show()
# виводимо списки країн
print('The best:')
#for i in X[y_hierc == 1, 0]:
#    print(i)
print(X[y_hierc==2, 0])
print(" ")
print('Average:')
#for i in X[y_hierc == 0, 0]:
#    print(i)
print(X[y_hierc == 0, 0])
print(" ")
print('Worst:')
#for i in X[y_hierc == 2, 0]:
#    print(i)
print(X[y_hierc == 1, 0])