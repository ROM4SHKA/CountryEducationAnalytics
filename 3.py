#місто село к середніх
import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import f_oneway
data = pd.read_csv('1.csv', sep=';')

print(data.head())
# зображення даних на графіку
plt.scatter(data['aUrban_1'], data['aRural_1'], c = 'b')
plt.title('Data Urban/Rural')
plt.xlabel('Urban')
plt.ylabel('Rural')
plt.show()

X = data.iloc[:, [0, 3, 4]].values
# метод ліктя
ar = []
for i in range(1,12):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
    kmeans.fit(X[:,[1,2]])
    ar.append(kmeans.inertia_)
plt.plot(range(1,12), ar)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('data')
plt.show()
#розбиваємо на кластери
kmeans = KMeans(n_clusters= 3, init= 'k-means++', random_state= 42)
y_kmeans = kmeans.fit_predict(X[:,[1,2]])
# виводимо отримані дані на графік
plt.scatter(X[y_kmeans == 0, 1], X[y_kmeans == 0,2], s = 100,c = 'b', label = 'The best')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1,2], s = 100,c = 'c', label = 'Worst')
plt.scatter(X[y_kmeans == 2, 1], X[y_kmeans == 2,2], s = 100,c = 'y', label =  'Average')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c = 'm', label ='centers')
plt.title('Clusters of countries k-means')
plt.xlabel('Urban')
plt.ylabel('Rural')
plt.legend()
plt.show()
# виводимо списки країн
print('The best:')
for i in X[y_kmeans == 0, 0]:
    print(i)
print(" ")
print('Average:')
for i in X[y_kmeans == 2, 0]:
    print(i)
print(" ")
print('Worst:')
for i in X[y_kmeans == 1, 0]:
    print(i)
# однофакорний дисперсний аналіз по групам країн
F, p = f_oneway(X[y_kmeans == 0, 1], X[y_kmeans == 0,2])
print("The Best countries:")
print(np.round(F,2))
print("p-фактор " + str(np.round(p,2)))
print("Average countries:")
F, p = f_oneway(X[y_kmeans == 2, 1], X[y_kmeans == 2,2])
print(np.round(F,2))
print("p-фактор " + str(np.round(p,2)))
print("Worst countries:")
F, p = f_oneway(X[y_kmeans == 1, 1], X[y_kmeans == 1,2])
print(np.round(F,2))
print("p-фактор " + str(np.round(p,2)))