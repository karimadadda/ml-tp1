#!/usr/bin/env python
# coding: utf-8
L’algorithme DBSCAN est un algorithme de clustering basé sur la densité. Il peut diviser des régions avec une densité suffisamment élevée en clusters et trouver des clusters de formes arbitraires dans une base de données spatiale bruyante. L’algorithme trouve les points anormaux en s’assurant qu’un seul point anormal ne génère pas de cluster. Il existe deux paramètres pour contrôler la génération du cluster : MinPts est le nombre minimum de nœuds dans le cluster. e est le rayon de l’amas [8]. Pour chaque point du cluster, il doit y avoir un autre point dans le cluster, et la distance entre eux est inférieure à un certain seuil.
# # Importing the required libraries

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# # Loading the data

# In[49]:


df =  pd.read_csv('corona.csv')
df


# In[9]:


df.isnull().sum()


# # Preprocessing the data
# 1. Normalisation
Étant donné que les caractéristiques de notre ensemble de données ne sont pas à la même échelle, nous devons normaliser l’ensemble de l’ensemble de données. En d’autres termes, chaque caractéristique de notre ensemble de données a des magnitudes et une portée uniques pour ses données. Une augmentation d’un point de Deaths / 100 Cases n’équivaut pas à une augmentation d’un point de Recovered / 100 Cases et vice versa. Étant donné que DBSCAN utilise la distance (euclidienne) entre les points pour déterminer la similitude, les données non mises à l’échelle créent un problème. Si une caractéristique a une plus grande variabilité dans ses données, le calcul de distance sera plus affecté par cette caractéristique. En mettant à l’échelle nos fonctionnalités, nous alignons toutes les fonctionnalités sur une moyenne de zéro et un écart-type de un.
# In[51]:


scaler = StandardScaler()
scaler.fit(df)
X_scale = scaler.transform(df)
df_scale = pd.DataFrame(X_scale, columns=df.columns)
df_scale.head()


# ## 2. Réduction des fonctionnalités
Certains algorithmes tels que KMeans ont du mal à construire des clusters avec précision si l’ensemble de données a trop de caractéristiques . 
La théorie derrière la réduction des caractéristiques ou de la dimensionnalité consiste à convertir l’ensemble de caractéristiques d’origine en moins de caractéristiques dérivées artificiellement qui conservent encore la plupart des informations englobées dans les caractéristiques d’origine.
L’une des techniques de réduction des caractéristiques les plus répandues est l’analyse en composantes principales ou APC. PCA réduit le jeu de données d’origine à un nombre spécifié de fonctionnalités que PCA appelle les composants principaux. Nous devons sélectionner le nombre de composantes principales que nous souhaitons voir. 
Tout d’abord, nous devons déterminer le nombre approprié de composants principaux. Il semblerait que 3 composantes principales représentent environ 90 % de l’écart.
# In[52]:


pca = PCA(n_components=7)
pca.fit(df_scale)
variance = pca.explained_variance_ratio_ 
var=np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12,6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(0,100.5)
plt.plot(var)

Maintenant que nous connaissons le nombre de composants principaux nécessaires pour maintenir un pourcentage spécifique de variance, appliquons un APC à 3 composants à notre ensemble de données d’origine. Notez que la première composante principale représente 65 % de l’écart par rapport à l’ensemble de données d’origine.
# In[32]:


pca = PCA(n_components=3)
pca.fit(df_scale)
pca_scale = pca.transform(df_scale)
pca_df = pd.DataFrame(pca_scale, columns=['pc1', 'pc2', 'pc3'])
print(pca.explained_variance_ratio_)


# In[53]:


pca_df

En traçant nos données dans un espace 3D, nous pouvons voir certains problèmes potentiels pour DBSCAN. Si vous vous souvenez, l’un des principaux inconvénients de DBSCAN est son incapacité à regrouper avec précision des données de densité variable et à partir du graphique ci-dessous, nous pouvons voir deux clusters distincts de densité très différente. En appliquant l’algorithme DBSCAN, nous pourrions être en mesure de trouver des clusters dans le cluster inférieur de points de données, mais de nombreux points de données dans le cluster supérieur peuvent être classés comme valeurs aberrantes / bruit. Tout cela dépend bien sûr de notre sélection d’epsilon et de valeurs minimales de points.
# In[65]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(colorscale='Greys', opacity=0.3, size = 10, ))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# # 3. Clustering DBSCAN
Première approche 
Avant d’appliquer l’algorithme de clustering, nous devons déterminer le niveau d’epsilon approprié à l’aide de la « méthode Elbow » dont nous avons discuté ci-dessus. Il semblerait que la valeur epsilon optimale soit d’environ 0,2. Enfin, comme nous avons 3 composantes principales à nos données, nous allons fixer nos critères de points minimum à 6.
# In[66]:


db = DBSCAN(eps=0.2, min_samples=6).fit(pca_df)
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))

En réglant l’epsilon à 0,2 et min_samples à 6, on a obtenu 4 clusters, un score Silhouette de -0,238 (indique que les points de données sont incorrectement regroupés.)
## Le score Silhouette est la distance entre un échantillon et le cluster le plus proche dont l’échantillon ne fait pas partie.
La meilleure valeur est 1 et la pire valeur est -1. Les valeurs proches de 0 indiquent des clusters qui se chevauchent. Les valeurs négatives indiquent généralement qu’un échantillon a été affecté au mauvais cluster, car un cluster différent est plus similaire.##
Il y'as 139 points de données considérés comme des valeurs aberrantes / bruit. les 4 clusters obtenus pourraient être considérées comme informatives, mais nous avons un ensemble de données de 15 000 employés. 
En regardant le graphique 3D ci-dessous, nous pouvons voir un cluster englobant la majorité des points de données. Il y a un groupe plus petit mais significatif qui a émergé,  ces clusters ne sont pas très informatifs car la plupart des employés appartiennent à seulement un cluster. 
# In[67]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = db.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (4) Derived from PCA', font=dict(size=12,))
fig.show()

Deuxième approche
Au lieu d’utiliser la « méthode Elbow » et l’heuristique de la valeur minimale, adoptons une approche itérative pour affiner notre modèle DBSCAN. Nous allons itérer à travers une plage de valeurs epsilon et de points minimums lorsque nous appliquons l’algorithme DBSCAN à nos données.
Dans notre exemple, nous allons à l’itération à travers des valeurs epsilon allant de 0,5 à 1,5 à des intervalles de 0,1 et des valeurs ponctuelles minimales allant de 2 à 7. La boucle for exécutera l’algorithme DBSCAN à l’aide de l’ensemble de valeurs et produira le nombre de clusters et le score de silhouette pour chaque itération. Gardez à l’esprit que vous devrez ajuster vos paramètres en fonction de vos données.
# In[75]:


pca_eps_values = np.arange(0.1,3.5,0.1) 
pca_min_samples = np.arange(2,7) 
pca_dbscan_params = list((pca_eps_values, pca_min_samples))
pca_no_of_clusters = []
pca_sil_score = []
pca_epsvalues = []
pca_min_samp = []
for p in pca_dbscan_params:
    pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
    pca_epsvalues.append(p[0])
    pca_min_samp.append(p[1])
    pca_no_of_clusters.append(
len(np.unique(pca_dbscan_cluster.labels_)))
    pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))
pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
pca_eps_min_df


# In[76]:


db = DBSCAN(eps=0.1, min_samples=0.2).fit(pca_df)
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# In[77]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = db.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (157) Derived from PCA', font=dict(size=12,))
fig.show()


# In[78]:


db = DBSCAN(eps=2, min_samples=3).fit(pca_df)
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# In[79]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = db.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (2) Derived from PCA', font=dict(size=12,))
fig.show()


# In[48]:


np.unique(labels, return_counts=True)


# In[ ]:




