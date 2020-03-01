import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


color = ['red','green','yellow','brown']
df = pd.read_excel("cars_data.xlsx",skiprows=1)
df = df.drop(0)
dff = df.drop(["Car Model"],axis=1)
pca = PCA(n_components = 3)
x = dff.values
x = scale(x);x #normalising the data
pca.fit(x)
var = np.array(pca.explained_variance_)
exp = np.sum(var)/11
print("The amountof variance(in %) explained by the 3 PCs is:")
print(exp*100)  
# This is the amount of explained variance in the 3 PCAs
print("The 3 PCAs are:")
print(pca.components_)
PC = pca.transform(x)
print("The PCA transform is:")
print(PC)
y = PC
ax = plt.axes(projection='3d')
ax.scatter3D(PC[:,0],PC[:,1],PC[:,2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title("PC projection")
plt.show()



ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(y)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

model = KMeans(n_clusters=4)
model.fit(y)	
label=model.labels_

i = 1
ax = plt.axes(projection='3d')
for a,b,c in PC:
    ax.scatter(a,b,color = color[label[i-1]] )
    # plt.text(a+0.1,b+0.1,df.at[i,'Car Model'])
    i = i+1
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title("Clustering in PC projection")
plt.show()

cluster1= []
cluster2 = []
cluster3 = []
cluster4 = []
# df.at[1,'Car Model']
for i in range(32):
    if(label[i]==0):
        cluster1.append(df.at[i+1,'Car Model'])
    elif(label[i]==1):
        cluster2.append(df.at[i+1,'Car Model'])
    elif(label[i]==2):
        cluster3.append(df.at[i+1,'Car Model'])
    else:
        cluster4.append(df.at[i+1,'Car Model'])

print('The Kmeans clusters in the PC projection are:')
print(cluster1)
print(cluster2)
print(cluster3)
print(cluster4)

i = 1
for a,b,c in PC:
    plt.scatter(a,b,color = color[label[i-1]] )
    plt.text(a+0.1,b+0.1,df.at[i,'Car Model'])
    i = i+1
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(x)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


model = KMeans(n_clusters=4)
model.fit(x)
label1 = model.labels_
cluster1= []
cluster2 = []
cluster3 = []
cluster4 = []
# df.at[1,'Car Model']
for i in range(32):
    if(label1[i]==0):
        cluster1.append(df.at[i+1,'Car Model'])
    elif(label1[i]==1):
        cluster2.append(df.at[i+1,'Car Model'])
    elif(label1[i]==2):
        cluster3.append(df.at[i+1,'Car Model'])
    else:
        cluster4.append(df.at[i+1,'Car Model'])
print('The Kmeans clusters on the data are:')
print(cluster1)
print(cluster2)
print(cluster3)
print(cluster4)



plt.title("Cars Dendograms")
dend = shc.dendrogram(shc.linkage(x, method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
label2=cluster.fit_predict(x)
cluster1= []
cluster2 = []
cluster3 = []
cluster4 = []
# df.at[1,'Car Model']
for i in range(32):
    if(label2[i]==0):
        cluster1.append(df.at[i+1,'Car Model'])
    elif(label2[i]==1):
        cluster2.append(df.at[i+1,'Car Model'])
    elif(label2[i]==2):
        cluster3.append(df.at[i+1,'Car Model'])
    else:
        cluster4.append(df.at[i+1,'Car Model'])

print('The agglomerative heirarchical clustering gives the following clusters:')
print(cluster1)
print(cluster2)
print(cluster3)
print(cluster4)