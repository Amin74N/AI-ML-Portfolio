# Importing data:
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples= 500, centers= 4, random_state= 42, cluster_std= 1)
import pandas as pd
df = pd.DataFrame(X)
df["label"] = y
df.columns = ["X1","X2","label"]
print("Dataset:\n", df)
# ---------------------------------------------------------------------
# Showing raw data:
import matplotlib.pyplot as plt
import seaborn as sns
plt.title('Original Data')
sns.scatterplot(data= df, x= "X1", y= "X2", hue= "label", palette= "tab10")
plt.show()
X = df.drop("label", axis=1)
# ---------------------------------------------------------------------
# Feature scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
print("Mean of Scaled X:", scaled_X.mean())
print("Standard Deviation of Scaled X:", scaled_X.std())
# ---------------------------------------------------------------------
# Showing scaled data:
plt.title('Scaled Data')
sns.scatterplot(x= scaled_X[:,0], y= scaled_X[:,1])
plt.show()
# ---------------------------------------------------------------------
# K-Means clustering:
from sklearn.cluster import KMeans
clusters = range(1,10)
kmean = KMeans(n_clusters= 3, n_init= 10).fit(scaled_X)
print("Within-Cluster sum of squares:", kmean.inertia_)
# ---------------------------------------------------------------------
# Elbow method:
fig, ax = plt.subplots()
sum_squared_distances = [KMeans(k, n_init= 10).fit(scaled_X).inertia_ for k in clusters]
ax.plot(clusters, sum_squared_distances, marker= 'o')
ax.set(title= "Elbow plot to find K",xlabel= "Number of clusters", ylabel= "Within sum of squared distances to cluster centers")
plt.show()
# ---------------------------------------------------------------------
# Silhouette score:
from sklearn.metrics import silhouette_score
clusters = range(2,10)
kmeans = [KMeans(n_clusters=k, n_init=10).fit(scaled_X) for k in clusters]
silhouette_list = [silhouette_score(scaled_X, model.labels_) for model in kmeans]
fig, ax = plt.subplots(1)
ax.plot(clusters, silhouette_list, marker='o')
ax.set(title="Silhouette Plot", xlabel="K clusters", ylabel="Silhouette Score")
plt.show()
# ---------------------------------------------------------------------
# Plotting K-Means:
df_plot = pd.DataFrame(scaled_X, columns=["X1","X2"])
fig, axes = plt.subplots(2, 4, figsize= (16,8), constrained_layout= True)
for i, ax in enumerate(axes.flatten()):
    cluster_centers = kmeans[i].cluster_centers_
    df_plot["label"] = kmeans[i].labels_
    sns.scatterplot(data= df_plot, x= "X1", y= "X2", hue= "label", ax= ax, palette= "tab10")
    ax.scatter(cluster_centers[:,0], cluster_centers[:,1], s= 200, marker= "*", color= "black", label= "centroid")
    ax.legend([])
    ax.set(title=f"{clusters[i]} Clusters")
plt.show()
