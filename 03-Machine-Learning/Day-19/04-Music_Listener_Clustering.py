# Importing data:
import pandas as pd
raw_df = pd.read_excel('spotify_data.xlsx')
df = raw_df[['listening_hours', 'favorite_genre']]
df["listening_hours_group"] = pd.cut(df["listening_hours"], bins=[0, 60, 120, 180], labels=["Low", "Medium", "High"])
num_df = df.copy()
# ---------------------------------------------------------------------
# Feature encoding:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
num_df['listening_hours_group'] = encoder.fit_transform(df['listening_hours_group'])
num_df['favorite_genre'] = encoder.fit_transform(df['favorite_genre'])
print("Dataset Information:\n", num_df.describe())
# ---------------------------------------------------------------------
# Feature scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)
# ---------------------------------------------------------------------
# Elbow method:
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
scores = []
for k in range(1, 10):
    model = KMeans(n_clusters= k, random_state= 42, n_init= 10)
    model.fit(scaled_data)
    scores.append(model.inertia_)
plt.title('Elbow plot to find K')
plt.xlabel('Number of clusters')
plt.ylabel('Within sum of squared distances to cluster centers')
plt.plot(range(1,10),scores)
plt.savefig('04-Music Listener Clustering Elbow plot.png', dpi = 300)
plt.show()
# ---------------------------------------------------------------------
# K-Means clustering:
from sklearn.cluster import KMeans
model = KMeans(n_clusters= 3, random_state= 42, n_init= 10)
model.fit(scaled_data)
# ---------------------------------------------------------------------
# Showing Clusters info:
num_df["cluster"] = model.labels_
print("Clusters Information:\n",num_df.groupby("cluster").mean())
# ---------------------------------------------------------------------
# ScatterPlot Visualization:
import seaborn as sns
plt.figure(figsize=(18,6))
df["cluster"] = model.labels_
sns.boxplot(data= df, x= "favorite_genre", y= "listening_hours", hue= "cluster", palette= "tab10")
plt.legend([])
plt.savefig('04-Music Listener Clustering Box plot.png', dpi = 300)
plt.show()
