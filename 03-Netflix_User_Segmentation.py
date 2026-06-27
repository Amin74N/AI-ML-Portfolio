# Importing data:
import pandas as pd
raw_df = pd.read_csv('netflix_users.csv')
df = raw_df[['Age', 'Watch_Time_Hours', 'Rating']]
print("Dataset Information:\n", df.describe())
# ---------------------------------------------------------------------
# Feature scaling:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
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
plt.savefig('03-Netflix User Segmentation Elbow plot.png', dpi = 300)
plt.show()
# ---------------------------------------------------------------------
# K-Means clustering:
from sklearn.cluster import KMeans
model = KMeans(n_clusters= 2, random_state= 42, n_init= 10)
model.fit(scaled_data)
# ---------------------------------------------------------------------
# Showing Clusters info:
df["cluster"] = model.labels_
print("Clusters Information:\n",df.groupby("cluster").mean())
# ---------------------------------------------------------------------
# ScatterPlot Visualization:
import seaborn as sns
sns.scatterplot(data= df, x= "Watch_Time_Hours", y= "Rating", hue= "cluster", palette= "tab10")
plt.legend([])
plt.savefig('03-Netflix User Segmentation Scatter plot.png', dpi = 300)
plt.show()
