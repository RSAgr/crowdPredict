# Apply KMeans clustering with K=4 to identify peak visit times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

df = pd.read_csv("responses.csv")
df["Visit_Hour"] = pd.to_datetime(df["What time visit the cafeteria most often?"], errors='coerce').dt.hour
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Visit_Hour"]].dropna())

# Add cluster centers for reference
cluster_centers = kmeans.cluster_centers_.flatten()

# Visualize clusters of visit times
plt.figure(figsize=(10, 6))
sns.histplot(data=df_clean, x="Visit_Hour", hue="Cluster", palette="viridis", bins=24, multiple="stack")
plt.title("Cafeteria Visit Times Clustered (K=4)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Visits")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show(), cluster_centers


