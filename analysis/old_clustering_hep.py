import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import tslearn.clustering
import tslearn.neighbors
import tslearn.preprocessing

df = pd.read_csv(
    "https://raw.githubusercontent.com/RealityBending/PrimalsInteroception/main/data/data_hep.csv"
)


# Format data ========================================
data = {}
for c in np.unique(df["Condition"]):
    for ch in ["AF7", "AF8"]:
        data[c + "_" + ch] = df[(df["Condition"] == c)].pivot(
            index="time", columns="participant_id", values=ch
        )
        data[c + "_" + ch] = nk.standardize(data[c + "_" + ch])

X1 = tslearn.utils.to_time_series_dataset(data["RestingState_AF7"].T)
X2 = tslearn.utils.to_time_series_dataset(data["RestingState_AF8"].T)
# X1 = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(X1)


# How many clusters? ========================================
def silhouette(X, n_clusters=2):
    m = tslearn.clustering.KShape(n_clusters=n_clusters).fit(X)
    return tslearn.clustering.silhouette_score(X, m.labels_, metric="euclidean")


# High silhouette score = good clustering
plt.plot(range(2, 8), [silhouette(X1, n_clusters=i) for i in range(2, 8)], label="AF7")
plt.plot(range(2, 8), [silhouette(X2, n_clusters=i) for i in range(2, 8)], label="AF8")
plt.legend()

# Clustering ================================================
# Manual clusters (based on observation)
# data["RestingState_AF7"].plot(subplots=True)
# manual_clusters = [1, 5, 10, 13, 15, 17, 18, 20]
# numbers = [int(s.split("-")[1]) for s in data["RestingState_AF7"].columns]
# manual_clusters = np.array([0 if n in manual_clusters else 1 for n in numbers])

# Clustering using various metrics
models = {}
for metric in ["euclidean", "dtw", "softdtw", "kshape"]:
    print(metric)
    if metric == "kshape":
        ks = tslearn.clustering.KShape(n_clusters=2)
        models[metric] = ks.fit(X2)
    else:
        km = tslearn.clustering.TimeSeriesKMeans(n_clusters=2, metric=metric)
        models[metric] = km.fit(X2)


# Visualize
def plot_clusters(data, clusters, ax=None):
    if isinstance(clusters, np.ndarray):
        model = None
    else:
        model = clusters
        clusters = model.labels_
    colors = {0: "red", 1: "black"}
    for ppt, cluster in enumerate(clusters):
        ax.plot(
            data.index,
            data.values[:, ppt],
            color=colors[cluster],
            alpha=1 / 10,
            linewidth=0.5,
        )
    for c in range(2):
        ax.plot(
            data.index,
            data.iloc[:, clusters == c].mean(axis=1),
            color=colors[c],
            label=c,
            linewidth=1,
            linestyle="dashed",
        )
        if model is not None:
            ax.plot(
                data.index,
                model.cluster_centers_[c].ravel(),
                color=colors[c],
                linewidth=1.5,
            )


fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(10, 12))
for i, ch in enumerate(["AF7", "AF8"]):
    # plot_clusters(data["RestingState_" + ch], manual_clusters, ax=ax[0, i])
    # ax[0, i].set_title(f"Manual clustering ({ch})")
    for j, metric in enumerate(["euclidean", "dtw", "softdtw", "kshape"]):
        plot_clusters(data["RestingState_" + ch], models[metric], ax=ax[j, i])
        ax[j, i].set_title(f"{metric} clustering ({ch})")
plt.legend()


# How many participants in each cluster?
np.unique(models["euclidean"].labels_, return_counts=True)
np.unique(models["kshape"].labels_, return_counts=True)

# Save clusters
clusters = models["euclidean"].labels_
df["Cluster"] = [
    "N100" if s in data["RestingState_AF7"].columns[clusters == 0] else "P200"
    for s in df["participant_id"]
]
df.to_csv("../data/data_hep.csv", index=False)
