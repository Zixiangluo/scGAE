import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score


def myscatter(Y, class_idxs, legend=True, ran=True, seed=229):
    if ran:
        np.random.seed(seed)
    Y = np.array(Y)
    fig, ax = plt.subplots(figsize=(9, 6))
    classes = list(np.unique(class_idxs))
    markers = 'osD' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    if ran:
        np.random.shuffle(colors)

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(Y[class_idxs == cls, 0], Y[class_idxs == cls, 1], marker=mark,
                linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                markeredgecolor='black', markeredgewidth=0.2)
    if legend:
        ax.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0)

    return ax


def doumap(X, mind=0.5):
    reducer = umap.UMAP(min_dist=mind)
    Y = reducer.fit_transform(X)
    return Y


def dotsne(X, dim=2, ran=23):
    tsne = TSNE(n_components=dim, random_state=ran)
    Y_tsne = tsne.fit_transform(X)
    return Y_tsne


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def plot(Y, idents, dim=2, f="tsne", pca=30, legend=True, ran=False):
    if Y.shape[1] == 2:
        myscatter(Y, idents, legend=legend, ran=ran)
    else:
        if pca:
            Y = dopca(Y, dim=pca)
        if (f == "umap"):
            Y2 = doumap(Y)
            plot(Y2, idents, dim, f, legend=legend, ran=ran)
        elif (f == "tsne"):
            Y2 = dotsne(Y, dim)
            plot(Y2, idents, dim, f, legend=legend, ran=ran)


def measure(true, pred):
    NMI = round(normalized_mutual_info_score(true, pred), 2)
    RAND = round(adjusted_rand_score(true, pred), 2)
    HOMO = round(homogeneity_score(true, pred), 2)
    COMP = round(completeness_score(true, pred), 2)
    return [NMI, RAND, HOMO, COMP]


def clustering(h, n_cluster, k=150, f="louvain"):
    from preprocessing import get_adj
    from clustering import louvain
    adj, adj_n = get_adj(h, k=k, pca=False)
    if f == "louvain":
        cl_model = louvain(level=0.5)
        cl_model.update(h, adj_mat=adj)
        labels = cl_model.labels
    elif f == "spektral":
        labels = SpectralClustering(n_clusters=n_cluster, affinity="precomputed", assign_labels="discretize",
                                    random_state=0).fit_predict(adj)
    elif f == "kmeans":
        labels = KMeans(n_clusters=n_cluster, random_state=0).fit(h).labels_
    return labels


def dpt(data, h, k=15):
    import scanpy as sc
    import scipy.stats as stats
    times = np.array(
        pd.read_csv("/Users/zixiangluo/Desktop/DR/Data/data" + str(data) + "/time" + str(data) + ".csv", index_col=0,
                    sep="\t"))
    adata = sc.AnnData(X=h)
    adata.obs['times'] = times
    adata.uns['iroot'] = np.flatnonzero(adata.obs['times'] == 0)[0]
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=k, use_rep="X")
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    ds = adata.obs["dpt_pseudotime"]
    tau, p_value = stats.kendalltau(times, ds)
    return tau, ds


def nn(count, h, k=35, metric="percentage"):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean").fit(count)
    dis, idx = nbrs.kneighbors(count)
    nbrsh = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric="euclidean").fit(h)
    dish, idxh = nbrsh.kneighbors(h)
    it = 0
    if metric == "percentage":
        for i in range(count.shape[0]):
            it += np.intersect1d(idx[i], idxh[i]).shape[0]
        it = it / count.shape[0]
    elif metric == "jaccard":
        from scipy.spatial import distance
        for i in range(count.shape[0]):
            it += 1-distance.jaccard(idx[i], idxh[i])
        it = it / count.shape[0]
    return it

