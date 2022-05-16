# Machine Learning Algorithms
## Clustering Algorithms
### k-Means
- assume k clusters
- loss function: for each point determine the minimum distance to all clusters and sum these distances up across all points. Minimize the sum of all distances wrt to the position of the clusters.
- regularization parameters: could be the distance between clusters or a penalty term for the number of clusters or calculate the distance^k.
- start values: one in the center of gravity, then along the principal components.

in https://scikit-learn.org/stable/modules/clustering.html#k-means
- start with clusters
- minimize in-cluster distances
- Voronoi-Diagrams
- assumes convex shaped clusters
- PCA can help

- starting point: select k clusters
- iterate:
    - assign to cluster
    - calculate new centroid based on the members of the cluster
- stop until the cluster centroids barely move

- "k-means++" initialization
- Mini-batch 

excellent description:
https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6
- k-Means in scikit learn has a parameter n_int= 20 -> select 20 random clusters
- number of iterations

- all features have to have the same scale 

### dimensionality reduction: PCA, t-SNE
- PCA: Eigenvalues of data
- t-SNE: T-distributed stochastic value embedding
- [FeatureAgglomoration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration)

### [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
Areas of high density surrounded by areas of low density. 

Keywords: core_sample, eps and min_sample parameters

a local definition of a cluster from core samples and chaining them together

A core_samples is defined as a subsample of the dataset with each core_sample having at least min_sample other core_samples within a distance of eps.
Thus core_samples are within a dense area of the space in which the dataset lives.
A cluster is a set of core_samples that have neighbors as core_samples, who again have core_samples as neighbors. Samples that are not core_samples anymore, because of less than min_samples within a distance eps define the edge of a cluster.

#### Advantages over k-Means
does not require the clusters to be convex and isotropic.

### hierarchical clustering
## Glossary
- RANSAC: random sampling consensus - Zufallsstichprobe
- Bottom-Up- and Top-Down-Approaches: Top is where one is, Bottom is where many are.

