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
- PCA removes _linear_ correlations across features (https://scikit-learn.org/stable/modules/preprocessing.html)

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

## [data preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- Standarization as a common requirement for ML estimators: approximately normal distributed
- subtract mean and normalize by the variance
- train_test_split()
- make_pipeline(StandardScaler(), LogisticRegression())
- RobustScaler(): when there are many outliers
- Mapping to a Gaussian distribution (interesting. How to map it back?)
- Encoding categorical features: preprocessing.OrdinalEncoder(encoded_missing_value= -1)

### Normalization
Normalization is the process of scaling individual samples to have unit norm.

### Encoding categorical features
- OrdinalEncoder()
- OneHotEncoder()
    - enc.categories_ - this illustrates the vector
    - enc.transform()
    - infrequent categories: group them into one - Arguments of OneHotEncoder()
- Discretization: quantization or binning


### [Generating polynomial features](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)

### Custom Transformers

- ```from sklearn.preprocessing import FunctionTransformer```
- something like an DataFrame::apply() with a lambda function

## [Imputation](https://scikit-learn.org/stable/modules/impute.html)
- most classifiers or regressors do not accept missing values
- imputation is the process of filling in variables with values calculated under certain assumptions
- [marking imputed values](https://scikit-learn.org/stable/modules/impute.html#marking-imputed-values)
- [Univariate vs. multivariate imputation](https://scikit-learn.org/stable/modules/impute.html#univariate-vs-multivariate-imputation)

## Decision Trees
- a supervised learning technique
- for classification and regression
- for tabular data, but also works for faces
- a model that predicts the value of a target variable by learning
- piecewise constant approximation
- approximating a sin curve with piecewise constant parts (if-else conditions)

### Advantages
- trees can be visualized, easy to understand
- numerical and categorical data
- low data preparation requirements -> low threshold to run it.

### Disadvantages
- prone to overfitting, especially for big amounts of features
- depending on details the tree can have very different shapes
    - addressed via ensemble methods
- needs a balanced dataset

### Classification
- class DecisionTreeClassifier: multi-class
- methods:
    - fit()
    - predict()
    - predict_proba()
    - plot_tree()
    - export_graphviz()

### Multi-output problems
a multi-output problem is a supervised learning problem with n-dimensional output with n>1. I.e. the output has dimension (n_samples, n_outputs).
If there are no correlations between the outputs and the inputs leading to these outputs, one can train individual models for each 1-dimensional output and combine them in the end.

If they are correlated a better method is to train one model capable of predicting simultaneously all n outputs.

1. lower training time required to built only one estimator
2. generalization power is often increased.

### Regressor
- class DecisionTreeRegressor

### Practical Tips
- works well with PCA or LDA
- start with max_depth=3 as initial depth, look at the data, then increase the depth

## [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
In order to increase robustness and generalizability individual estimators can be combined to form a more powerful ensemble.

There are two types of ensemble methods:
1. averaging: The variance of the ensemble is reduced compared to each individual estimator's prediction (bagging methods, random forests)
2. in boosting methods, base estimators are built sequentially with the goal to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble (AdaBoost, Gradient Tree Boosting)

### Bagging meta-estimator
- several instances of a black-box estimator on random subsets of the original training set and then aggregate their predictions to form a final prediction.
- reduce variance 
- avoid overfitting
- bagging methods work best with strong and complex models
- in contrast with boosting methods which usually work best with weak models (e.g. shallow deicsion trees)
- different bagging methods differ in the way they draw subsets of the training set.
- BaggingRegressor() and BaggingClassifier()

### Forests of randomized trees
sklearn.ensemble knows two averaging algorithms based on randomized decision trees using perturb-and-combine techniques:
- the RandomForest algorithm
- the Extra-Trees method
The prediction is the averaged prediction of the individual classifiers.
Reduce variance through introducing randomness. Individual decision trees exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking the average of many such trees, certain errors cancel out.

#### Random Forests
- RandomForestClassifier, RandomForestRegressor
- two sources of randomness: random subsample draw and splitting
    - each tree in the ensemble is built from a sample drawn with replacement
    - when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features.
- Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias.
- look at most discriminative thresholds
#### Extremely randomized trees
... exist as well :)

#### Parameters
- n_estimators (the larger the better, but it takes longer to compute) and 
- max_features (regression: max_features=1.0 down to 0.3, classification: max_features="sqrt")
- results will stop getting significantly better beyond a critical number of trees.
- (...)

### Evaluation and Feature importance evaluation
- based on out-of-bag samples: (oob_score=True)
- the higher up in the tree, the more samples are affected
- expected fraction of the samples as an estimate of the relative importance of the features.
- feature_importances_

### AdaBoost
- 1995 by Freund and Schapire
- fit a sequence of weak learners on repeatedly modified versions of the data. (weak learners are models only weakly better than random guessing).
- A weighted majority of these trees is then the overall prediction.
- boosting iteration: calculate different weights to each training sample (initially 1/N)
- weights of misclassified samples are increased, those of rightly classified samples are decreased. Thereby over the course of several iterations the misclassified samples gain ever more weight and the algorithm has to focus on them.

#### Parameters
- n_estimators (number of weak learners, e.g. 100)
- max_depth: the depth of the base estimators

### Gradient Tree Boosting
- generalization of boosting to arbitrary differentiable loss functions.
(...)
### Histogram-Based Gradient Boosting
(...)

## [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- ```sklearn.feature_selection```
### Removing features with low variance
- VarianceThreshold
- it requires the variance of one feature to go above a given threshold


### Univariate feature selection
- Univariate feature selection works by selecting the best features based on univariate statistical tests.
- scikit-learn exposes feature selection routines as objects that implement the transform method. (why not select-method?)
- SelectKBest(), SelectPercentile()
- Warning: do not use regression scoing functions with a classification problem and vice versa.

### Other methods
- Recursive feature elimination
- Feature selection using SelectFromModel
    - e.g. a Tree-based feature selection
- Sequential Feature Selection
- Feature selection as part of a pipeline

## [Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html)
- Some of the training data is not labeled. The semi-supervised estimators in sklearn.semi_supervised are able to make use of this additional unlabeled data to better capture the shape of the underlying data distribution and generalize better to new samples.
- Particularly suited in case of few labeled and much unlabeled data.
- unlabeled entries in Y get a dedicated identifier along with the labeled data when training the model with the fit method.

1. estimator trained in supervised way with labeled data
2. use the classifier above to train it further in unsupervised fashion

- SelfTrainingClassifier requires predict_proba()
- in each iteration the classifier predicts labels for the unlabeled samples and adds a subset of these labels to the labeled dataset.

### [Semi-supervised learning (Wikipedia)](https://en.wikipedia.org/wiki/Semi-supervised_learning#Assumptions)
- falls between supervised and unsupervised.
- special instance of weak supervision.
- acquisition of unlabeled data is relatively inexpensive
- transductive or inductive learning: The goal of transductive learning is to infer the correct labels for the given unlabeled data. The goal of inductive learning is to infer the correct mapping from input features to output targets.
- in practice, algorithms formally desinged for transduction or induction are often used interchangeably.

#### Assumptions
Semi-supervised learning applies at least one of the following assumptions
- continuity assumption: points that are close to each other are more likely to share a label. This is also a common assumption in supervised learning and gives rise to a simple decision boundary
- cluster assumption: data tend to form clusters and points in the same cluster are more likely to share a label. But data with the same label could spread across different clusters.
- manifold assumption: the data lie approximately on a manifold of much lower dimension than the input space. Learning can then proceed using distances and densities defined on the manifold.
E.g. human voice is controlled by a few vocla folds and facial expressions are controlled by a few muscles. In these cases distances and smoothness in the manifold of the problem are much reduced compared to the full space of all possible acoustic waves or images of faces.




## Glossary
- RANSAC: random sampling consensus - Zufallsstichprobe
- Bottom-Up- and Top-Down-Approaches: Top is where one is, Bottom is where many are.
- splines: piece-wise polynomials
- [scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html#glossary)
- https://en.wikipedia.org/wiki/Duck_typing
- Overfitting: describes the training data well, but does not generalize well to independent test datasets

## Interesting AOB
- [bias-variance decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html): in regression the mean squared error can be decomposed in terms of bias, variance and noise.
- Boolean features are Bernoulli random variables
- [Kernel methods to project data into alternate dimensional spaces](https://scikit-learn.org/stable/modules/semi_supervised.html#label-propagation)
# Neural Networks
see also [summary_Neural_Networks.md](summary_Neural_Networks.md)
