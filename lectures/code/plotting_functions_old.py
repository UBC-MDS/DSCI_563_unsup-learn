import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
from scipy.spatial import distance
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import scipy

# Adapted from the mglearn package
# https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_helpers.py#L27

data = np.random.rand(4, 4)
fig, (ax1, ax2) = plt.subplots(2)
ax1.imshow(data)
ax1.set_title("Default colormap")
plt.rc('image', cmap='viridis')
ax2.imshow(data)
ax2.set_title("Set default colormap")
colors = ['xkcd:azure', 'yellowgreen', 'tomato', 'teal', 'orangered', 'orchid', 'black', 'wheat']

def update_Z(X, centers):
    """
    returns distances and updated cluster assignments
    """
    dist = euclidean_distances(X, centers)
    return dist, np.argmin(dist, axis=1)


def update_centers(X, Z, old_centers, k):
    """
    returns new centers
    """
    new_centers = old_centers.copy()
    for kk in range(k):
        new_centers[kk] = np.mean(X[Z == kk], axis=0)
    return new_centers


def discrete_scatter(x1, x2, y=None, markers=None, s=8, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=0.6, 
                     label_points=False):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.
    Parameters
    ----------
    x1 : nd-array
        input data, first axis
    x2 : nd-array
        input data, second axis
    y : nd-array
        input data, discrete labels
    cmap : colormap
        Colormap to use.
    markers : list of string
        List of markers to use, or None (which defaults to 'o').
    s : int or float
        Size of the marker
    padding : float
        Fraction of the dataset range to use for padding the axes.
    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))        

    # unique_y = np.unique(y)
    unique_y, inds = np.unique(y, return_index=True)    

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    if len(unique_y) == 1: 
        cr = [-1]
    else: 
        cr = sorted([y[index] for index in sorted(inds)])

    for (i, (yy, color_ind)) in enumerate(zip(unique_y, cr)):
        mask = y == yy
        # print(f'color_ind= {color_ind} and i = {i}')
        # if c is none, use color cycle
        color = colors[color_ind]
        # print('color: ', color)
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .2:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,                             
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])
    if label_points: 
        labs = [str(label) for label in list(range(0,len(x1)))]
        for i, txt in enumerate(labs):
            font_size=10
            plt.annotate(txt, (x1[i], x2[i]), xytext= (x1[i]-0.1, x2[i]+0.5), c='k', size = font_size)

    return lines    


from scipy.spatial.distance import cdist

def plot_kmeans_circles(kmeans, X, n_clusters=3, ax=None):
    km_labels = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_
    
    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    discrete_scatter(X[:,0], X[:,1], km_labels, c=km_labels, markers='o', markeredgewidth=0.2, ax=ax);
    discrete_scatter(
        centers[:, 0], centers[:, 1], y=[0,1,2], markers="*", s=18
    );
    
    radii = [cdist(X[km_labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for center, radius in zip(centers, radii):
        ax.add_patch(plt.Circle(center, radius, fc='gray', alpha=0.4))
        

def plot_example_dist(data, centroids, fig, fontsize = 16, point_ind=None, ax=None):
    """
    Plot the distance of a point to the centroids.

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas dataframe with X1 and X2 coordinate. If more than two
        coordinates, only the first two will be used.
    centroids: pd.DataFrame
        A pandas dataframe composed by k rows of data, chosen randomly. (where k 
        stands for the number of clusters)
    w: int
        width of the plot
    h: int
        height of the plot
    point: int
        the index of the point to be used to calculate the distance
    """
    if ax is None:
        ax = plt.gca()
    k = centroids.shape[0]
    if point_ind is None:
        point = np.random.choice(range(0, data.shape[0]), size=1)

    point = data[point_ind, 0:2]
    centroids = centroids[:, 0:2]

    discrete_scatter(data[:, 0], data[:, 1], s=14, label_points=True, ax=ax)
    discrete_scatter(centroids[:, 0], centroids[:, 1], y=[0,1,2], s=18,
                markers='*', ax=ax)
    # ax.set_xlabel(data.columns[0], fontdict={'fontsize': fontsize})
    # ax.set_ylabel(data.columns[1], fontdict={'fontsize': fontsize})
    #ax.scatter(point[0], point[1])
    
    dist = np.zeros(k)
    for i in range(0, k):
        l = np.row_stack((point, centroids[i, :]))
        dist[i] = np.sum((point-centroids[i, :])**2)**0.5                 
        ax.plot(l[:, 0], l[:, 1], c=colors[i], linewidth=1.0, linestyle='-.')
        if (l[0, 1] <= l[1, 1]):
            ax.text(l[1, 0]+.20, l[1, 1]+.2,
                     f"d = {np.round(dist[i], 2)}", color=colors[i],
                     fontdict={'fontsize': fontsize})
        else:
            ax.text(l[1, 0]+.15, l[1, 1]+.2,
                     f"d = {np.round(dist[i], 2)}", color=colors[i],
                     fontdict={'fontsize': fontsize})

    i = np.argmin(dist)
    l = np.row_stack((point, centroids[i, :]))
    ax.plot(l[:, 0], l[:, 1], c=colors[i], linewidth=3.0, linestyle='-')
    title = f"Point {point_ind} will be assigned to {colors[np.argmin(dist)]} cluster (min dist = {np.round(np.min(dist),2)})"
    ax.set_title(title, fontdict={'fontsize': fontsize});
    plt.close()
    return fig
    
def plot_km_initialization(X, centers):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))     
    discrete_scatter(X[:, 0], X[:, 1], markers="o", ax=ax[0]);
    ax[0].set_title("Before clustering");    
    discrete_scatter(X[:, 0], X[:, 1], markers="o", ax=ax[1])
    discrete_scatter(
        centers[:, 0], centers[:, 1], y=np.arange(len(centers)), markers="*", s=14, ax=ax[1]
    );    
    ax[1].set_title("Initial centers");    
    
    
def plot_km_iteration(X, Z, centers, new_centers, iteration, fig, ax, fontsize=18):
    discrete_scatter(X[:,0], X[:,1], y=Z.tolist(), markers='o', s=12, ax = ax[0])
    discrete_scatter(centers[:,0], centers[:,1], y=np.arange(len(centers)), markers='*',s=18, ax = ax[0])
    ax[0].set_title(f'Iteration: {iteration}: Update Z', fontdict={'fontsize': fontsize})    
    discrete_scatter(X[:,0], X[:,1], y=Z.tolist(), markers='o', s=12, label_points=True, ax = ax[1])
    discrete_scatter(new_centers[:,0], new_centers[:,1], y=np.arange(len(centers)), markers='*',s=18, ax = ax[1])    
    aux = new_centers-(centers+(new_centers-centers)*0.9)
    aux = np.linalg.norm(aux, axis=1)    
    for i in range(0, 3):
        ax[1].arrow(centers[i, 0], centers[i, 1],
                  (new_centers[i, 0]-centers[i, 0])*0.8,
                  (new_centers[i, 1]-centers[i, 1])*0.8,
                  head_width=.1, head_length=aux[i], fc=colors[i], ec=colors[i])    
    ax[1].set_title(f'Iteration: {iteration}: Update cluster centers', fontdict={'fontsize': fontsize})
    plt.close()    
    return fig


def plot_km_iterative(X, starting_centroid, iterations=5, k=3):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    k = starting_centroid.shape[0]
    x = X[:, 0:2]
    centroids = starting_centroid.copy()
    dist, Z = update_Z(x, centroids)        
    y, inds = np.unique(Z, return_index=True)    
    # c = [Z[index] for index in sorted(inds)]
    discrete_scatter(x[:, 0], x[:, 1], y=Z, markers="o", ax=ax[0])              
    discrete_scatter(centroids[:, 0], centroids[:, 1], y=[0,1,2], markers='*', s=16, ax=ax[0])

    ax[0].set_title('Initial centers and cluster assignments')
    
    for i in range(iterations):
        discrete_scatter(x[:, 0], x[:, 1], y=Z, c=Z, markers="o", ax=ax[1])        
        new_centroids = update_centers(x, Z, centroids, k)            
        dist, Z = update_Z(x, new_centroids)
        discrete_scatter(new_centroids[:, 0], new_centroids[:, 1], y=[0,1,2], markers='*', s=16, ax=ax[1])

        aux = new_centroids-(centroids+(new_centroids-centroids)*0.9)
        aux = np.linalg.norm(aux, axis=1)
        for i in range(0, 3):
            if aux[i] > .005:
                plt.arrow(centroids[i, 0], centroids[i, 1],
                          (new_centroids[i, 0]-centroids[i, 0])*0.8,
                          (new_centroids[i, 1]-centroids[i, 1])*0.8,
                          head_width=.25, head_length=aux[i], fc=colors[i], ec=colors[i])
        centroids = new_centroids
        
    
    #plt.xlabel(data.columns[0], fontdict={'fontsize': w})
    #plt.ylabel(data.columns[1], fontdict={'fontsize': w})
    ax[1].set_title(f"Centers and cluster assignments after {iterations} iteration(s)")

def plot_silhouette_dist(w, h):

    n = 30
    df, target = make_blobs(n_samples=n,
                            n_features=2,
                            centers=[[0, 0], [1, 1], [2.5, 0]],
                            cluster_std=.15,
                            random_state=1)

    colors = np.array(['black', 'blue', 'red'])

    plt.figure(figsize=(w, h))
    ax = plt.gca()
    ax.set_ylim(-.45, 1.4)
    ax.set_xlim(-.25, 2.8)
    plt.scatter(df[:, 0], df[:, 1], c=colors[target])

    p = 1
    for i in range(0, n):
        plt.plot((df[p, 0], df[i, 0]), (df[p, 1], df[i, 1]),
                 linewidth=.7, c=colors[target[i]])

    plt.scatter(df[p, 0], df[p, 1], c="green", zorder=10, s=200)

    c1 = Circle((.1, -.12), 0.27, fill=False, linewidth=2, color='black')
    c2 = Circle((1.03, 1.04), 0.27, fill=False, linewidth=2, color='blue')
    c3 = Circle((2.48, 0.1), 0.27, fill=False, linewidth=2, color='red')
    ax.add_artist(c1)
    ax.add_artist(c2)
    ax.add_artist(c3)
    plt.xlabel("X1", fontdict={'fontsize': w})
    plt.ylabel("X2", fontdict={'fontsize': w})
    plt.title("Distances for silhouette", fontdict={'fontsize': w+h})


def Gaussian_mixture_1d(ϕ1, ϕ2, fig, μ1=0.0, μ2=5.0, Σ1=1, Σ2=3):
    """
    Plot a Gaussian Mixture with two components. 
    
    Parameters:
    -----------
    μ1: float
       the mean of the first Gaussian
    μ2: float
       the mean of the second Gaussian 
    Σ1: float
       the variance of the first Gaussian
    Σ2: float
       the variance of the second Gaussian
    ϕ1: float > 0
       the weight of the first component
    ϕ2: float > 0
       the weight of the second component
    w: int
       The width of the plot
    h: int
       the height of the plot       
    """

    # Creating the DataFrame
    data = pd.DataFrame({'x': np.arange(np.min([μ1 - 4*Σ1, μ2 - 4*Σ2]), np.max([μ1 + 4*Σ1, μ2 + 4*Σ2]), 1/1000)})
    data['f1(x|mu1,Sigma1)'] = scipy.stats.norm.pdf(data['x'], μ1, Σ1)
    data['f2(x|mu2,Sigma2)'] = scipy.stats.norm.pdf(data['x'], μ2, Σ2)
    data['mixture'] = ϕ1 * data['f1(x|mu1,Sigma1)'] + ϕ2 * data['f2(x|mu2,Sigma2)']


    ## Plotting
    plt.plot('x', 'mixture', data=data, label='Mixture')
    plt.plot('x', 'f1(x|mu1,Sigma1)', linestyle='--', alpha = .35, data=data, color='green', label=f'$f_1(x|\mu_1={μ1},\Sigma_1={Σ1})$')
    plt.plot('x', 'f2(x|mu2,Sigma2)', linestyle='--', alpha = .35, data=data, color='red', label=f'$f_2(x|\mu_2={μ2},\Sigma_2={Σ2})$')
    plt.legend(fontsize=16)
    plt.title(f'Gaussian Mixture: $\Phi_1$ = {round(ϕ1,2)} and $\Phi_2$ = {round(ϕ2,2)}', fontdict={'fontsize':18})
    plt.close()    
    return fig


def plot_cov_types(X_train, gmm_full_labels, gmm_tied_labels, gmm_diag_labels, gmm_spherical_labels): 
    fig, ax = plt.subplots(2, 2, figsize=(12, 8)) 
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_spherical_labels, c=gmm_spherical_labels, markers="o", ax=ax[0][0]);
    ax[0][0].set_title('Spherical');
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_diag_labels, c=gmm_diag_labels, markers="o", ax=ax[0][1]);
    ax[0][1].set_title('diag')
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_tied_labels, c=gmm_tied_labels, markers="o", ax=ax[1][0]);
    ax[1][0].set_title('tied')
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_full_labels, c=gmm_full_labels, markers="o", ax=ax[1][1]);
    ax[1][1].set_title('full')
    
def make_ellipses(gmm, ax):
    colors = ['xkcd:azure', 'yellowgreen', 'tomato']    
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], 2*v[0], 2*v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")    
        
def plot_gmm_cov_types(estimators, X_train): 
    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 5))
    plt.subplots_adjust(
        bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
    )

    for index, (name, estimator) in enumerate(estimators.items()):
        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)
        labels = estimator.predict(X_train)
        h = plt.subplot(2, n_estimators // 2, index + 1)
        discrete_scatter(X_train[:, 0], X_train[:, 1], labels, c=labels, markers="o", markeredgewidth=0.2, ax=h);    
        make_ellipses(estimator, h)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc="upper right", prop=dict(size=12))

    
def get_cluster_images(model, Z, inputs, cluster=0, n_img=5):
    fig, axes = plt.subplots(1, n_img + 1, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(10, 10), gridspec_kw={"hspace": .3})
    img_shape = [3,200,200]
    transpose_axes = (1,2,0)      
    
    if type(model).__name__ == 'KMeans': 
        center = model.cluster_centers_[cluster]
        mask = model.labels_ == cluster
        dists = np.sum((Z - center) ** 2, axis=1)
        dists[~mask] = np.inf
        inds = np.argsort(dists)[:n_img]        
        if Z.shape[1] == 1024: 
            axes[0].imshow(center.reshape((32,32)))
        else:
            axes[0].imshow(np.transpose(center.reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster center %d'%(cluster))       
    if type(model).__name__ == 'GaussianMixture':         
        cluster_probs = model.predict_proba(Z)[:,cluster]
        inds = np.argsort(cluster_probs)[-n_img:]        
        axes[0].imshow(np.transpose(inputs[inds[0]].reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster %d'%(cluster))   
        
    i = 1
    print(inds)
    for image in inputs[inds]:
        axes[i].imshow(np.transpose(image/2 + 0.5 , transpose_axes))
        i+=1
    plt.show()    
    
def plot_sup_x_unsup(data, w, h):
    """
        Function to generate a supervised vs unsupervised plot.
        Parameters:
        -----------
        data: pd.DataFrame
            A pandas dataframe with X1 and X2 coordinate, and a target column
            for the classes.
        w: int
            Width of the plot
        h: int
            height of the plot
    """
    # Colors to be used (upt to 5 classes)
    colors = np.array(['black', 'blue', 'red', 'green', 'purple'])

    # Getting the column and classes' names
    col_names = data.columns.to_numpy()
    target_names = data['target'].to_numpy()

    # Getting numerical values for the classes labels
    target = np.unique(data['target'].to_numpy(), return_inverse=True)

    # Getting X1 and X2
    data = data.iloc[:, 0:2].to_numpy()

    # Creates the Figure
    plt.figure(0, figsize=(w, h))

    # Create two subplots
    plt.subplots_adjust(right=2.5)

    # Get the first subplot, which is the Supervised one.
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    for i, label in enumerate(target[0]):
        plt.scatter(data[target_names == label, 0],
                    data[target_names == label, 1],
                    c=colors[i], label=label)

    # Creates the legend
    plt.legend(loc='best', fontsize=22, frameon=True)

    # Name the axes and creates title
    plt.xlabel(col_names[0], fontsize=1.5*(w + h))
    plt.ylabel(col_names[1], fontsize=1.5*(w + h))
    plt.title("Supervised", fontdict={'fontsize': 2 * (w + h)})

#     ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': w + h})
#     ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': w + h})

    # Creates the unsupervised subplot.
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("Unsupervised", fontdict={'fontsize': 2 * (w + h)})
    plt.xlabel(col_names[0], fontsize=1.5*(w + h))
    plt.ylabel(col_names[1], fontsize=1.5*(w + h))
#     ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': w + h})
#     ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': w + h})
    
    
    
