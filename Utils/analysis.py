import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import umap


"""perform interpolation

1. lerp: linear interpolation
2. slerp: spherical linear interpolation
"""


def lerp(v1, v2, alpha):
    return v1 * (1-alpha) + v2 * alpha


def slerp(v1, v2, alpha):
    is_torch = isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor)
    norm = torch.norm if is_torch else np.linalg.norm
    acos = torch.acos if is_torch else np.arccos
    dot = torch.dot if is_torch else np.dot
    sin = torch.sin if is_torch else np.sin

    z1_normalized = v1 / norm(v1)
    z2_normalized = v2 / norm(v2)
    omega = acos(dot(z1_normalized, z2_normalized))
    
    return (sin((1 - alpha) * omega) * v1 + sin(alpha * omega) * v2) / sin(omega)


"""perform dimension reduction

1. PCA: principal component analysis
2. kernel PCA: kernel principal component analysis
3. t-SNE: t-Distributed Stochastic Neighbor Embedding
4. UMAP
"""


def perform_pca(latent_vectors, n_components=2):
    """Performs PCA on the given latent vectors and returns the transformed latent vectors.
    
    Args:
        latent_vectors: A 2D array of shape (num_samples, num_latent_dimensions) containing the latent vectors.
        n_components: The number of components to keep. Default is 2.
    
    Returns:
        A 2D array of shape (num_samples, n_components) containing the transformed latent vectors.
    """
    pca = PCA(n_components=n_components)
    latent_transformed = pca.fit_transform(latent_vectors)
    return latent_transformed


def perform_kernel_pca(latent_vectors, kernel='rbf', n_components=2):
    """Performs kernel PCA on the given latent vectors and returns the transformed latent vectors.
    
    Args:
        latent_vectors: A 2D array of shape (num_samples, num_latent_dimensions) containing the latent vectors.
        kernel: The kernel to use for the PCA. Default is 'rbf'.
        n_components: The number of components to keep. Default is 2.
    
    Returns:
        A 2D array of shape (num_samples, n_components) containing the transformed latent vectors.
    """
    kpca = KernelPCA(kernel=kernel, n_components=n_components)
    latent_transformed = kpca.fit_transform(latent_vectors)
    return kpca, latent_transformed


def perform_tsne(latent_vectors, n_components=2, perplexity=30,
                 learning_rate=200, n_iter=1000, n_jobs=4):
    """Performs t-SNE on the given latent vectors and returns the transformed latent vectors.
    
    Args:
        latent_vectors: A 2D array of shape (num_samples, num_latent_dimensions) containing the latent vectors.
        n_components: The number of components to keep. Default is 2.
        perplexity: The perplexity parameter for t-SNE. Default is 30.
        learning_rate: The learning rate parameter for t-SNE. Default is 200.
        n_iter: The number of iterations for t-SNE. Default is 1000.
        n_jobs: The number of parallel jobs to run. Default is 4.
    
    Returns:
        A 2D array of shape (num_samples, n_components) containing the transformed latent vectors.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter,
                n_jobs=n_jobs)
    latent_transformed = tsne.fit_transform(latent_vectors)
    return latent_transformed


def perform_umap(latent_vectors, n_components=2, n_neighbors=15,
                 min_dist=0.1, metric='euclidean'):
    """
    Performs UMAP on the given latent vectors and returns the transformed latent vectors.
    :param latent_vectors: A 2D array of shape (num_samples, num_latent_dimensions) containing the latent vectors.
    :param n_components: The number of components to keep. Default is 2.
    :param n_neighbors: The number of neighbors to use for the k-nearest neighbor graph. Default is 15.
    :param min_dist: The minimum distance threshold for the low-dimensional embedding. Default is 0.1.
    :param metric: The distance metric to use. Default is 'euclidean'.
    :return: A 2D array of shape (num_samples, n_components) containing the transformed latent vectors.
    """
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, metric=metric)
    latent_transformed = umap_model.fit_transform(latent_vectors)
    return latent_transformed


"""function wrapper

1. interpolation
2. dimension reduction
"""


interpolation = {
    'lerp': lerp,
    'slerp': slerp
}


dimension_reduction = {
    'pca'       : perform_pca,
    'kernel-pca': perform_kernel_pca,
    't-sne'     : perform_tsne,
    'umap'      : perform_umap
}