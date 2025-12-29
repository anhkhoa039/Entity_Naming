import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class DipMeans:
    """
    Lightweight Dip-means style clustering:
    - Recursively split clusters that appear multimodal on a 1D projection.
    """

    def __init__(self, alpha: float = 0.05, min_size: int = 10):
        self.alpha = alpha
        self.min_size = min_size

    @staticmethod
    def _project_1d(X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= 2:
            return np.zeros(X.shape[0])
        pca = PCA(n_components=1)
        return pca.fit_transform(X).reshape(-1)

    @staticmethod
    def _is_multimodal(values: np.ndarray, alpha: float) -> bool:
        """
        Surrogate for Hartigan's Dip Test:
        - project to 1D
        - histogram peak counting
        """
        if len(values) < 2:
            return False
        hist, _ = np.histogram(values, bins="auto")
        peaks = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks += 1
        return peaks > 1

    def _split_once(self, X: np.ndarray) -> tuple[bool, np.ndarray]:
        if X.shape[0] < self.min_size:
            return False, np.zeros(X.shape[0], dtype=int)
        proj = self._project_1d(X)
        if not self._is_multimodal(proj, self.alpha):
            return False, np.zeros(X.shape[0], dtype=int)
        km = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        return True, labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        clusters = {0: np.arange(len(X))}
        next_label = 1
        changed = True
        while changed:
            changed = False
            for cid, idxs in list(clusters.items()):
                split, sub = self._split_once(X[idxs])
                if split:
                    left = idxs[sub == 0]
                    right = idxs[sub == 1]
                    del clusters[cid]
                    clusters[next_label] = left
                    clusters[next_label + 1] = right
                    next_label += 2
                    changed = True
                    break
        labels = np.zeros(len(X), dtype=int)
        for cid, idxs in clusters.items():
            labels[idxs] = cid
        return labels


