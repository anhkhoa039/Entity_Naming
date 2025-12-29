import json
from typing import List

import numpy as np

from .encoding import Mention


def load_mentions(dataset_path: str) -> List[Mention]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mentions: List[Mention] = []
    for item in data:
        sid = item["id"]
        words = item["str_words"]
        tags = item.get("tags_ner_pred", [])
        for span in tags:
            start, end, _ = span
            text = " ".join(words[start:end])
            context = " ".join(words)
            mentions.append(Mention((sid, start, end), text, context))
    return mentions


def select_examples(X: np.ndarray, mentions: List[Mention], labels: np.ndarray, k: int = 5):
    examples = {}
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        cluster_vecs = X[idxs]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists)
        centroids = [mentions[idxs[i]].text for i in order[:k]]
        boundaries = [mentions[idxs[i]].text for i in order[-k:]] if len(order) >= k else []
        examples[int(cid)] = {"centroids": centroids, "boundaries": boundaries}
    return examples


