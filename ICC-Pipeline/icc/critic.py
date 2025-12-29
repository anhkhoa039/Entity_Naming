from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


class CriticRefiner:
    """
    Merge/split refinement driven by a critic model.

    critic_model: "logreg" (fast, on embeddings) or "distilbert" (text-based).
    """

    def __init__(
        self,
        tau_merge: float = 0.35,
        tau_split: float = 0.60,
        max_iter: int = 5,
        min_cluster: int = 5,
        critic_model: str = "logreg",
        critic_name: str = "distilbert-base-cased",
        critic_lr: float = 5e-5,
        critic_epochs: int = 3,
        critic_batch: int = 16,
        device: str = None,
        ami_eps: float = None,
    ):
        self.tau_merge = tau_merge
        self.tau_split = tau_split
        self.max_iter = max_iter
        self.min_cluster = min_cluster
        self.critic_model = critic_model
        self.critic_name = critic_name
        self.critic_lr = critic_lr
        self.critic_epochs = critic_epochs
        self.critic_batch = critic_batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ami_eps = ami_eps

    # --- Logistic Regression critic (embedding -> cluster)
    def _train_critic_logreg(self, X: np.ndarray, y: np.ndarray):
        clf = LogisticRegression(max_iter=200, multi_class="auto", n_jobs=-1)
        clf.fit(X, y)
        return clf

    # --- DistilBERT critic (text -> cluster)
    def _train_critic_distilbert(self, texts: List[str], labels: np.ndarray, num_labels: int):
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.critic_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.critic_name, num_labels=num_labels
        ).to(self.device)

        def encode_batch(batch_texts):
            return tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=196,
                return_tensors="pt",
            ).to(self.device)

        dataset = TensorDataset(torch.tensor(labels, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.critic_batch, shuffle=True)
        optim = torch.optim.AdamW(model.parameters(), lr=self.critic_lr)

        model.train()
        for _ in range(self.critic_epochs):
            for batch_indices in loader:
                idx_tensor = batch_indices[0]
                idx_list = idx_tensor.cpu().tolist()
                batch_texts = [texts[i] for i in idx_list]
                enc = encode_batch(batch_texts)
                lbl = torch.tensor([labels[i] for i in idx_list], dtype=torch.long, device=self.device)
                out = model(**enc, labels=lbl)
                out.loss.backward()
                optim.step()
                optim.zero_grad()
        return tokenizer, model

    def _predict_distilbert(self, tokenizer, model, texts: List[str]) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.critic_batch):
                batch_texts = texts[i : i + self.critic_batch]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=196,
                    return_tensors="pt",
                ).to(self.device)
                logits = model(**enc).logits
                preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    # --- Helpers
    def _merge_candidates(self, cm: np.ndarray, labels: np.ndarray) -> List[Tuple[int, int]]:
        merges = []
        sizes = cm.sum(axis=1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                scs = (cm[i, j] + cm[j, i]) / max(1e-6, sizes[i] + sizes[j])
                if scs > self.tau_merge:
                    merges.append((labels[i], labels[j]))
        return merges

    def _split_candidates(self, cm: np.ndarray, labels: np.ndarray) -> List[int]:
        splits = []
        sizes = cm.sum(axis=1)
        for i, lab in enumerate(labels):
            if sizes[i] < self.min_cluster:
                continue
            ics = cm[i, i] / max(1e-6, sizes[i])
            if ics < self.tau_split:
                splits.append(lab)
        return splits

    # --- Main refinement loop
    def refine(self, X: np.ndarray, y: np.ndarray, texts: List[str]) -> np.ndarray:
        labels = y.copy()
        prev_labels = None
        for _ in range(self.max_iter):
            train_idx, val_idx, y_train, y_val = train_test_split(
                np.arange(len(labels)),
                labels,
                test_size=0.2,
                random_state=0,
                stratify=labels,
            )

            if self.critic_model == "distilbert":
                train_texts = [texts[i] for i in train_idx]
                val_texts = [texts[i] for i in val_idx]
                num_labels = len(np.unique(labels))
                tok, model = self._train_critic_distilbert(train_texts, y_train, num_labels)
                preds_val = self._predict_distilbert(tok, model, val_texts)
                cm_labels = np.unique(labels)
                cm = confusion_matrix(y_val, preds_val, labels=cm_labels)
            else:
                X_train = X[train_idx]
                X_val = X[val_idx]
                critic = self._train_critic_logreg(X_train, y_train)
                preds_val = critic.predict(X_val)
                cm_labels = np.unique(labels)
                cm = confusion_matrix(y_val, preds_val, labels=cm_labels)

            merges = self._merge_candidates(cm, cm_labels)
            splits = self._split_candidates(cm, cm_labels)

            new_labels = labels.copy()
            # merge
            for a, b in merges:
                new_labels[labels == b] = a
            # split
            for lab in splits:
                idxs = np.where(new_labels == lab)[0]
                if len(idxs) < self.min_cluster * 2:
                    continue
                km = KMeans(n_clusters=2, n_init=10, random_state=0)
                sub_labels = km.fit_predict(X[idxs])
                new_id = new_labels.max() + 1
                new_labels[idxs[sub_labels == 1]] = new_id

            # reindex to consecutive ids for stability
            uniq = np.unique(new_labels)
            remap = {old: i for i, old in enumerate(uniq)}
            new_labels = np.array([remap[v] for v in new_labels])

            # AMI-based early stop if requested
            if self.ami_eps is not None and prev_labels is not None:
                delta_ami = adjusted_mutual_info_score(prev_labels, new_labels)
                if delta_ami < self.ami_eps:
                    labels = new_labels
                    break

            if np.array_equal(new_labels, labels):
                break
            prev_labels = labels
            labels = new_labels
        return labels


