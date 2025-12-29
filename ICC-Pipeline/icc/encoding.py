from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class Mention:
    span_id: Tuple[int, int, int]  # (sentence_id, start, end)
    text: str
    context: str


class MentionEncoder:
    """
    Prompt-based mention encoder:
    Input prompt: "<context> [SEP] <mention> is a [MASK]"
    Representation: hidden state at [MASK]
    """

    def __init__(self, model_name: str, device: str = None, max_length: int = 196):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def encode(self, mentions: List[Mention], batch_size: int = 16) -> np.ndarray:
        embs: List[np.ndarray] = []
        mask_token = self.tokenizer.mask_token
        for i in range(0, len(mentions), batch_size):
            batch = mentions[i : i + batch_size]
            prompts = [
                f"{m.context} {self.tokenizer.sep_token} {m.text} is a {mask_token}"
                for m in batch
            ]
            enc = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state  # [B, L, H]
            input_ids = enc.input_ids
            mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
            mask_hidden = []
            for idx in range(input_ids.size(0)):
                pos = mask_positions[mask_positions[:, 0] == idx]
                if len(pos) == 0:
                    mask_hidden.append(out[idx, 0])  # fallback to CLS
                else:
                    mask_hidden.append(out[idx, pos[0, 1]])
            batch_emb = torch.stack(mask_hidden).cpu().numpy()
            embs.append(batch_emb)
        return np.concatenate(embs, axis=0)


