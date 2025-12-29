"""
Iterative Cluster-Critic (ICC) pipeline: Modules II & III from the
“Framework for Iterative NER Cluster Refinement”.

Module I (mention detection) is assumed to be completed via
DTrans-MPrompt/Entity-Detection. This script starts from detected spans
exported into Type-Prediction/dataset/<target>_<split>.json and runs:
  1) Prompt-based mention encoding
  2) Adaptive (Dip-means style) clustering
  3) Critic-based refinement loop (merge/split via confusion matrix)

Outputs:
  - <output_dir>/clusters.json: per-span cluster assignments
  - <output_dir>/cluster_examples.json: examples for naming (centroids/boundaries)
"""

import argparse
import json
import os

from icc.clustering import DipMeans
from icc.critic import CriticRefiner
from icc.data_utils import load_mentions, select_examples
from icc.encoding import MentionEncoder
from icc.naming import build_prompts, save_prompts, call_ollama


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True, help="Path to target dataset json (with tags_ner_pred)")
    ap.add_argument("--model_name", default="bert-base-cased", help="Encoder model")
    ap.add_argument("--output_dir", required=True, help="Where to write cluster outputs")
    ap.add_argument("--alpha", type=float, default=0.05, help="Dip-means alpha")
    ap.add_argument("--tau_merge", type=float, default=0.35, help="Merge threshold")
    ap.add_argument("--tau_split", type=float, default=0.60, help="Split threshold")
    ap.add_argument("--max_iter", type=int, default=5, help="Max refinement iterations")
    ap.add_argument("--ami_eps", type=float, default=None, help="Stop when ΔAMI < ami_eps (optional)")
    ap.add_argument(
        "--critic_model",
        type=str,
        default="logreg",
        choices=["logreg", "distilbert"],
        help="Critic model used in refinement loop",
    )
    ap.add_argument("--critic_name", type=str, default="distilbert-base-cased", help="DistilBERT model name")
    ap.add_argument("--critic_lr", type=float, default=5e-5, help="DistilBERT critic learning rate")
    ap.add_argument("--critic_epochs", type=int, default=3, help="DistilBERT critic epochs per loop")
    ap.add_argument("--critic_batch", type=int, default=16, help="DistilBERT critic batch size")
    ap.add_argument(
        "--emit_prompts",
        action="store_true",
        help="Also emit naming prompts (Module IV) to naming_prompts.json",
    )
    ap.add_argument(
        "--ollama_model",
        type=str,
        default=None,
        help="If set, run naming prompts through this Ollama model and save results",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mentions = load_mentions(args.dataset_path)
    if len(mentions) == 0:
        raise ValueError("No mentions found in dataset. Ensure tags_ner_pred exists.")

    encoder = MentionEncoder(args.model_name)
    X = encoder.encode(mentions)

    prompt_texts = [
        f"{m.context} {encoder.tokenizer.sep_token} {m.text} is a {encoder.tokenizer.mask_token}"
        for m in mentions
    ]

    dip = DipMeans(alpha=args.alpha)
    init_labels = dip.fit_predict(X)

    refiner = CriticRefiner(
        tau_merge=args.tau_merge,
        tau_split=args.tau_split,
        max_iter=args.max_iter,
        critic_model=args.critic_model,
        critic_name=args.critic_name,
        critic_lr=args.critic_lr,
        critic_epochs=args.critic_epochs,
        critic_batch=args.critic_batch,
        ami_eps=args.ami_eps,
    )
    refined = refiner.refine(X, init_labels, prompt_texts)

    clusters_out = [
        {"span_id": mentions[i].span_id, "text": mentions[i].text, "cluster": int(refined[i])}
        for i in range(len(mentions))
    ]
    with open(os.path.join(args.output_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters_out, f, indent=2, ensure_ascii=False)

    examples = select_examples(X, mentions, refined)
    with open(os.path.join(args.output_dir, "cluster_examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    if args.emit_prompts:
        prompts = build_prompts(examples)
        prompts_path = os.path.join(args.output_dir, "naming_prompts.json")
        save_prompts(prompts, prompts_path)
        print(f"Saved naming prompts to {prompts_path}")
        if args.ollama_model:
            results = call_ollama(prompts, args.ollama_model, examples)
            results_path = os.path.join(args.output_dir, "naming_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved Ollama naming results to {results_path}")

    print(f"Saved clusters to {args.output_dir}")


if __name__ == "__main__":
    main()

