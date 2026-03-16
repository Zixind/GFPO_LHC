"""
Step 1: Extract RNN + Linear head weights from a PyTorch StateEncoder checkpoint.

Saves weight matrices as .npy files for downstream Keras model building,
without importing or modifying any RL/ source files.

Usage:
    python firmware/extract_weights.py \
        --checkpoint outputs/.../model.pt \
        --rnn-type gru \
        --output-dir firmware/weights/
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import torch


def extract_weights(checkpoint_path: str, rnn_type: str = "gru"):
    """
    Load a PyTorch checkpoint and return RNN + head weight arrays.

    Handles multiple checkpoint formats:
      - Raw state_dict
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - Nested encoder prefixes (encoder.rnn.*, policy.encoder.rnn.*, etc.)

    Returns
    -------
    dict with keys: W_ih, W_hh, b_ih, b_hh, W_head, b_head,
                    feat_dim, hidden, out_dim, rnn_type
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Unwrap checkpoint container
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    # Find RNN weight keys (handle arbitrary prefix depth)
    rnn_prefix = None
    for k in sd:
        if "rnn.weight_ih_l0" in k:
            rnn_prefix = k[: k.index("rnn.weight_ih_l0")] + "rnn."
            break
    if rnn_prefix is None:
        raise KeyError(
            f"Cannot find rnn.weight_ih_l0 in checkpoint.\n"
            f"Available keys: {sorted(sd.keys())}"
        )

    W_ih = sd[f"{rnn_prefix}weight_ih_l0"].numpy()
    W_hh = sd[f"{rnn_prefix}weight_hh_l0"].numpy()
    b_ih = sd[f"{rnn_prefix}bias_ih_l0"].numpy()
    b_hh = sd[f"{rnn_prefix}bias_hh_l0"].numpy()

    # Find head weights
    head_prefix = None
    for k in sd:
        if "head.weight" in k:
            head_prefix = k[: k.index("head.weight")] + "head."
            break
    if head_prefix is None:
        raise KeyError(
            f"Cannot find head.weight in checkpoint.\n"
            f"Available keys: {sorted(sd.keys())}"
        )

    W_head = sd[f"{head_prefix}weight"].numpy()
    b_head = sd[f"{head_prefix}bias"].numpy()

    hidden = W_hh.shape[1]
    feat_dim = W_ih.shape[1]
    out_dim = W_head.shape[0]

    return {
        "W_ih": W_ih, "W_hh": W_hh, "b_ih": b_ih, "b_hh": b_hh,
        "W_head": W_head, "b_head": b_head,
        "feat_dim": feat_dim, "hidden": hidden, "out_dim": out_dim,
        "rnn_type": rnn_type,
    }


def save_weights(weights: dict, output_dir: str):
    """Save extracted weight arrays as .npy files + a metadata .npz."""
    os.makedirs(output_dir, exist_ok=True)

    for name in ["W_ih", "W_hh", "b_ih", "b_hh", "W_head", "b_head"]:
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, weights[name])

    # Save metadata
    meta = {
        "feat_dim": np.array(weights["feat_dim"]),
        "hidden": np.array(weights["hidden"]),
        "out_dim": np.array(weights["out_dim"]),
        "rnn_type": np.array(weights["rnn_type"]),
    }
    np.savez(os.path.join(output_dir, "metadata.npz"), **meta)


def load_weights(weights_dir: str) -> dict:
    """Load previously saved weight arrays from a directory."""
    weights = {}
    for name in ["W_ih", "W_hh", "b_ih", "b_hh", "W_head", "b_head"]:
        weights[name] = np.load(os.path.join(weights_dir, f"{name}.npy"))

    meta = np.load(os.path.join(weights_dir, "metadata.npz"), allow_pickle=True)
    weights["feat_dim"] = int(meta["feat_dim"])
    weights["hidden"] = int(meta["hidden"])
    weights["out_dim"] = int(meta["out_dim"])
    weights["rnn_type"] = str(meta["rnn_type"])

    return weights


def main():
    parser = argparse.ArgumentParser(description="Extract StateEncoder weights")
    parser.add_argument("--checkpoint", required=True, help="PyTorch .pt file")
    parser.add_argument("--rnn-type", default="gru",
                        choices=["gru", "rnn", "rnn_relu", "lstm"])
    parser.add_argument("--output-dir", default="firmware/weights/")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    weights = extract_weights(args.checkpoint, args.rnn_type)

    print(f"\nExtracted weights:")
    print(f"  rnn_type  = {weights['rnn_type']}")
    print(f"  feat_dim  = {weights['feat_dim']}")
    print(f"  hidden    = {weights['hidden']}")
    print(f"  out_dim   = {weights['out_dim']}")
    print(f"  W_ih      : {weights['W_ih'].shape}")
    print(f"  W_hh      : {weights['W_hh'].shape}")
    print(f"  W_head    : {weights['W_head'].shape}")

    save_weights(weights, args.output_dir)
    print(f"\nWeights saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
