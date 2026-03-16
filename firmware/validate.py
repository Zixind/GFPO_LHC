"""
Step 3: Validate that the unrolled Keras model produces identical outputs
to the original PyTorch StateEncoder.

Loads the PyTorch checkpoint, builds both the PyTorch and Keras models,
runs random inputs through both, and reports numerical agreement.
This does NOT import from RL/ — it reimplements the minimal PyTorch forward
pass needed for comparison.

Usage:
    python firmware/validate.py \
        --checkpoint outputs/.../model.pt \
        --weights-dir firmware/weights/ \
        --seq-len 10 \
        --rnn-type gru \
        --n-samples 500
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn


def build_pytorch_model(checkpoint_path: str, rnn_type: str, feat_dim: int,
                        hidden: int, out_dim: int):
    """
    Build a minimal PyTorch RNN + Linear model and load weights.
    Standalone — does not import from RL/.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    # Build RNN
    if rnn_type == "gru":
        rnn = nn.GRU(feat_dim, hidden, num_layers=1, batch_first=True)
    elif rnn_type == "rnn":
        rnn = nn.RNN(feat_dim, hidden, num_layers=1, batch_first=True,
                     nonlinearity="tanh")
    elif rnn_type == "rnn_relu":
        rnn = nn.RNN(feat_dim, hidden, num_layers=1, batch_first=True,
                     nonlinearity="relu")
    else:
        raise ValueError(f"Unsupported rnn_type={rnn_type}")

    head = nn.Linear(hidden, out_dim)

    # Load weights — find the right prefix
    rnn_prefix = None
    for k in sd:
        if "rnn.weight_ih_l0" in k:
            rnn_prefix = k[: k.index("rnn.weight_ih_l0")] + "rnn."
            break

    head_prefix = None
    for k in sd:
        if "head.weight" in k:
            head_prefix = k[: k.index("head.weight")] + "head."
            break

    rnn_sd = {
        "weight_ih_l0": sd[f"{rnn_prefix}weight_ih_l0"],
        "weight_hh_l0": sd[f"{rnn_prefix}weight_hh_l0"],
        "bias_ih_l0": sd[f"{rnn_prefix}bias_ih_l0"],
        "bias_hh_l0": sd[f"{rnn_prefix}bias_hh_l0"],
    }
    rnn.load_state_dict(rnn_sd)

    head_sd = {
        "weight": sd[f"{head_prefix}weight"],
        "bias": sd[f"{head_prefix}bias"],
    }
    head.load_state_dict(head_sd)

    return rnn, head


@torch.no_grad()
def pytorch_forward(rnn, head, x_3d):
    """Run the PyTorch model: (B, K, F) → (B, out_dim)."""
    if isinstance(rnn, nn.GRU) or isinstance(rnn, nn.RNN):
        _, h = rnn(x_3d)
    else:
        _, (h, _) = rnn(x_3d)
    return head(h[-1]).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Validate Keras unrolled model against PyTorch"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights-dir", default="firmware/weights/")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--rnn-type", default="gru",
                        choices=["gru", "rnn", "rnn_relu"])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for pass/fail")
    args = parser.parse_args()

    # Load saved weights metadata
    from extract_weights import load_weights
    w = load_weights(args.weights_dir)
    feat_dim, hidden, out_dim = w["feat_dim"], w["hidden"], w["out_dim"]

    print(f"Validation config:")
    print(f"  rnn_type   = {args.rnn_type}")
    print(f"  feat_dim   = {feat_dim}, hidden = {hidden}, out_dim = {out_dim}")
    print(f"  seq_len    = {args.seq_len}")
    print(f"  n_samples  = {args.n_samples}")
    print(f"  atol       = {args.atol}")

    # Build PyTorch model
    print("\nLoading PyTorch model...")
    rnn, head = build_pytorch_model(
        args.checkpoint, args.rnn_type, feat_dim, hidden, out_dim
    )

    # Build Keras model
    print("Building Keras unrolled model...")
    if args.rnn_type == "gru":
        from unroll_gru import build_unrolled_gru_keras
        keras_model = build_unrolled_gru_keras(
            w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
            w["W_head"], w["b_head"],
            args.seq_len, feat_dim, hidden,
        )
    else:
        from unroll_rnn import build_unrolled_rnn_keras
        act = "relu" if args.rnn_type == "rnn_relu" else "tanh"
        keras_model = build_unrolled_rnn_keras(
            w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
            w["W_head"], w["b_head"],
            args.seq_len, feat_dim, hidden, activation=act,
        )

    # Generate random test data
    print(f"\nRunning {args.n_samples} random test cases...")
    np.random.seed(42)
    x_np = np.random.randn(args.n_samples, args.seq_len, feat_dim).astype(np.float32)

    # PyTorch forward
    x_pt = torch.from_numpy(x_np)
    y_pt = pytorch_forward(rnn, head, x_pt)

    # Keras forward (flattened input)
    x_flat = x_np.reshape(args.n_samples, -1)
    y_keras = keras_model.predict(x_flat, verbose=0)

    # Compare
    abs_err = np.abs(y_pt - y_keras)
    max_err = abs_err.max()
    mean_err = abs_err.mean()
    median_err = np.median(abs_err)

    print(f"\nResults:")
    print(f"  Max abs error   : {max_err:.2e}")
    print(f"  Mean abs error  : {mean_err:.2e}")
    print(f"  Median abs error: {median_err:.2e}")

    # Per-output-dim breakdown
    for d in range(out_dim):
        d_err = abs_err[:, d]
        print(f"  Output dim {d}: max={d_err.max():.2e}, mean={d_err.mean():.2e}")

    if max_err < args.atol:
        print(f"\n  PASS — max error {max_err:.2e} < atol {args.atol}")
        sys.exit(0)
    else:
        print(f"\n  FAIL — max error {max_err:.2e} >= atol {args.atol}")
        print("  This may indicate a weight extraction or unrolling bug.")
        sys.exit(1)


if __name__ == "__main__":
    main()
