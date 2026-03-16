"""
Step 2: Build unrolled Keras model from extracted weights and convert to HLS C++ via hls4ml.

Reads numpy weight files from Step 1 (extract_weights.py), builds the
unrolled Keras model, then runs hls4ml conversion and optional Vivado synthesis.

Usage:
    python firmware/convert_hls.py \
        --weights-dir firmware/weights/ \
        --seq-len 10 \
        --output-dir firmware/hls_output/ \
        --precision "ap_fixed<16,6>" \
        --reuse-factor 1 \
        --clock-period 5

    # With Vivado synthesis:
    python firmware/convert_hls.py \
        --weights-dir firmware/weights/ \
        --seq-len 10 \
        --output-dir firmware/hls_output/ \
        --synth
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import yaml


def load_config(config_path: str = "firmware/config.yaml") -> dict:
    """Load default config, returns empty dict if file missing."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def build_keras_model(weights_dir: str, seq_len: int):
    """Load weights and build the appropriate unrolled Keras model."""
    from extract_weights import load_weights

    w = load_weights(weights_dir)
    rnn_type = w["rnn_type"]
    feat_dim = w["feat_dim"]
    hidden = w["hidden"]

    print(f"Building unrolled Keras model:")
    print(f"  rnn_type = {rnn_type}")
    print(f"  feat_dim = {feat_dim}, hidden = {hidden}, out_dim = {w['out_dim']}")
    print(f"  seq_len  = {seq_len}")
    print(f"  Input shape: ({seq_len * feat_dim},)")

    if rnn_type == "gru":
        from unroll_gru import build_unrolled_gru_keras
        model = build_unrolled_gru_keras(
            w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
            w["W_head"], w["b_head"],
            seq_len, feat_dim, hidden,
        )
    elif rnn_type in ("rnn", "rnn_relu"):
        from unroll_rnn import build_unrolled_rnn_keras
        activation = "relu" if rnn_type == "rnn_relu" else "tanh"
        model = build_unrolled_rnn_keras(
            w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
            w["W_head"], w["b_head"],
            seq_len, feat_dim, hidden, activation=activation,
        )
    else:
        raise ValueError(
            f"Unsupported rnn_type={rnn_type!r}. "
            f"Use gru, rnn, or rnn_relu. (LSTM unrolling not implemented.)"
        )

    return model


def convert_to_hls(
    keras_model,
    output_dir: str,
    clock_period: int = 5,
    fpga_part: str = "xcu250-figd2104-2L-e",
    reuse_factor: int = 1,
    precision: str = "ap_fixed<16,6>",
    io_type: str = "io_parallel",
    backend: str = "Vivado",
):
    """Convert Keras model to HLS C++ project via hls4ml."""
    try:
        import hls4ml
    except ImportError:
        raise ImportError(
            "hls4ml not installed. Run:\n"
            "  conda run -n adaptive pip install hls4ml[profiling]"
        )

    # Auto-configure from Keras model
    config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")

    config["Model"]["Precision"] = precision
    config["Model"]["ReuseFactor"] = reuse_factor

    # Per-layer precision
    for layer_name in config["LayerName"]:
        config["LayerName"][layer_name]["Precision"] = {
            "weight": precision,
            "bias": precision,
            "result": precision,
        }
        config["LayerName"][layer_name]["ReuseFactor"] = reuse_factor

    print(f"\nhls4ml configuration:")
    print(f"  Backend       : {backend}")
    print(f"  FPGA part     : {fpga_part}")
    print(f"  Clock period  : {clock_period} ns ({1000 / clock_period:.0f} MHz)")
    print(f"  Precision     : {precision}")
    print(f"  Reuse factor  : {reuse_factor}")
    print(f"  I/O type      : {io_type}")

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        clock_period=clock_period,
        fpga_part=fpga_part,
        io_type=io_type,
    )

    hls_model.compile()
    print(f"\nHLS project compiled to: {output_dir}/")

    # Numerical validation: Keras vs HLS (fixed-point)
    print("\nNumerical validation (Keras float32 vs HLS fixed-point)...")
    n_test = 200
    x_test = np.random.randn(n_test, keras_model.input_shape[1]).astype(np.float32)
    y_keras = keras_model.predict(x_test, verbose=0)
    y_hls = hls_model.predict(x_test)

    abs_err = np.abs(y_keras - y_hls)
    print(f"  Samples  : {n_test}")
    print(f"  Max error: {abs_err.max():.6f}")
    print(f"  Mean error: {abs_err.mean():.6f}")
    print(f"  Median error: {np.median(abs_err):.6f}")

    if abs_err.max() > 0.5:
        print("  WARNING: Large max error — consider wider precision (e.g. ap_fixed<18,8>)")
    elif abs_err.max() > 0.1:
        print("  NOTE: Moderate error — acceptable for trigger but monitor carefully")
    else:
        print("  OK: Fixed-point approximation is tight")

    return hls_model


def run_synthesis(hls_model):
    """Run Vivado HLS C-synthesis and print resource/latency report."""
    print("\nRunning Vivado HLS C-synthesis (this may take several minutes)...")
    try:
        report = hls_model.build(csim=False, synth=True, cosim=False)
        print("\n" + "=" * 50)
        print("  Vivado HLS Synthesis Report")
        print("=" * 50)
        for key in ["EstimatedClockPeriod", "WorstLatency", "BestLatency",
                     "IntervalMin", "IntervalMax",
                     "DSP48E", "BRAM_18K", "LUT", "FF"]:
            val = report.get(key, "N/A")
            print(f"  {key:25s}: {val}")
        print("=" * 50)
        return report
    except Exception as e:
        print(f"\nVivado synthesis failed: {e}")
        print("The HLS C++ source is still available in the output directory.")
        print("You can run synthesis manually with:")
        print("  cd <output-dir> && vivado_hls -f build_prj.tcl")
        return None


def main():
    defaults = load_config()

    parser = argparse.ArgumentParser(
        description="Convert extracted weights → Keras → hls4ml HLS C++"
    )
    parser.add_argument("--weights-dir", default="firmware/weights/",
                        help="Directory with .npy weight files from extract_weights.py")
    parser.add_argument("--seq-len", type=int,
                        default=defaults.get("seq_len", 10))
    parser.add_argument("--output-dir", default="firmware/hls_output/")
    parser.add_argument("--precision",
                        default=defaults.get("precision", "ap_fixed<16,6>"))
    parser.add_argument("--reuse-factor", type=int,
                        default=defaults.get("reuse_factor", 1))
    parser.add_argument("--clock-period", type=int,
                        default=defaults.get("clock_period", 5))
    parser.add_argument("--fpga-part",
                        default=defaults.get("fpga_part", "xcu250-figd2104-2L-e"))
    parser.add_argument("--io-type",
                        default=defaults.get("io_type", "io_parallel"))
    parser.add_argument("--backend",
                        default=defaults.get("backend", "Vivado"))
    parser.add_argument("--synth", action="store_true",
                        help="Run Vivado C-synthesis after conversion")
    parser.add_argument("--save-keras", action="store_true",
                        help="Save intermediate Keras model (.keras)")
    args = parser.parse_args()

    # Build Keras model from saved weights
    keras_model = build_keras_model(args.weights_dir, args.seq_len)
    keras_model.summary()

    # Optionally save Keras model
    if args.save_keras:
        os.makedirs(args.output_dir, exist_ok=True)
        keras_path = os.path.join(args.output_dir, "unrolled_model.keras")
        keras_model.save(keras_path)
        print(f"Keras model saved: {keras_path}")

    # Convert to HLS
    hls_model = convert_to_hls(
        keras_model,
        output_dir=args.output_dir,
        clock_period=args.clock_period,
        fpga_part=args.fpga_part,
        reuse_factor=args.reuse_factor,
        precision=args.precision,
        io_type=args.io_type,
        backend=args.backend,
    )

    # Optional Vivado synthesis
    if args.synth:
        run_synthesis(hls_model)

    print(f"\nDone! HLS C++ project: {args.output_dir}/")


if __name__ == "__main__":
    main()
