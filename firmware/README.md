# FPGA Firmware Export: StateEncoder (GRU + Linear) → HLS C++

Convert the trained PyTorch `StateEncoder` (from `RL/state_encoder.py`) to
synthesizable HLS C++ for Xilinx FPGAs via **hls4ml**, without modifying any
files in the `RL/` directory.

## Architecture being converted

```
Input (B, K=10, F=2)
    │
    ▼
┌────────────────────┐
│  GRU (hidden=32)   │  ← 3 gates × (W_ih, W_hh, b) per timestep
│  K=10 timesteps    │
└────────┬───────────┘
         │ h[-1] → (B, 32)
         ▼
┌────────────────────┐
│  Linear (32 → n_a) │  ← single Dense layer
└────────┬───────────┘
         │
         ▼
    logits (B, n_actions)
```

## Conversion strategy

hls4ml has limited native RNN support. We **unroll** the GRU into K explicit
timesteps of Dense-layer gate operations, producing an equivalent feedforward
Keras model that hls4ml synthesizes natively:

```
PyTorch StateEncoder (.pt checkpoint)
    ↓  extract_weights.py — pulls W_ih, W_hh, b, W_head, b_head
Unrolled Keras model (K × 6 Dense layers for GRU gates + 1 head)
    ↓  convert_hls.py — hls4ml converts to HLS C++
Vivado HLS project (firmware/hls_output/)
    ↓  vivado_hls -f build_prj.tcl  (or --synth flag)
FPGA bitstream
```

## Setup

```bash
# Install hls4ml into the adaptive conda environment
conda run -n adaptive pip install hls4ml[profiling]

# For Vivado synthesis (optional — only needed for resource/latency reports):
# Install Xilinx Vivado HLS 2020.1+ and source settings64.sh
```

## Quick start

```bash
# Step 1: Extract weights from a trained checkpoint
conda run -n adaptive python firmware/extract_weights.py \
    --checkpoint outputs/your_model_dir/model.pt \
    --rnn-type gru \
    --output-dir firmware/weights/

# Step 2: Build unrolled Keras model and convert to HLS
conda run -n adaptive python firmware/convert_hls.py \
    --weights-dir firmware/weights/ \
    --seq-len 10 \
    --output-dir firmware/hls_output/ \
    --precision "ap_fixed<16,6>" \
    --reuse-factor 1 \
    --clock-period 5

# Step 3 (optional): Validate numerical equivalence against PyTorch
conda run -n adaptive python firmware/validate.py \
    --checkpoint outputs/your_model_dir/model.pt \
    --weights-dir firmware/weights/ \
    --seq-len 10 \
    --rnn-type gru

# Step 4 (optional): Run Vivado C-synthesis for resource estimates
conda run -n adaptive python firmware/convert_hls.py \
    --weights-dir firmware/weights/ \
    --seq-len 10 \
    --output-dir firmware/hls_output/ \
    --synth
```

## Supported RNN types

| `--rnn-type` | Gates | FPGA cost   | Notes                          |
|--------------|-------|-------------|--------------------------------|
| `gru`        | 3     | Moderate    | Default — best quality/cost    |
| `rnn`        | 1     | Lowest      | Elman RNN with tanh            |
| `rnn_relu`   | 1     | Lowest      | Elman RNN with ReLU (no LUT)  |

## Key hls4ml parameters

| Parameter        | Default              | Description                              |
|------------------|----------------------|------------------------------------------|
| `--precision`    | `ap_fixed<16,6>`     | Fixed-point: 16 total bits, 6 integer    |
| `--reuse-factor` | `1`                  | 1 = fully parallel (min latency)         |
| `--clock-period` | `5` (ns)             | 5 ns = 200 MHz (CMS L1T typical)         |
| `--fpga-part`    | `xcu250-figd2104-2L-e` | Xilinx Alveo U250                     |
| `--io-type`      | `io_parallel`        | Fully parallel I/O                       |

For CMS Phase-2 L1 Trigger, use `--fpga-part xcvu13p-flga2577-2-e` (VU13P).

## Resource estimate (GRU, hidden=32, K=10, feat=2)

- Parameters: ~4,300
- Unrolled layers: 61 Dense + 10 sigmoid + 10 tanh + elementwise ops
- At reuse_factor=1: ~40–60 DSP48E slices, single-digit clock-cycle latency per timestep
- Well within CMS L1T FPGA budget

## Files

```
firmware/
├── README.md              ← this file
├── extract_weights.py     ← Step 1: PyTorch .pt → numpy weight files
├── convert_hls.py         ← Step 2: weights → Keras → hls4ml → HLS C++
├── validate.py            ← Step 3: numerical equivalence check
├── unroll_gru.py          ← GRU unroll logic (Keras functional model)
├── unroll_rnn.py          ← Simple RNN unroll logic
└── config.yaml            ← default FPGA/synthesis parameters
```
