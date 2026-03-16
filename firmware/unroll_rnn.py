"""
Unroll a simple Elman RNN into an equivalent Keras functional model of Dense layers.

Each of K timesteps becomes 2 Dense layers (input + hidden) plus activation.
Supports both tanh (rnn) and ReLU (rnn_relu) nonlinearities.

Elman RNN equation:
    h_t = activation(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
"""

from __future__ import annotations
import numpy as np


def build_unrolled_rnn_keras(
    W_ih: np.ndarray,    # (hidden, feat_dim)
    W_hh: np.ndarray,    # (hidden, hidden)
    b_ih: np.ndarray,    # (hidden,)
    b_hh: np.ndarray,    # (hidden,)
    W_head: np.ndarray,  # (out_dim, hidden)
    b_head: np.ndarray,  # (out_dim,)
    seq_len: int,
    feat_dim: int,
    hidden: int,
    activation: str = "tanh",
):
    """
    Build a Keras functional model equivalent to RNN(feat_dim, hidden) + Linear(hidden, out_dim),
    unrolled over `seq_len` timesteps.

    Input shape:  (batch, seq_len * feat_dim)
    Output shape: (batch, out_dim)

    Parameters
    ----------
    activation : str
        "tanh" for standard Elman RNN, "relu" for rnn_relu variant.
        ReLU avoids tanh lookup tables on FPGA — most hardware-friendly.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(seq_len * feat_dim,), name="input_flat")

    # Slice into timesteps
    x_steps = []
    for t in range(seq_len):
        xt = layers.Lambda(
            lambda x, _t=t, _f=feat_dim: x[:, _t * _f : (_t + 1) * _f],
            name=f"slice_t{t}",
        )(inp)
        x_steps.append(xt)

    # h_0 = zeros
    h = layers.Lambda(
        lambda x: tf.zeros((tf.shape(x)[0], hidden)),
        name="h_init",
    )(inp)

    # Unroll RNN timesteps
    for t in range(seq_len):
        xt = x_steps[t]
        h_x = layers.Dense(hidden, use_bias=True, name=f"h_x_t{t}")(xt)
        h_h = layers.Dense(hidden, use_bias=True, name=f"h_h_t{t}")(h)
        h = layers.Add(name=f"h_add_t{t}")([h_x, h_h])
        h = layers.Activation(activation, name=f"h_act_t{t}")(h)

    # Linear head
    out = layers.Dense(W_head.shape[0], use_bias=True, name="head")(h)

    model = keras.Model(inputs=inp, outputs=out, name=f"unrolled_rnn_{activation}")

    # Load PyTorch weights
    for t in range(seq_len):
        model.get_layer(f"h_x_t{t}").set_weights([W_ih.T, b_ih])
        model.get_layer(f"h_h_t{t}").set_weights([W_hh.T, b_hh])

    model.get_layer("head").set_weights([W_head.T, b_head])

    return model
