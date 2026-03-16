"""
Unroll a GRU into an equivalent Keras functional model of Dense layers.

Each of K timesteps becomes 6 Dense layers (3 gates × input/hidden)
plus elementwise operations. The resulting model is fully synthesizable
by hls4ml with no recurrent layer support required.

GRU equations (PyTorch convention):
    r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
"""

from __future__ import annotations
import numpy as np


def build_unrolled_gru_keras(
    W_ih: np.ndarray,    # (3*hidden, feat_dim)
    W_hh: np.ndarray,    # (3*hidden, hidden)
    b_ih: np.ndarray,    # (3*hidden,)
    b_hh: np.ndarray,    # (3*hidden,)
    W_head: np.ndarray,  # (out_dim, hidden)
    b_head: np.ndarray,  # (out_dim,)
    seq_len: int,
    feat_dim: int,
    hidden: int,
):
    """
    Build a Keras functional model equivalent to GRU(feat_dim, hidden) + Linear(hidden, out_dim),
    unrolled over `seq_len` timesteps.

    Input shape:  (batch, seq_len * feat_dim)   — flattened sequence
    Output shape: (batch, out_dim)

    Returns the Keras model with weights already set from the PyTorch arrays.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Partition PyTorch GRU weights: [reset, update, new] each of size `hidden`
    W_ir, W_iz, W_in = np.split(W_ih, 3, axis=0)
    W_hr, W_hz, W_hn = np.split(W_hh, 3, axis=0)
    b_ir, b_iz, b_in = np.split(b_ih, 3)
    b_hr, b_hz, b_hn = np.split(b_hh, 3)

    # Input: flattened (seq_len * feat_dim,)
    inp = keras.Input(shape=(seq_len * feat_dim,), name="input_flat")

    # Slice input into K timesteps
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

    # Unroll GRU timesteps
    for t in range(seq_len):
        xt = x_steps[t]

        # Reset gate: r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        r_x = layers.Dense(hidden, use_bias=True, name=f"r_x_t{t}")(xt)
        r_h = layers.Dense(hidden, use_bias=True, name=f"r_h_t{t}")(h)
        r = layers.Add(name=f"r_add_t{t}")([r_x, r_h])
        r = layers.Activation("sigmoid", name=f"r_sig_t{t}")(r)

        # Update gate: z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        z_x = layers.Dense(hidden, use_bias=True, name=f"z_x_t{t}")(xt)
        z_h = layers.Dense(hidden, use_bias=True, name=f"z_h_t{t}")(h)
        z = layers.Add(name=f"z_add_t{t}")([z_x, z_h])
        z = layers.Activation("sigmoid", name=f"z_sig_t{t}")(z)

        # New gate: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        n_x = layers.Dense(hidden, use_bias=True, name=f"n_x_t{t}")(xt)
        n_h = layers.Dense(hidden, use_bias=True, name=f"n_h_t{t}")(h)
        rn_h = layers.Multiply(name=f"rn_mul_t{t}")([r, n_h])
        n = layers.Add(name=f"n_add_t{t}")([n_x, rn_h])
        n = layers.Activation("tanh", name=f"n_tanh_t{t}")(n)

        # Hidden state update: h = (1 - z) * n + z * h_prev
        one_minus_z = layers.Lambda(lambda z: 1.0 - z, name=f"omz_t{t}")(z)
        h_new = layers.Multiply(name=f"h_new_t{t}")([one_minus_z, n])
        h_old = layers.Multiply(name=f"h_old_t{t}")([z, h])
        h = layers.Add(name=f"h_upd_t{t}")([h_new, h_old])

    # Linear head
    out = layers.Dense(W_head.shape[0], use_bias=True, name="head")(h)

    model = keras.Model(inputs=inp, outputs=out, name="unrolled_gru")

    # Load PyTorch weights (transposed: PyTorch is (out, in), Keras is (in, out))
    for t in range(seq_len):
        model.get_layer(f"r_x_t{t}").set_weights([W_ir.T, b_ir])
        model.get_layer(f"r_h_t{t}").set_weights([W_hr.T, b_hr])
        model.get_layer(f"z_x_t{t}").set_weights([W_iz.T, b_iz])
        model.get_layer(f"z_h_t{t}").set_weights([W_hz.T, b_hz])
        model.get_layer(f"n_x_t{t}").set_weights([W_in.T, b_in])
        model.get_layer(f"n_h_t{t}").set_weights([W_hn.T, b_hn])

    model.get_layer("head").set_weights([W_head.T, b_head])

    return model
