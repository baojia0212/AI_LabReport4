"""
BSD3513 - Lab Report 4 (Chapter 4: Neural Networks)
App: Sigmoid activation function visualiser + neural network demo (Streamlit)

Run:  streamlit run app_sigmoid.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Sigmoid", layout="wide")

# scikit-learn name of this activation, used inside the MLP hidden layers
SK_ACTIVATION = "logistic"
SHORT = "Sigmoid"


# ---------------------------------------------------------
# Activation function and its derivative
# ---------------------------------------------------------
def activation(x):
    return 1.0 / (1.0 + np.exp(-x))


def activation_derivative(x):
    s = activation(x)
    return s * (1.0 - s)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Curve")
x_input = st.sidebar.slider("Input value x", -6.0, 6.0, 1.0, step=0.25)

st.sidebar.header("Neural Network Demo")
dataset_type = st.sidebar.selectbox("Dataset", ["Moons", "Circles"])
n_samples = st.sidebar.slider("Samples", 100, 1000, 300, step=50)
noise = st.sidebar.slider("Noise", 0.0, 0.4, 0.2, step=0.05)

st.sidebar.subheader("Model Parameters")
hidden_text = st.sidebar.text_input("Hidden layer sizes (e.g. 16,16)", value="16,16")
alpha = st.sidebar.slider("L2 regularisation (alpha)", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
learning_rate = st.sidebar.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
max_iter = st.sidebar.slider("Max iterations (epochs)", 100, 3000, 800, step=100)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

# ---------------------------------------------------------
# Header and formula
# ---------------------------------------------------------
st.title("Sigmoid")
st.write("Sigmoid maps any real number onto a smooth S-shaped curve between 0 and 1, so its output can be read as a probability.")
st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
st.latex(r"f'(x) = f(x)\,\bigl(1 - f(x)\bigr)")

# ---------------------------------------------------------
# Output for the chosen input (shown right below the formula)
# ---------------------------------------------------------
x = np.linspace(-6, 6, 400)
y = activation(x)
dy = activation_derivative(x)
y_in = float(activation(np.array(x_input)))
dy_in = float(activation_derivative(np.array(x_input)))

with st.container(border=True):
    st.subheader("Output for your input")
    m1, m2, m3 = st.columns(3)
    m1.metric("x", f"{x_input:.2f}")
    m2.metric(f"{SHORT}(x)", f"{y_in:.4f}")
    m3.metric("f'(x)", f"{dy_in:.4f}")

    st.subheader("Properties")
    st.markdown(
        """
- Range: (0, 1), so it can be read as a probability
- Smooth and differentiable everywhere
- Maximum gradient is only 0.25 (at x = 0)
- Saturates for large |x|, causing vanishing gradients
- Output is not zero-centred
"""
    )

# ---------------------------------------------------------
# 1. Activation curve
# ---------------------------------------------------------
st.header(f"1. {SHORT} Curve")

_, col_plot, _ = st.columns([1, 3, 1])
with col_plot:
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(x, y, color="#00897B", linewidth=2.2, label=f"{SHORT}(x)")
        ax.plot(x, dy, color="#8E24AA", linestyle="-.", linewidth=1.8, label="Derivative f'(x)")
        ax.scatter([x_input], [y_in], color="#F4511E", edgecolor="black", zorder=5, s=60, label="Your input")
        ylim = (-0.1, 1.1)
        if ylim:
            ax.set_ylim(*ylim)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("output")
        ax.set_title(f"{SHORT} Activation")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
        st.pyplot(fig)

# ---------------------------------------------------------
# 2. Neural network using this activation
# ---------------------------------------------------------
st.header(f"2. Neural Network using {SHORT}")
st.write(
    f"A Multi-Layer Perceptron (MLP) is trained on a toy 2-D dataset. "
    f"Every hidden layer uses the **{SHORT}** activation, so the decision boundary below "
    f"shows what this activation lets the network learn."
)


@st.cache_resource(show_spinner="Training the neural network...")
def train_network(dataset, n, noise_level, hidden, l2, lr, iters, rs):
    """Generate data, scale it and train an MLP with the chosen activation."""
    if dataset == "Moons":
        X, y = make_moons(n_samples=n, noise=noise_level, random_state=rs)
    else:
        X, y = make_circles(n_samples=n, noise=noise_level, factor=0.5, random_state=rs)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=SK_ACTIVATION,
            alpha=l2,
            learning_rate_init=lr,
            max_iter=iters,
            random_state=rs,
        )),
    ])
    model.fit(X, y)
    return model, X, y


hidden_sizes = tuple(int(s.strip()) for s in hidden_text.split(",") if s.strip().isdigit())
if not hidden_sizes:
    hidden_sizes = (16, 16)

model, X, y = train_network(
    dataset_type, n_samples, noise, hidden_sizes, alpha, learning_rate, max_iter, int(seed)
)

# Decision boundary
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300),
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_nn, ax_nn = plt.subplots(figsize=(6, 5))
ax_nn.contourf(xx, yy, Z, cmap=ListedColormap(["#EDE7F6", "#E0F2F1"]), alpha=0.8)
ax_nn.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor="k", cmap=ListedColormap(["#6A1B9A", "#00897B"]))
ax_nn.set_title(f"MLP with {SHORT}")
ax_nn.set_xlabel("x1")
ax_nn.set_ylabel("x2")

_, col_nn, _ = st.columns([1, 2, 1])
with col_nn:
    st.pyplot(fig_nn)