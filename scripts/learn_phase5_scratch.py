"""
Phase 5 — Attention & Transformers: built from scratch in numpy.

Learning path:
  1. Single-head attention — the core formula
  2. WHY the √d_k scaling matters (we break it without it)
  3. Multi-head attention — parallel attention "perspectives"
  4. Positional encoding — how Transformers know word order
  5. One full Transformer encoder block
"""

import numpy as np

np.random.seed(42)


# ============================================================
# PART 1: Single-head scaled dot-product attention
# ============================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtract max for numerical stability.

    WHY subtract max: e^500 overflows float32. Subtracting max
    shifts all values ≤ 0, so e^x stays in [0, 1]. The result
    is mathematically identical — try it:
      softmax([1,2,3]) == softmax([-2,-1,0])  → True
    """
    x = x - x.max(axis=-1, keepdims=True)   # stability trick
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,   # (seq_len, d_k)
    K: np.ndarray,   # (seq_len, d_k)
    V: np.ndarray,   # (seq_len, d_v)
) -> tuple[np.ndarray, np.ndarray]:
    """The entire attention mechanism in 3 lines of math.

    Returns:
        output:  (seq_len, d_v) — the attended values
        weights: (seq_len, seq_len) — attention scores (useful for viz)
    """
    d_k = Q.shape[-1]

    # Step 1: how much does each query match each key?
    # Shape: (seq_len, seq_len) — every word vs every word
    scores = Q @ K.T / np.sqrt(d_k)   # WHY √d_k: see Part 2 below

    # Step 2: turn raw scores into a probability distribution
    weights = softmax(scores)          # rows sum to 1

    # Step 3: weighted sum of values
    output = weights @ V               # (seq_len, d_v)

    return output, weights


# --- Demo: 4 tokens, d_model=8 ---

print("=" * 55)
print("PART 1: Single-head attention")
print("=" * 55)

tokens = ["invoice", "INV-0042", "from", "Acme"]
seq_len = len(tokens)
d_model = 8

# In a real Transformer, these are learned weight matrices.
# Here we initialise them randomly to show the mechanics.
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

# Fake token embeddings — normally from an embedding table
X = np.random.randn(seq_len, d_model)   # (4, 8)

# Project into Q, K, V spaces
Q = X @ W_Q   # (4, 8)
K = X @ W_K
V = X @ W_V

output, weights = scaled_dot_product_attention(Q, K, V)

print("\nAttention weight matrix (rows = query, cols = key):")
print("     " + "  ".join(f"{t:>8}" for t in tokens))
for i, row in enumerate(weights):
    print(f"{tokens[i]:>8}  " + "  ".join(f"{w:8.3f}" for w in row))

print(f"\nOutput shape: {output.shape}  ← same seq_len, now context-aware")


# ============================================================
# PART 2: WHY √d_k scaling — break it without it
# ============================================================

print("\n" + "=" * 55)
print("PART 2: What happens WITHOUT √d_k scaling")
print("=" * 55)

def attention_no_scale(Q, K, V):
    scores = Q @ K.T          # no division
    weights = softmax(scores)
    return weights @ V, weights

# Increase d_k to 64 — realistic size in BERT
d_k_large = 64
Q_large = np.random.randn(seq_len, d_k_large)
K_large = np.random.randn(seq_len, d_k_large)
V_large = np.random.randn(seq_len, d_k_large)

_, weights_scaled   = scaled_dot_product_attention(Q_large, K_large, V_large)
_, weights_unscaled = attention_no_scale(Q_large, K_large, V_large)

print(f"\nWith scaling    — entropy of row 0: {(-weights_scaled[0]   * np.log(weights_scaled[0]   + 1e-9)).sum():.3f}")
print(f"Without scaling — entropy of row 0: {(-weights_unscaled[0] * np.log(weights_unscaled[0] + 1e-9)).sum():.3f}")
print(f"\nWith scaling    — max weight: {weights_scaled[0].max():.3f}")
print(f"Without scaling — max weight: {weights_unscaled[0].max():.3f}")
print("""
WHY this matters:
  QK^T dot products grow with d_k. At d_k=64 the raw scores
  are ~8× larger. Softmax on large values → one entry = ~1.0,
  rest → ~0.0. The model becomes a hard argmax: it attends to
  exactly ONE token and ignores everything else.
  Dividing by √d_k keeps scores in a healthy range → soft,
  informative distribution → richer gradients during training.
""")


# ============================================================
# PART 3: Multi-head attention
# ============================================================

print("=" * 55)
print("PART 3: Multi-head attention")
print("=" * 55)

def multi_head_attention(
    X: np.ndarray,    # (seq_len, d_model)
    W_Q: list,        # list of (d_model, d_k) per head
    W_K: list,
    W_V: list,
    W_O: np.ndarray,  # (h * d_v, d_model) — output projection
) -> np.ndarray:
    """Run h attention heads in parallel, concatenate, project.

    WHY multiple heads: each head learns to attend to a DIFFERENT
    relationship. In our invoice example:
      Head 1 might focus on: entity co-reference (invoice ↔ INV-0042)
      Head 2 might focus on: syntactic role (subject ↔ verb)
      Head 3 might focus on: proximity (adjacent words)
    One head can't do all three at once — different linear projections
    let each head specialise.
    """
    heads = []
    all_weights = []

    for wq, wk, wv in zip(W_Q, W_K, W_V):
        q = X @ wq
        k = X @ wk
        v = X @ wv
        head_out, head_weights = scaled_dot_product_attention(q, k, v)
        heads.append(head_out)
        all_weights.append(head_weights)

    # Concatenate all heads → (seq_len, h * d_v)
    concat = np.concatenate(heads, axis=-1)

    # Final linear projection back to d_model
    output = concat @ W_O

    return output, all_weights


num_heads = 2
d_k = d_model // num_heads    # 4 — split d_model evenly across heads

W_Qs = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_Ks = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_Vs = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_O  = np.random.randn(num_heads * d_k, d_model) * 0.1

mha_output, head_weights = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)

print(f"\nInput shape:  {X.shape}")
print(f"Output shape: {mha_output.shape}  ← same shape, 2 perspectives baked in")
print(f"\nHead 1 max attention: token '{tokens[np.argmax(head_weights[0][0])]}' for query '{tokens[0]}'")
print(f"Head 2 max attention: token '{tokens[np.argmax(head_weights[1][0])]}' for query '{tokens[0]}'")
print("↑ Different heads focus on different tokens — that's the point.\n")


# ============================================================
# PART 4: Positional encoding
# ============================================================

print("=" * 55)
print("PART 4: Positional encoding")
print("=" * 55)

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sine/cosine positional encoding from 'Attention Is All You Need'.

    WHY we need this: attention has NO idea about word order.
    "Acme owes Corp" and "Corp owes Acme" produce identical
    attention scores without positional encoding — the set of
    tokens is the same.

    WHY sine/cosine: the encoding for position p+k can be expressed
    as a LINEAR FUNCTION of the encoding for position p. This means
    the model can learn to attend to "the token 3 positions ahead"
    by learning the right linear transformation. Random or learned
    fixed embeddings can't do this.

    Formula:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)          # (seq_len, 1)
    dims      = np.arange(0, d_model, 2)                    # even indices
    div_term  = np.power(10000.0, dims / d_model)           # (d_model/2,)

    PE[:, 0::2] = np.sin(positions / div_term)   # even dims → sine
    PE[:, 1::2] = np.cos(positions / div_term)   # odd  dims → cosine

    return PE


PE = positional_encoding(seq_len=6, d_model=d_model)
X_with_pos = X + PE[:seq_len]   # add directly to embeddings

print("\nPositional encoding for positions 0-3 (first 4 dims shown):")
for i, tok in enumerate(tokens):
    vals = "  ".join(f"{PE[i, j]:+.3f}" for j in range(4))
    print(f"  pos {i} ({tok:>10}): [{vals} ...]")

print("""
Key property: every position gets a UNIQUE fingerprint.
The model learns to read these fingerprints during training.
""")


# ============================================================
# PART 5: One full Transformer encoder block
# ============================================================

print("=" * 55)
print("PART 5: Full Transformer encoder block")
print("=" * 55)

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalise each token's embedding to mean=0, std=1.

    WHY layer norm (not batch norm): batch norm normalises across
    the batch dimension — at inference with batch_size=1 it breaks.
    Layer norm normalises across the feature dimension — works
    identically at any batch size, any sequence length.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def feed_forward(x: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    """Position-wise FFN: two linear layers with ReLU in between.

    WHY FFN after attention: attention mixes information ACROSS tokens
    (which word looks at which). FFN processes each token INDEPENDENTLY
    — it's where the model stores factual knowledge. Researchers who
    ablate the FFN find the model loses world knowledge but keeps
    its ability to do co-reference and syntax.
    """
    hidden = np.maximum(0, x @ W1 + b1)   # ReLU
    return hidden @ W2 + b2


def transformer_encoder_block(
    X: np.ndarray,
    W_Qs, W_Ks, W_Vs, W_O,
    W1, b1, W2, b2,
) -> np.ndarray:
    """One encoder block = MHA + Add&Norm + FFN + Add&Norm.

    The residual connections (x + sublayer(x)) are the second
    reason Transformers don't suffer from vanishing gradients.
    Gradient flows directly through the + sign, bypassing the
    sublayer entirely if needed — same principle as LSTM's cell
    state highway, but even simpler.
    """
    # Sub-layer 1: multi-head attention
    attn_out, _ = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
    X = layer_norm(X + attn_out)            # residual + norm

    # Sub-layer 2: feed-forward network
    ff_out = feed_forward(X, W1, b1, W2, b2)
    X = layer_norm(X + ff_out)              # residual + norm

    return X


d_ff = 32   # FFN hidden dim — typically 4× d_model in real Transformers
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

X_encoded = transformer_encoder_block(
    X_with_pos, W_Qs, W_Ks, W_Vs, W_O, W1, b1, W2, b2
)

print(f"\nInput shape:          {X_with_pos.shape}")
print(f"Encoder block output: {X_encoded.shape}")
print("""
Shape is preserved. Each token's vector now contains:
  - its own meaning (from embeddings)
  - context from every other token (from attention)
  - positional information (from PE)
  - factual processing (from FFN)

Stack 12 of these blocks → BERT base.
Stack 96 → GPT-3.
""")

print("=" * 55)
print("Summary: what we built from scratch")
print("=" * 55)
print("""
  softmax              — stable probability distribution
  scaled_dot_product   — QK^T / √d_k → weights → weighted V
  multi_head_attention — h parallel attention perspectives
  positional_encoding  — sine/cosine position fingerprints
  layer_norm           — stable training at any batch size
  feed_forward         — per-token knowledge processing
  encoder_block        — the full Transformer encoder unit
""")