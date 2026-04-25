"""
Phase 7.5 — LoRA from first principles.
SEE the rank concept before touching PEFT.
"""
import numpy as np

# ── WHY RANK MATTERS ──────────────────────────────────────────
print("=" * 60)
print("CONCEPT: What does 'low rank' actually mean?")
print("=" * 60)

# A full weight update for one BERT attention layer
W_full = np.random.randn(768, 768)
print(f"\nFull weight matrix W:  {W_full.shape}")
print(f"Total params to train: {W_full.size:,}")

# LoRA says: approximate W_full with B @ A where r << 768
r = 8  # rank — the bottleneck dimension
B = np.random.randn(768, r)   # tall thin matrix
A = np.random.randn(r, 768)   # short wide matrix

delta_W = B @ A   # same shape as W_full but low-rank
print(f"\nLoRA approximation:")
print(f"  B shape: {B.shape}  →  {B.size:,} params")
print(f"  A shape: {A.shape}  →  {A.size:,} params")
print(f"  Total LoRA params: {B.size + A.size:,}")
print(f"  Reduction: {(1 - (B.size + A.size) / W_full.size) * 100:.1f}%")
print(f"\n  delta_W = B @ A shape: {delta_W.shape}  ✓ same as W")

# ── SCALING FACTOR ────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCEPT: Alpha scaling — how loud is the new knowledge?")
print("=" * 60)

alpha = 16   # standard default
scale = alpha / r   # = 2.0
print(f"\nalpha={alpha}, r={r} → scale = {scale}")
print(f"Forward pass: output = W @ x  +  (B @ A) @ x  *  {scale}")
print("Scale=2.0 means LoRA update is 2x amplified.")
print("Scale=0.5 would mean 'trust base model more'.")
print("Rule of thumb: alpha = 2r gives scale=2. Standard starting point.")

# ── RANK COMPARISON ───────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCEPT: How rank affects expressiveness vs params")
print("=" * 60)

for rank in [1, 4, 8, 16, 64]:
    params = 2 * 768 * rank
    pct = params / W_full.size * 100
    print(f"  r={rank:3d}  → {params:7,} params  ({pct:.2f}% of full)  "
          f"{'← standard' if rank == 8 else ''}")

print("\nr=8 is the sweet spot: expressive enough, tiny enough.")