"""Phase 4 — Build an RNN from scratch.

No PyTorch. No libraries. Pure numpy so you SEE every step.

What this teaches:
  1. How hidden state updates work (the core RNN formula)
  2. Why vanishing gradients happen (watch the numbers shrink)
  3. Why LSTM's gates fix it (protected memory highway)

Run: uv run python scripts/learn_phase4_scratch.py
"""
import numpy as np

np.random.seed(42)


# ============================================================================
# STEP 1: Fake word embeddings (reusing Phase 3's concept)
# ============================================================================
# In production, these come from Word2Vec/BERT. Here we hand-craft them.
VOCAB = {
    "the":    np.array([0.1, 0.0, 0.0]),
    "service": np.array([0.0, 0.5, 0.0]),
    "was":    np.array([0.0, 0.0, 0.1]),
    "not":    np.array([-0.8, 0.0, 0.0]),  # strong negative signal
    "good":   np.array([0.7, 0.0, 0.0]),   # strong positive signal
    "at":     np.array([0.0, 0.0, 0.05]),
    "all":    np.array([0.3, 0.0, 0.0]),   # intensifier
    "bad":    np.array([-0.7, 0.0, 0.0]),
    "doctor": np.array([0.0, 0.0, 0.9]),
    "patient": np.array([0.0, 0.0, 0.7]),
}

EMBED_DIM = 3   # each word is a 3D vector
HIDDEN_DIM = 4  # RNN has 4 hidden neurons


# ============================================================================
# STEP 2: The RNN — just two weight matrices and a tanh
# ============================================================================
class SimpleRNN:
    """A vanilla RNN in ~30 lines.

    The formula at each time step:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + bias)

    Where:
        x_t     = current word's embedding (EMBED_DIM,)
        h_{t-1} = previous hidden state (HIDDEN_DIM,)
        W_ih    = input-to-hidden weights (HIDDEN_DIM, EMBED_DIM)
        W_hh    = hidden-to-hidden weights (HIDDEN_DIM, HIDDEN_DIM)
        bias    = bias term (HIDDEN_DIM,)
        h_t     = new hidden state (HIDDEN_DIM,)

    That's the ENTIRE algorithm. Everything else is bookkeeping.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        # Xavier initialization — keeps values from exploding or vanishing
        # WHY Xavier? Random values too large → tanh saturates → gradients die
        # Random values too small → signal disappears → gradients die
        # Xavier picks the sweet spot: stddev = sqrt(2 / (fan_in + fan_out))
        scale_ih = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        self.W_ih = np.random.randn(hidden_dim, input_dim) * scale_ih
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_hh
        self.bias = np.zeros(hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, words: list[str]) -> list[dict]:
        """Process a sentence word by word. Return the trace of hidden states.

        This IS the RNN forward pass. Everything GPT does starts from this.
        """
        h = np.zeros(self.hidden_dim)  # initial hidden state = all zeros
        trace = []

        for i, word in enumerate(words):
            x = VOCAB.get(word, np.zeros(EMBED_DIM))

            # THE CORE FORMULA — this is the entire RNN
            # -----------------------------------------
            # Part 1: What does the current word contribute?
            input_part = self.W_ih @ x        # shape: (hidden_dim,)

            # Part 2: What does the memory contribute?
            hidden_part = self.W_hh @ h       # shape: (hidden_dim,)

            # Part 3: Combine and squish through tanh
            raw = input_part + hidden_part + self.bias
            h = np.tanh(raw)  # squish to [-1, 1]
            # -----------------------------------------

            trace.append({
                "step": i,
                "word": word,
                "input_vec": x.copy(),
                "input_contribution": input_part.copy(),
                "memory_contribution": hidden_part.copy(),
                "raw_sum": raw.copy(),
                "hidden_state": h.copy(),
            })

        return trace


# ============================================================================
# STEP 3: Vanishing gradient demonstration
# ============================================================================
def show_vanishing_gradient() -> None:
    """Show WHY RNNs forget long-range dependencies.

    The gradient flows backward through time. At each step, it gets
    multiplied by W_hh. If the largest eigenvalue of W_hh is < 1,
    the gradient SHRINKS exponentially. After 20 steps, it's near zero.

    This is the vanishing gradient problem.
    """
    print("\n" + "=" * 60)
    print("VANISHING GRADIENT DEMO")
    print("=" * 60)

    # Simulate gradient flowing backward through 20 time steps
    W_hh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.5  # small weights

    gradient = np.ones(HIDDEN_DIM)  # start with gradient = 1
    print(f"\n  Step  0: gradient magnitude = {np.linalg.norm(gradient):.6f}")

    for step in range(1, 21):
        gradient = W_hh.T @ gradient  # multiply by W_hh at each step
        mag = np.linalg.norm(gradient)
        bar = "█" * max(1, int(mag * 20))
        print(f"  Step {step:2d}: gradient magnitude = {mag:.6f}  {bar}")
        if mag < 0.0001:
            print(f"\n  *** Gradient effectively DEAD after {step} steps ***")
            print(f"  The RNN can't learn from words more than {step} steps ago.")
            print(f"  This is why 'the DOCTOR who worked at three")
            print(f"  hospitals for fifteen years TREATED the patient'")
            print(f"  fails — 'doctor' and 'treated' are too far apart.")
            break


# ============================================================================
# STEP 4: LSTM gates — conceptual demo
# ============================================================================
def show_lstm_gates() -> None:
    """Show how LSTM gates solve vanishing gradients.

    The key insight: the cell state C is updated with ADDITION, not multiplication.
    C_t = forget_gate * C_{t-1} + input_gate * candidate

    Even if forget_gate = 1 (keep everything) and input_gate = 0 (add nothing),
    the cell state passes through UNCHANGED. The gradient flows perfectly.

    This is the "highway" that lets information survive across 100+ steps.
    """
    print("\n" + "=" * 60)
    print("LSTM GATES DEMO")
    print("=" * 60)

    sentence = ["The", "doctor", "who", "worked", "at", "many",
                "hospitals", "for", "years", "treated", "the", "patient"]

    cell_state = 0.0   # the protected memory
    hidden = 0.0        # the output

    print(f"\n  Processing: {' '.join(sentence)}")
    print(f"  {'Word':<12} {'Forget':<8} {'Input':<8} {'Cell':<8} {'What happens'}")
    print(f"  {'─'*70}")

    for word in sentence:
        # Simulate gate values (in real LSTM, these are learned)
        if word == "doctor":
            forget, input_gate, candidate = 1.0, 0.9, 0.8
            note = "STORE 'doctor' in cell memory"
        elif word in ("who", "worked", "at", "many", "hospitals", "for", "years"):
            forget, input_gate, candidate = 0.95, 0.1, 0.0
            note = "keep memory, ignore filler"
        elif word == "treated":
            forget, input_gate, candidate = 0.9, 0.3, 0.2
            note = "RECALL 'doctor' — cell still has it!"
        elif word == "patient":
            forget, input_gate, candidate = 0.9, 0.5, 0.6
            note = "connect doctor→treated→patient"
        else:
            forget, input_gate, candidate = 0.9, 0.1, 0.0
            note = "neutral"

        # THE LSTM CELL UPDATE — this is the core innovation
        cell_state = forget * cell_state + input_gate * candidate
        hidden = np.tanh(cell_state) * 0.8  # output gate

        bar = "█" * max(1, int(cell_state * 15))
        print(f"  {word:<12} {forget:<8.2f} {input_gate:<8.2f} {cell_state:<8.3f} {bar} {note}")

    print(f"\n  Final cell state: {cell_state:.3f}")
    print(f"  'doctor' information survived {len(sentence)} steps!")
    print(f"  In vanilla RNN, it would have vanished after ~5 steps.")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: RNN from scratch")
    print("=" * 60)

    rnn = SimpleRNN(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)

    # --- Process "The service was not good at all" ---
    sentence = ["the", "service", "was", "not", "good", "at", "all"]
    print(f"\n  Sentence: \"{' '.join(sentence)}\"")
    print(f"  {'─' * 55}")

    trace = rnn.forward(sentence)

    for step in trace:
        h = step["hidden_state"]
        word = step["word"]
        inp = step["input_contribution"]
        mem = step["memory_contribution"]

        print(f"\n  Step {step['step']}: \"{word}\"")
        print(f"    Input contribution:  [{', '.join(f'{v:+.3f}' for v in inp)}]")
        print(f"    Memory contribution: [{', '.join(f'{v:+.3f}' for v in mem)}]")
        print(f"    New hidden state:    [{', '.join(f'{v:+.3f}' for v in h)}]")
        print(f"    Hidden magnitude:    {np.linalg.norm(h):.3f}")

    # --- Vanishing gradient demo ---
    show_vanishing_gradient()

    # --- LSTM gates demo ---
    show_lstm_gates()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. RNN reads words one by one, updating hidden state each step")
    print("  2. Hidden state IS the memory — it carries context forward")
    print("  3. Vanilla RNN's memory FADES (vanishing gradients)")
    print("  4. LSTM adds a cell state HIGHWAY — memory survives 100+ steps")
    print("  5. That's why LSTM dominated NLP from 1997 to 2017")
    print("=" * 60)