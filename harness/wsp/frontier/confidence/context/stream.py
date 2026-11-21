"""Run a batch of conversation turns while keeping state owned by each caller."""
import numpy as np

OUTPUTS = ["logits", "ctx_score", "ctx_anger", "ctx_anger_f05", "ctx_next"]


def step(session, tokens, previous=None):
    """Omit previous at dialogue start. Pass one prior state row per batch row."""
    if "ctx_previous" in tokens:
        raise ValueError("pass previous as its own argument")
    feed = dict(tokens)
    if previous is not None:
        state = np.asarray(previous, dtype=np.float64)
        if state.shape != (feed["input_ids"].shape[0], 8) or not np.isfinite(state).all():
            raise ValueError("expected eight finite coordinates per conversation")
        feed["ctx_previous"] = state
    return session.run(OUTPUTS, feed)
