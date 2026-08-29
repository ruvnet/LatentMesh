# 037. Candle RoPE bug — upstream verification

- **Status**: Investigation complete, no code shipped
- **Date**: 2026-08-28
- **Closes**: the ADR-033 upstream-contribution obligation's three verification steps
  (`docs/adr/033-vendored-dependency-and-upstream-contribution-policy.md`, "The upstream-contribution
  obligation")
- **Scope**: verification and preparation only. No PR opened, no comment posted to any external
  service. A human decides whether/how to act on this.

## TL;DR

The bug is real, still present on `candle` main, and **already has an open, unreviewed upstream
PR fixing exactly this file** ([huggingface/candle#3520](https://github.com/huggingface/candle/pull/3520),
opened 2026-05-07, last activity 2026-05-11, zero maintainer reviews as of this writing). It is not
qwen2-specific — the same pre-matmul `.to_dtype(dtype)` pattern is confirmed present in at least
ten other model files. HuggingFace `transformers`' own reference `Qwen2RotaryEmbedding` explicitly
forces F32 for this computation with a comment reading `# Force float32`, so this is unambiguously
a bug candle would recognize as a bug, not a defensible perf tradeoff. **Recommendation: do not
submit a new PR — a correct, well-evidenced one already exists for the exact file. Track PR #3520's
outcome instead.**

## Step 1 — does the bug still exist in candle main?

**Yes, confirmed live.** Cloned `huggingface/candle` at commit `81f247a8985e0b5b6c7c7c5b35c07dc685e005e9`
(2026-08-23, `candle-transformers` version `0.11.0` — two minor versions ahead of the `0.9.2` this
repository vendors). Read `candle-transformers/src/models/qwen2.rs`, `RotaryEmbedding::new`:

```rust
// candle-transformers/src/models/qwen2.rs:54-63 (candle main, 81f247a)
let inv_freq_len = inv_freq.len();
let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;   // line 55 — BUG
let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
    .to_dtype(dtype)?                                                                  // line 57 — BUG
    .reshape((max_seq_len, 1))?;
let freqs = t.matmul(&inv_freq)?;
Ok(Self {
    sin: freqs.sin()?,   // line 61 — no cast needed, already in `dtype`
    cos: freqs.cos()?,   // line 62 — no cast needed, already in `dtype`
})
```

This is byte-for-byte the same buggy pattern as the vendored `0.9.2` copy this repository fixed
(`crates/latentmesh-runtime/src/models/qwen2_a.rs` deviation 4). **Our vendored fix is still
needed** for this repository's own `qwen2_a.rs`/`qwen2_b.rs` — nothing about the upgrade from
`0.9.2` → `0.11.0`'s qwen2.rs changed this code path.

### Is the pattern generic beyond qwen2?

Yes — materially raising the value of a fix. Grepped every model file in
`candle-transformers/src/models/` for the same "cast position index / inv_freq to `dtype` before
the matmul" shape:

| File | Pattern | Status |
|---|---|---|
| `qwen2.rs` | `t`/`inv_freq` → `dtype` before matmul | **buggy** |
| `qwen2_moe.rs` | same | **buggy** |
| `gemma.rs` | same | **buggy** |
| `gemma2.rs` | same | **buggy** |
| `gemma3.rs` | same | **buggy** |
| `stable_lm.rs` | same | **buggy** |
| `yi.rs` | same | **buggy** |
| `starcoder2.rs` | same | **buggy** |
| `mixtral.rs` | same | **buggy** |
| `falcon.rs` | same (via cached `inv_freq.to_dtype(dtype)`) | **buggy** |
| `phi3.rs` | same (both short- and long-rope branches) | **buggy** |
| `llama.rs` | `t`/`inv_freq` → `DType::F32`, cast only final sin/cos | **fixed** |
| `mistral.rs` | `t`/`inv_freq` → `DType::F32`, cast only final sin/cos | **fixed** |
| `qwen3.rs` | `t`/`inv_freq` → `DType::F32`, cast only final sin/cos | **fixed** |

`llama.rs`, `mistral.rs`, and `qwen3.rs` already use the correct F32-table pattern — this repo's
ADR-033 evidence base already noted `llama.rs` does this; the newer `qwen3.rs` apparently inherited
the fixed pattern when it was added, while `qwen2.rs` (older, unmaintained on this specific point)
did not. PR #3520's own author independently grepped and additionally found `olmo.rs`, `olmo2.rs`,
`chatglm.rs`, `recurrent_gemma.rs`, and `glm4.rs` with the same buggy shape — not independently
re-verified here, but consistent with this table's finding that the split in the codebase is real
and roughly 50/50 old-vs-new model files.

### Existing upstream activity — an unmerged PR already fixes this exact file

`gh search issues --repo huggingface/candle "qwen2 rope" --include-prs` surfaced
**[PR #3520 — "qwen2: build RoPE cos/sin tables in fp32 (fixes long-context divergence)"](https://github.com/huggingface/candle/pull/3520)**,
opened 2026-05-07 by `toddwbucy`, **still `OPEN`, zero reviews**, last comment 2026-05-11. It is,
line for line, the identical fix this repository independently derived: keep `inv_freq`/`t` in F32
through the matmul and trig, cast only the resulting `sin`/`cos` to `dtype`. Its diff is 17
additions / 4 deletions against the same five lines identified above.

Its evidence is stronger than a synthetic aliasing demo — a 1000-sample stratified cosine-similarity
harness against `transformers` + `flash_attn 2.8.3` on real Qwen2.5-VL (Jina Embeddings V4)
inference:

| Stratum | n | P95 cosine before | P95 cosine after |
|---|---|---:|---:|
| retrieval.query (~50–500B tokens) | 200 | 0.9996 | 0.9996 |
| retrieval.passage (~200–2kB) | 200 | 0.9970 | 0.9997 |
| code | 200 | 0.9959 | 0.9990 |
| **long-context (~16k tokens)** | 100 | **0.7705** | **0.9999** |

Long-context min cosine 0.7032 → 0.9994; max-abs diff at post-RoPE Q on a 15962-token input:
38.0 → 0.25. The PR's discovery narrative (linked issue #3515, a six-step layer-by-layer
bisection isolating the divergence to `apply_rotary_emb_qkv`) independently corroborates this
repository's own S1a diagnosis via a completely different route (embedding-similarity regression
vs. this repository's greedy-decoding token-duplication observation) — two unrelated bug hunts
converging on the same five lines is strong confirmation this is a real, not theoretical, defect.

The PR sat unreviewed for over three months against an actively-maintained repo (5 merges in the
week before this check, most reviewed/merged within 1-3 days) — worth noting as a maintenance-risk
signal, not something this investigation can resolve.

**No other open issue or PR reporting this independently was found** beyond #3520 and its
prerequisite issue #3515.

## Step 2 — minimal reproducer

Built as a standalone Rust binary with **no candle dependency at all** — just the `half` crate's
`bf16` type, matching Qwen2.5-1.5B's actual `head_dim=128`, `rope_theta=1_000_000.0`. It reproduces
both computation paths (buggy: cast-then-multiply; fixed: multiply-then-cast) and measures where
they diverge. CPU-only, compiles and runs in ~6s.

`/tmp/claude-1000/.../scratchpad/candle-upstream/rope-repro/src/main.rs` (`Cargo.toml` depends only
on `half = "2.4"`):

```rust
use half::bf16;

const HEAD_DIM: usize = 128; // Qwen2.5-1.5B: hidden_size 1536 / 12 heads
const ROPE_THETA: f64 = 1_000_000.0; // Qwen2.5 config.json rope_theta
const MAX_SEQ_LEN: u32 = 8192;

fn inv_freq_table(dim: usize, theta: f64) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect()
}

fn to_bf16_f32(x: f32) -> f32 { bf16::from_f32(x).to_f32() }

fn main() {
    let inv_freq = inv_freq_table(HEAD_DIM, ROPE_THETA);

    // Part 1: does bf16 rounding of the raw position index collide?
    let mut seen = std::collections::HashMap::new();
    let mut first_collision = None;
    for pos in 0..MAX_SEQ_LEN {
        let bits = to_bf16_f32(pos as f32).to_bits();
        if let Some(&earlier) = seen.get(&bits) {
            if first_collision.is_none() { first_collision = Some((earlier, pos)); }
        } else { seen.insert(bits, pos); }
    }
    // ... (unique-value counts per threshold; full listing in the reproducer file)

    // Part 2/3: dot((cos_true,sin_true), (cos_buggy,sin_buggy)) at dim 0
    // (inv_freq[0] == 1.0, so angle ≈ raw position) — see reproducer for
    // the full sampled-position table and the aggregate P95 metric.
}
```

Full source is at
`/tmp/claude-1000/-home-ruvultra-projects-LatentMesh-1/5e5545f1-6360-406b-b384-6806f2e012c9/scratchpad/candle-upstream/rope-repro/src/main.rs`
(scratchpad, not committed).

### Numeric evidence (actual run output, `cargo run --release`)

```
first collision: position 256 and position 257 both round to bf16 value 256
total collisions in [0,8192): 7295 distinct positions alias onto an earlier one

  positions [0,  256):   256 unique bf16 values (0 lost)
  positions [0,  512):   385 unique bf16 values (127 lost)
  positions [0, 1024):   513 unique bf16 values (511 lost)
  positions [0, 2048):   641 unique bf16 values (1407 lost)
  positions [0, 4096):   769 unique bf16 values (3327 lost)
  positions [0, 8192):   897 unique bf16 values (7295 lost)

=== RoPE angle divergence at dim 0 (inv_freq=1.0) ===
     pos       cos_true      cos_buggy      sin_buggy dot(true,buggy)
       0       1.000000       1.000000       0.000000     1.0000
     100       0.863281       0.863281      -0.507812     1.0031
     255      -0.863281      -0.863281      -0.507812     1.0031
     256      -0.039795      -0.039795      -1.000000     1.0016
     257       0.820312      -0.039795      -1.000000     0.5416
    4096       0.804688       0.804688      -0.593750     1.0001
    8000       0.065430       0.065430       0.996094     0.9965
    8191      -0.644531       0.292969      -0.957031     0.5402

=== aggregate rotation-vector cosine similarity, true vs buggy ===
  [0, 256)    -- below alias threshold: min=1.0000 worst=1.0000  (n=256)
  [256, 2048) -- past alias threshold:  min=-0.9900 worst=-0.9900 (n=1792)
  [2048, 8192)-- deep long-context:     min=-0.9900 worst=-0.9900 (n=6144)
```

**Exact collision point**: positions 256 and 257 are the first pair that alias — precisely matching
the "~256" figure both this repository's ADR-033 and upstream PR #3520 independently arrived at
(bf16's 8-bit mantissa gives exactly 256 exactly-representable consecutive small integers before
the first rounding gap). Past that point, aliasing gets rapidly worse: only 897 of the 8192
positions below 8192 survive bf16 rounding as distinct values — an 89% collision rate at that
range. **Magnitude of the resulting angle error**: the rotation-vector cosine similarity
(`dot(cos_true,sin_true, cos_buggy,sin_buggy)` — 1.0 means identical rotation, -1.0 means the exact
opposite rotation) collapses from a clean 1.0 below position 256 to as low as **-0.99** past it —
i.e. some aliased positions receive not just "the wrong angle" but a rotation applied in
*essentially the opposite direction*. This matches both this repository's own observed symptom
(token duplication past ~256) and PR #3520's measured P95 cosine collapse (0.9996 → 0.7705) at
long context, via a completely independent method (pure position-index arithmetic here vs.
full-model output comparison there).

## Step 3 — isolated patch

Against **current candle main** (`81f247a8985e0b5b6c7c7c5b35c07dc685e005e9`), touching only the
five buggy lines in `qwen2.rs`'s unmodified, unsplit `RotaryEmbedding::new` — none of this
repository's other three (inert) deviations from `qwen2_a.rs`'s ledger (with_tracing removal,
inlined `repeat_kv`, `pub(super)` visibility) apply, since this patch targets the real upstream
file using its own `with_tracing` types as-is:

```diff
diff --git a/candle-transformers/src/models/qwen2.rs b/candle-transformers/src/models/qwen2.rs
index 8a29646..f107c14 100644
--- a/candle-transformers/src/models/qwen2.rs
+++ b/candle-transformers/src/models/qwen2.rs
@@ -52,14 +52,20 @@ impl RotaryEmbedding {
             .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
             .collect();
         let inv_freq_len = inv_freq.len();
-        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
+        // Build the table in F32 and cast only the final sin/cos to `dtype`.
+        // BF16 has an 8-bit mantissa: integer position indices above ~256
+        // round to a shared representable value, so casting `t`/`inv_freq`
+        // to `dtype` *before* the outer product aliases distinct positions
+        // onto identical rotary angles once max_seq_len exceeds ~256. See
+        // llama.rs / mistral.rs / qwen3.rs, which already use this pattern.
+        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
         let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
-            .to_dtype(dtype)?
+            .to_dtype(DType::F32)?
             .reshape((max_seq_len, 1))?;
         let freqs = t.matmul(&inv_freq)?;
         Ok(Self {
-            sin: freqs.sin()?,
-            cos: freqs.cos()?,
+            sin: freqs.sin()?.to_dtype(dtype)?,
+            cos: freqs.cos()?.to_dtype(dtype)?,
         })
     }
```

**Verified this patch compiles clean** against candle main: `cargo check -p candle-transformers
--no-default-features` (CPU-only, no CUDA feature enabled) finished with zero errors/warnings in
~24s. `DType` is already imported at the top of `qwen2.rs` (`use candle_core::{DType, Device,
Module, Result, Tensor}` — same import list this repository's vendored copy uses), so no import
changes are needed. Patch file saved at
`/tmp/claude-1000/-home-ruvultra-projects-LatentMesh-1/5e5545f1-6360-406b-b384-6806f2e012c9/scratchpad/candle-upstream/isolated-rope-fix.patch`.

This patch is **functionally identical in shape** to PR #3520's diff (both keep `inv_freq`/`t` in
F32 through the matmul+trig, cast only the final `sin`/`cos`) — independently arrived at from a
different starting bug (this repo's greedy-decoding degeneration vs. PR #3520's embedding-cosine
regression), which is itself corroborating evidence the fix is correct and the bug is real.

## HF reference comparison — is this a defensible perf tradeoff, or a bug?

**Unambiguously a bug**, not a deliberate tradeoff. HuggingFace `transformers`' own
`Qwen2RotaryEmbedding.forward` (`src/transformers/models/qwen2/modeling_qwen2.py`, current main)
explicitly disables autocast and forces float32 for exactly this computation, with a comment
saying so:

```python
with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling

return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

Both operands (`inv_freq_expanded`, `position_ids_expanded`) are explicitly `.float()`-cast, the
matmul and trig run inside an `enabled=False` autocast context specifically to *prevent* the
ambient BF16 autocast policy from touching this computation, and only the final `cos`/`sin` tensors
are cast back to the model's working dtype at the very last line. This is the reference
implementation candle's `qwen2.rs` is supposed to match — candle's buggy version does the opposite
of what the reference deliberately guards against. There is no perf rationale on record anywhere
(not in `transformers`, not in candle's own `llama.rs`/`mistral.rs`/`qwen3.rs`, not in PR #3520's
discussion) for skipping the F32 tables; `llama.rs`/`mistral.rs`/`qwen3.rs` already pay this exact
cost (a `(max_seq_len, dim/2)` table built once at model-load time, not per-forward-pass) without
comment or caveat, so candle would almost certainly accept this as a straightforward correctness
fix rather than debate a tradeoff.

## Boundary table — verified vs. asserted (mirrors ADR-033's table, now resolved)

| Claim | Status (ADR-033) | Status (this doc) |
|---|---|---|
| Bug present in `candle` main / a release after 0.9.2 | UNVERIFIED | **Verified — present, `candle-transformers` 0.11.0, commit 81f247a** |
| Minimal reproducer isolated from this repository's harness | Not built | **Built — standalone, no candle dep, CPU-only, `half`-crate-only** |
| Upstream contribution submitted | Not done | **Not done — and should not be: PR #3520 already covers this file** |
| Bug generic beyond qwen2 | Not assessed | **Confirmed generic — 10+ model files share the pattern; 3 already fixed** |
| Reference implementation comparison | Not done | **Done — HF `transformers` forces F32 explicitly, "Force float32" comment** |

## Recommendation

**Don't contribute a new PR.** [PR #3520](https://github.com/huggingface/candle/pull/3520) already
proposes the identical fix, against the identical file, with stronger evidence (real-model cosine
regression at 16k-token context, not just a synthetic aliasing demo) than this repository could add
without contributing to noise on an already-open PR. Opening a second, competing PR for the same
five lines would not help — it would fragment reviewer attention.

**What's actually useful here, for a human to decide**:
1. **Track PR #3520**, not file a new one. If/when it merges, this repository's next `candle`
   upgrade can drop deviation 4 from `qwen2_a.rs`'s ledger — after re-running the bit-parity
   mechanics gates per ADR-033's maintenance-liability section, since "the same fix" is not
   verified numerically identical to "our fix" without a golden-vector check.
2. If a human wants to help move #3520 forward, the reproducer and generic-pattern findings in this
   document (particularly the confirmed list of 10+ other affected model files, which the PR author
   flagged but didn't verify against this many files themselves) would strengthen a **comment on
   the existing PR** — not a competing submission. That's explicitly not done here per this task's
   instructions; it's a decision for the human, not this investigation.
3. Given the PR has sat unreviewed for 3.5 months on an actively-merging repo, there's a real chance
   it stalls or goes stale (rebasing risk as `qwen2.rs` receives unrelated changes). Worth
   re-checking its status at the next `candle` upgrade even if no one comments on it now.
4. ADR-033's step 1 ("verify the bug persists upstream") is now answered definitively: **yes, it
   persists**, and this repository's local deviation 4 remains necessary until #3520 (or an
   equivalent) merges and this repository upgrades past it.
