//! M6 draw state (ADR-047), split out of `run2_m6_probe.rs` for file-size
//! discipline. `m5_draw.rs`'s sibling, with two registered differences.
//!
//! **The primary is `aligned` vs `mismatched`, not `aligned` vs `random`.**
//! That is the whole rung: `random` is wrong on two axes at once — content-free
//! AND off-manifold — so beating it identifies neither factor (ADR-047 §1).
//! `mismatched` is another episode's genuine payload: on-manifold, norm-matched
//! by the same rescale rule, wrong only in content. `random` is still run and
//! still reported, as a control, on every endpoint.
//!
//! **There is one e-process, not two.** ADR-047 §6.1 registered two, one per
//! axis. The MANIFOLD primary was withdrawn with the `aligned_displaced` cell
//! after the §5 manipulation check (coordinator error #24). The CONTENT
//! primary is unchanged, including its power anchor and its n-dependent bar.
//!
//! **No bar field is stored as a verdict.** In M5 a derived bar field was
//! wrong twice, in both directions: first compared against a fixed count valid
//! only at one n, then against a fixed rate that disagreed with the wealth rule
//! in 34 (n, wins) combinations. The wealth rule is the sole authority, so
//! `crossed` IS `crossed_at.is_some()` and the required-wins figure is emitted
//! as a computed diagnostic beside it, never as the verdict.

use crate::common::m6::Quint;

// ---- ADR-036 Decision 1 e-process parameters, frozen ----------------------
/// Betting fraction, `λ = 2θ−1` tuned to θ = 0.65. Never re-parametrised.
pub const LAMBDA: f64 = 0.30;
/// α for the wealth boundary. PASS at `W_i ≥ 1/α`.
pub const E_ALPHA: f64 = 0.05;
/// The registered budget.
pub const N_MAX: usize = 300;
/// ADR-047 §6: below this discordant count the pair is UNINFORMATIVE and the
/// power model is recorded as wrong — a finding about our estimation, not
/// about the apparatus. Never reported as a null.
pub const UNINFORMATIVE_BELOW_N_DISC: usize = 30;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats). Reported.
pub const FUSE_NOOP_TOL: f32 = 1e-6;

/// Discordant wins needed for the wealth process to reach `1/α` at `n`
/// discordant pairs — the smallest `k` solving
/// `(1+λ/2)^k (1−λ/2)^(n−k) ≥ 1/α`.
///
/// **Diagnostic only.** It is computed from the same λ and α the wealth
/// process uses, so it agrees with it by construction rather than by a
/// remembered constant — which is exactly what the two M5 bar-field errors got
/// wrong. `None` means the boundary is unreachable at this `n`.
pub fn wins_needed(n: usize) -> Option<usize> {
    let (up, down) = (1.0 + LAMBDA / 2.0, 1.0 - LAMBDA / 2.0);
    (0..=n).find(|&k| up.powi(k as i32) * down.powi((n - k) as i32) >= 1.0 / E_ALPHA)
}

/// One step of the registered wealth process.
pub struct EStep {
    pub order: usize,
    pub item: usize,
    pub aligned_correct: bool,
    pub mismatched_correct: bool,
    pub discordant: bool,
    pub x: Option<u8>,
    pub wealth: f64,
}

/// The draw's accumulated state. `wealth` starts at 1 and is only ever
/// updated on a discordant item, per the registered rule.
pub struct DrawOutcome {
    pub paired: Vec<Quint>,
    pub rows: Vec<serde_json::Value>,
    pub trajectory: Vec<EStep>,
    pub wealth: f64,
    pub max_wealth: f64,
    pub crossed_at: Option<usize>,
    pub items_drawn: usize,
    pub wins: usize,
    pub losses: usize,
    pub degenerate: usize,
}

impl Default for DrawOutcome {
    fn default() -> Self {
        Self {
            paired: Vec::new(),
            rows: Vec::new(),
            trajectory: Vec::new(),
            wealth: 1.0,
            max_wealth: 1.0,
            crossed_at: None,
            items_drawn: 0,
            wins: 0,
            losses: 0,
            degenerate: 0,
        }
    }
}

impl DrawOutcome {
    /// Record one evaluated item and apply the registered wealth update.
    /// Returns whether the item was discordant **on the registered primary**.
    pub fn push_pair(
        &mut self,
        order: usize,
        item: usize,
        row: serde_json::Value,
        q: Quint,
    ) -> bool {
        let discordant = q.real.0 != q.mism.0;
        let x = discordant.then(|| u8::from(q.real.0));
        if discordant {
            if q.real.0 {
                self.wins += 1;
            } else {
                self.losses += 1;
            }
            self.wealth *= 1.0 + LAMBDA * (f64::from(x.unwrap()) - 0.5);
            self.max_wealth = self.max_wealth.max(self.wealth);
        }
        self.trajectory.push(EStep {
            order,
            item,
            aligned_correct: q.real.0,
            mismatched_correct: q.mism.0,
            discordant,
            x,
            wealth: self.wealth,
        });
        self.rows.push(row);
        self.paired.push(q);
        discordant
    }

    /// A degenerate sender capture yields no pair and therefore no wealth
    /// update — identical in effect to a concordant item — but it still
    /// CONSUMES one of the `N_MAX` budget items. Registered, not chosen after
    /// seeing the data.
    ///
    /// It also leaves the carried-forward `mismatched` payload untouched, so
    /// the next item's control is the last item that actually produced one.
    /// That is the only honest option: there is no payload to carry.
    pub fn push_degenerate(&mut self, order: usize, item: usize) {
        self.degenerate += 1;
        self.trajectory.push(EStep {
            order,
            item,
            aligned_correct: false,
            mismatched_correct: false,
            discordant: false,
            x: None,
            wealth: self.wealth,
        });
        self.rows
            .push(serde_json::json!({"item": item, "skipped": "degenerate sender capture pass"}));
    }

    pub fn n(&self) -> usize {
        self.paired.len()
    }
    pub fn n_disc(&self) -> usize {
        self.wins + self.losses
    }
    pub fn e_pass(&self) -> bool {
        self.crossed_at.is_some()
    }
    pub fn uninformative(&self) -> bool {
        self.n_disc() < UNINFORMATIVE_BELOW_N_DISC
    }
    pub fn count(&self, f: impl Fn(&Quint) -> bool) -> usize {
        self.paired.iter().filter(|q| f(q)).count()
    }
    pub fn mean(&self, f: impl Fn(&Quint) -> f32) -> f32 {
        self.paired.iter().map(f).sum::<f32>() / self.n().max(1) as f32
    }
    /// `(accuracy disagreements, bit-identical NLLs, max |ΔNLL|, pass)` for
    /// the fuse zero-payload no-op diagnostic (ADR-047 §8).
    pub fn noop_stats(&self) -> (usize, usize, f32, bool) {
        let dis = self.count(|q| q.zero.0 != q.base.0);
        let exact = self.count(|q| q.zero.1 == q.base.1);
        let max_d = self
            .paired
            .iter()
            .map(|q| (q.zero.1 - q.base.1).abs())
            .fold(0f32, f32::max);
        (dis, exact, max_d, dis == 0 && max_d <= FUSE_NOOP_TOL)
    }
    pub fn trajectory_json(&self) -> Vec<serde_json::Value> {
        self.trajectory
            .iter()
            .map(|s| {
                serde_json::json!({
                    "order": s.order, "item": s.item,
                    "aligned_correct": s.aligned_correct,
                    "mismatched_correct": s.mismatched_correct,
                    "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
                })
            })
            .collect()
    }
}

/// The console summary the operator reads while the draw finishes. Every
/// number here is also a stored receipt field; this is a view, not a source.
pub fn print_summary(n_max: usize, o: &DrawOutcome, r: &serde_json::Value) {
    let (n, n_disc) = (o.n(), o.n_disc());
    println!(
        "\nM6[content axis / question-tail / fuse / de-pooled]: e-process {} — items drawn {}, \
         W_final {:.4} (max {:.4}, threshold {}), n_disc {n_disc} ({}W/{}L)",
        if o.e_pass() { "PASS" } else { "FAIL" },
        o.items_drawn,
        o.wealth,
        o.max_wealth,
        1.0 / E_ALPHA,
        o.wins,
        o.losses
    );
    match o.crossed_at {
        Some(k) => println!("wealth boundary crossed at item {k} of the stream"),
        None => println!(
            "wealth boundary NOT crossed within N_max = {n_max}; full trajectory committed"
        ),
    }
    println!(
        "bar at the REALISED n_disc = {n_disc}: {} wins needed (the wealth rule is the authority; \
         this is the same λ and α, recomputed, not a remembered constant); uninformative (n_disc \
         < {UNINFORMATIVE_BELOW_N_DISC}) = {}",
        wins_needed(n_disc).map_or("unreachable".to_string(), |k| k.to_string()),
        o.uninformative()
    );
    println!(
        "accuracy: aligned {}/{n} mismatched {}/{n} baseline {}/{n} zerovec {}/{n} random {}/{n}",
        o.count(|q| q.real.0),
        o.count(|q| q.mism.0),
        o.count(|q| q.base.0),
        o.count(|q| q.zero.0),
        o.count(|q| q.rand.0)
    );
    println!(
        "NLL means: aligned {:.4} mismatched {:.4} baseline {:.4} zerovec {:.4} random {:.4}",
        o.mean(|q| q.real.1),
        o.mean(|q| q.mism.1),
        o.mean(|q| q.base.1),
        o.mean(|q| q.zero.1),
        o.mean(|q| q.rand.1)
    );
    for key in [
        "aligned_real_vs_mismatched",
        "aligned_real_vs_random",
        "aligned_real_vs_baseline_uninjected",
        "mismatched_vs_random",
        "mismatched_vs_baseline_uninjected",
        "random_vs_baseline_uninjected",
        "zerovec_injected_vs_baseline_uninjected",
    ] {
        let p = &r["control_vs_control_battery"]["pairs"][key];
        println!(
            "  {key}: acc {}W/{}L, nll {}W/{}L (nll sign p {:.3e})",
            p["accuracy"]["wins"],
            p["accuracy"]["losses"],
            p["nll_lower_is_better"]["wins"],
            p["nll_lower_is_better"]["losses"],
            p["nll_lower_is_better"]["p_one_sided"]
                .as_f64()
                .unwrap_or(1.0)
        );
    }
    println!(
        "PROTOCOL IDENTITY: ADR-036 e-process on adaptation-512. The registered primary is \
         aligned vs MISMATCHED; aligned vs random is reported but is NOT this rung's verdict, and \
         no cross-rung comparison to M5's primary is offered."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bar the receipt reports must be the bar the wealth process
    /// actually enforces. M5 stored two different derived bars and both were
    /// wrong; this pins the derivation against a literal simulation of the
    /// update rule rather than against a table.
    #[test]
    fn wins_needed_agrees_with_the_wealth_rule() {
        for n in 0..=120 {
            let k = wins_needed(n);
            match k {
                None => {
                    // Unreachable: even all-wins must fall short.
                    let w = (1.0 + LAMBDA / 2.0f64).powi(n as i32);
                    assert!(
                        w < 1.0 / E_ALPHA,
                        "n={n} claimed unreachable but all-wins crosses"
                    );
                }
                Some(k) => {
                    let sim = |wins: usize| {
                        let mut w = 1.0f64;
                        for i in 0..n {
                            let x = f64::from(u8::from(i < wins));
                            w *= 1.0 + LAMBDA * (x - 0.5);
                        }
                        w
                    };
                    assert!(sim(k) >= 1.0 / E_ALPHA, "n={n}: {k} wins does not cross");
                    if k > 0 {
                        assert!(
                            sim(k - 1) < 1.0 / E_ALPHA,
                            "n={n}: {} wins already crosses",
                            k - 1
                        );
                    }
                }
            }
        }
        // The M5 anchors, recomputed rather than recalled.
        assert_eq!(wins_needed(65), Some(45));
        assert_eq!(wins_needed(69), Some(48));
        assert_eq!(wins_needed(77), Some(52));
        assert_eq!(wins_needed(67), Some(46));
        // ADR-047 §6's registered expectation band.
        assert_eq!(wins_needed(50), Some(37));
        assert_eq!(wins_needed(60), Some(43));
    }

    /// Only the registered primary drives the wealth process. A `random`
    /// disagreement must move nothing — that was M5's primary, not this one.
    #[test]
    fn only_aligned_vs_mismatched_moves_the_wealth() {
        let q = |real: bool, mism: bool, rand: bool| Quint {
            real: (real, 1.0),
            mism: (mism, 1.0),
            base: (false, 1.0),
            zero: (false, 1.0),
            rand: (rand, 1.0),
        };
        let mut o = DrawOutcome::default();
        // aligned == mismatched, but aligned != random: concordant here.
        assert!(!o.push_pair(1, 10, serde_json::json!({}), q(true, true, false)));
        assert_eq!((o.wins, o.losses, o.wealth), (0, 0, 1.0));
        // aligned beats mismatched: a win.
        assert!(o.push_pair(2, 11, serde_json::json!({}), q(true, false, true)));
        assert_eq!((o.wins, o.losses), (1, 0));
        assert!((o.wealth - 1.15).abs() < 1e-12);
        // mismatched beats aligned: a loss.
        assert!(o.push_pair(3, 12, serde_json::json!({}), q(false, true, false)));
        assert_eq!((o.wins, o.losses), (1, 1));
        assert!((o.wealth - 1.15 * 0.85).abs() < 1e-12);
        assert!((o.max_wealth - 1.15).abs() < 1e-12);
    }
}
