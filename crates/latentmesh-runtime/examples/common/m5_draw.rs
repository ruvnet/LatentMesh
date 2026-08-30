//! M5 draw state and receipt construction (ADR-045), split out of
//! `run2_m5_probe.rs` purely for file-size discipline.
//!
//! Holds the frozen e-process parameters, the wealth trajectory, the
//! per-condition accounting, and the one large receipt literal. The probe
//! itself keeps the gates and the draw loop, so what a reader must audit
//! against ADR-045 — which factors are held fixed, and in what order the
//! gates fire — stays in one readable file.

use crate::common::m3::Quad;

// ---- ADR-036 Decision 1 / ADR-030 §3.2 e-process parameters, frozen -------
/// Betting fraction, `λ = 2θ−1` tuned to θ = 0.65. Never re-parametrised.
pub const LAMBDA: f64 = 0.30;
/// α for the wealth boundary. PASS at `W_i ≥ 1/α`.
pub const E_ALPHA: f64 = 0.05;
/// The registered budget.
pub const N_MAX: usize = 300;
/// ADR-045's registered crossing bar: ≥ 45 of an expected ~65 discordant wins.
pub const REGISTERED_BAR_WINS: usize = 45;
pub const REGISTERED_BAR_OF: usize = 65;
/// ADR-045: below this discordant count the rung is UNINFORMATIVE and the
/// power model is recorded as wrong — a finding about our estimation, not
/// about the apparatus.
pub const UNINFORMATIVE_BELOW_N_DISC: usize = 30;
/// Tolerance for the fuse zero-payload no-op diagnostic (nats). Reported.
pub const FUSE_NOOP_TOL: f32 = 1e-6;

/// One step of the registered wealth process.
pub struct EStep {
    pub order: usize,
    pub item: usize,
    pub aligned_correct: bool,
    pub random_correct: bool,
    pub discordant: bool,
    pub x: Option<u8>,
    pub wealth: f64,
}

/// The draw's accumulated state. `wealth` starts at 1 and is only ever
/// updated on a discordant item, per the registered rule.
pub struct DrawOutcome {
    pub paired: Vec<Quad>,
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
    /// Returns whether the item was discordant.
    pub fn push_pair(
        &mut self,
        order: usize,
        item: usize,
        row: serde_json::Value,
        q: Quad,
    ) -> bool {
        let discordant = q.real.0 != q.rand.0;
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
            random_correct: q.rand.0,
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
    pub fn push_degenerate(&mut self, order: usize, item: usize) {
        self.degenerate += 1;
        self.trajectory.push(EStep {
            order,
            item,
            aligned_correct: false,
            random_correct: false,
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
    pub fn count(&self, f: impl Fn(&Quad) -> bool) -> usize {
        self.paired.iter().filter(|q| f(q)).count()
    }
    pub fn mean(&self, f: impl Fn(&Quad) -> f32) -> f32 {
        self.paired.iter().map(f).sum::<f32>() / self.n().max(1) as f32
    }
    /// `(accuracy disagreements, bit-identical NLLs, max |ΔNLL|, pass)` for
    /// the fuse zero-payload no-op diagnostic.
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
                    "aligned_correct": s.aligned_correct, "random_correct": s.random_correct,
                    "discordant": s.discordant, "x": s.x, "wealth": s.wealth,
                })
            })
            .collect()
    }
}

/// The console summary the operator reads while the draw finishes. Every
/// number here is also a stored receipt field; this is a view, not a source.
pub fn print_summary(rank: usize, n_max: usize, o: &DrawOutcome, r: &serde_json::Value) {
    let (n, n_disc) = (o.n(), o.n_disc());
    println!(
        "\nM5[receiver-adapted r{rank}/question-tail/fuse/de-pooled]: e-process {} — items drawn \
         {}, W_final {:.4} (max {:.4}, threshold {}), n_disc {n_disc} ({}W/{}L)",
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
        "registered bar: >= {REGISTERED_BAR_WINS} of {REGISTERED_BAR_OF} discordant wins => \
         crossed = {}; uninformative (n_disc < {UNINFORMATIVE_BELOW_N_DISC}) = {}",
        o.wins >= REGISTERED_BAR_WINS,
        o.uninformative()
    );
    println!(
        "accuracy: aligned {}/{n} baseline {}/{n} zerovec {}/{n} random {}/{n}",
        o.count(|q| q.real.0),
        o.count(|q| q.base.0),
        o.count(|q| q.zero.0),
        o.count(|q| q.rand.0)
    );
    println!(
        "NLL means: aligned {:.4} baseline {:.4} zerovec {:.4} random {:.4}",
        o.mean(|q| q.real.1),
        o.mean(|q| q.base.1),
        o.mean(|q| q.zero.1),
        o.mean(|q| q.rand.1)
    );
    for key in [
        "aligned_real_vs_baseline_uninjected",
        "aligned_real_vs_zerovec_injected",
        "aligned_real_vs_random",
        "random_vs_baseline_uninjected",
        "random_vs_zerovec_injected",
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
        "inversion accounting: aligned NLL worse than baseline on {}/{n}, than zerovec on {}/{n}, \
         than random on {}/{n}",
        o.count(|q| q.real.1 > q.base.1),
        o.count(|q| q.real.1 > q.zero.1),
        o.count(|q| q.real.1 > q.rand.1)
    );
    println!(
        "PROTOCOL IDENTITY: ADR-036 e-process on adaptation-512, RECEIVER-ADAPTED model. Not \
         comparable to any frozen-40-item-protocol result; no p-value translation is offered."
    );
}
