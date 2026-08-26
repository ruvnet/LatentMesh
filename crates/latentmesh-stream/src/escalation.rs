//! Confidence-gated authority escalation (ADR-004/ADR-015): a stream starts
//! at `ObserveOnly` and earns higher effective authority one rung at a time as
//! aggregate confidence accumulates — never above the frame's own declared
//! authority (which the gate has already checked against the edge ceiling),
//! never skipping a rung, and falling back when confidence drops or the
//! stream gaps.

use latentmesh_core::Authority;

/// Escalation policy. Thresholds are on an exponential moving average of
/// per-frame confidence, so a burst of low-confidence frames de-escalates
/// promptly while one noisy frame does not.
#[derive(Clone, Copy, Debug)]
pub struct EscalationConfig {
    /// EMA smoothing factor in `(0, 1]`; higher weighs recent frames more.
    pub ema_alpha: f32,
    /// Minimum EMA to hold `ContextInject`.
    pub context_inject_threshold: f32,
    /// Minimum EMA to hold `LatentPrefix`.
    pub latent_prefix_threshold: f32,
    /// Minimum EMA to hold `ActionInfluencing`.
    pub action_influencing_threshold: f32,
    /// Frames that must be accepted at the current rung before the next
    /// escalation is allowed (rate-limits the climb).
    pub min_frames_per_rung: u32,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        EscalationConfig {
            ema_alpha: 0.3,
            context_inject_threshold: 0.5,
            latent_prefix_threshold: 0.75,
            action_influencing_threshold: 0.9,
            min_frames_per_rung: 3,
        }
    }
}

impl EscalationConfig {
    fn threshold_for(&self, authority: Authority) -> f32 {
        match authority {
            Authority::ObserveOnly => 0.0,
            Authority::ContextInject => self.context_inject_threshold,
            Authority::LatentPrefix => self.latent_prefix_threshold,
            Authority::ActionInfluencing => self.action_influencing_threshold,
        }
    }
}

fn next_rung(authority: Authority) -> Option<Authority> {
    match authority {
        Authority::ObserveOnly => Some(Authority::ContextInject),
        Authority::ContextInject => Some(Authority::LatentPrefix),
        Authority::LatentPrefix => Some(Authority::ActionInfluencing),
        Authority::ActionInfluencing => None,
    }
}

fn previous_rung(authority: Authority) -> Option<Authority> {
    match authority {
        Authority::ObserveOnly => None,
        Authority::ContextInject => Some(Authority::ObserveOnly),
        Authority::LatentPrefix => Some(Authority::ContextInject),
        Authority::ActionInfluencing => Some(Authority::LatentPrefix),
    }
}

/// Per-stream escalation state.
#[derive(Clone, Debug)]
pub struct AuthorityEscalator {
    config: EscalationConfig,
    current: Authority,
    ema: f32,
    frames_at_rung: u32,
    seen_any: bool,
}

impl AuthorityEscalator {
    pub fn new(config: EscalationConfig) -> Self {
        AuthorityEscalator {
            config,
            current: Authority::ObserveOnly,
            ema: 0.0,
            frames_at_rung: 0,
            seen_any: false,
        }
    }

    /// The stream's current earned authority level.
    pub fn current(&self) -> Authority {
        self.current
    }

    /// Current confidence EMA in `[0, 1]`.
    pub fn confidence(&self) -> f32 {
        self.ema
    }

    /// A gap in the stream resets earned authority to `ObserveOnly`
    /// (ADR-015): missing partial state invalidates the evidence the climb
    /// was based on.
    pub fn reset(&mut self) {
        self.current = Authority::ObserveOnly;
        self.ema = 0.0;
        self.frames_at_rung = 0;
        self.seen_any = false;
    }

    /// Fold one accepted frame's confidence in and return the *effective*
    /// authority for that frame: `min(declared, earned)`.
    pub fn observe(&mut self, declared: Authority, frame_confidence: f32) -> Authority {
        let c = frame_confidence.clamp(0.0, 1.0);
        if self.seen_any {
            self.ema = self.config.ema_alpha * c + (1.0 - self.config.ema_alpha) * self.ema;
        } else {
            self.ema = c;
            self.seen_any = true;
        }
        self.frames_at_rung = self.frames_at_rung.saturating_add(1);

        // De-escalate first: drop rungs until the EMA supports the level.
        while self.ema < self.config.threshold_for(self.current) {
            match previous_rung(self.current) {
                Some(lower) => {
                    self.current = lower;
                    self.frames_at_rung = 0;
                }
                None => break,
            }
        }

        // Escalate at most one rung per observation, rate-limited.
        if self.frames_at_rung >= self.config.min_frames_per_rung {
            if let Some(higher) = next_rung(self.current) {
                if self.ema >= self.config.threshold_for(higher) {
                    self.current = higher;
                    self.frames_at_rung = 0;
                }
            }
        }

        self.current.min(declared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_observe_only_and_climbs_one_rung_at_a_time() {
        let mut esc = AuthorityEscalator::new(EscalationConfig::default());
        let mut levels = Vec::new();
        for _ in 0..12 {
            levels.push(esc.observe(Authority::ActionInfluencing, 0.95));
        }
        assert_eq!(levels[0], Authority::ObserveOnly);
        // Reaches the top eventually, but never skips a rung.
        assert_eq!(*levels.last().unwrap(), Authority::ActionInfluencing);
        for pair in levels.windows(2) {
            let step = (pair[1] as u8) as i16 - (pair[0] as u8) as i16;
            assert!(step <= 1, "escalated more than one rung: {pair:?}");
        }
    }

    #[test]
    fn never_exceeds_the_frames_declared_authority() {
        let mut esc = AuthorityEscalator::new(EscalationConfig::default());
        for _ in 0..20 {
            let effective = esc.observe(Authority::ContextInject, 1.0);
            assert!(effective <= Authority::ContextInject);
        }
    }

    #[test]
    fn low_confidence_de_escalates_and_gap_resets() {
        let mut esc = AuthorityEscalator::new(EscalationConfig::default());
        for _ in 0..12 {
            esc.observe(Authority::ActionInfluencing, 0.95);
        }
        assert_eq!(esc.current(), Authority::ActionInfluencing);
        for _ in 0..8 {
            esc.observe(Authority::ActionInfluencing, 0.1);
        }
        assert!(esc.current() < Authority::ActionInfluencing);
        esc.reset();
        assert_eq!(esc.current(), Authority::ObserveOnly);
        assert_eq!(esc.confidence(), 0.0);
    }
}
