//! The per-node world model `W_i : (state, action) → state'` and the held-out
//! transition log its admission decisions are validated against.

use crate::rule::TransitionRule;
use std::collections::BTreeMap;

/// One observed transition, used both to learn local rules and as held-out
/// evidence when validating a candidate rule from a peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition {
    pub pre: String,
    pub action: String,
    pub post: String,
}

/// A node's local dynamics table plus its held-out log.
#[derive(Clone, Debug, Default)]
pub struct WorldModel {
    rules: BTreeMap<(String, String), TransitionRule>,
    holdout: Vec<Transition>,
}

impl WorldModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install (or replace) a rule keyed by `(pre, action)`.
    pub fn install(&mut self, rule: TransitionRule) {
        self.rules
            .insert((rule.pre.clone(), rule.action.clone()), rule);
    }

    pub fn remove(&mut self, pre: &str, action: &str) -> Option<TransitionRule> {
        self.rules.remove(&(pre.to_string(), action.to_string()))
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn rules(&self) -> impl Iterator<Item = &TransitionRule> {
        self.rules.values()
    }

    /// Predict the post-state for `(pre, action)`, if a rule covers it.
    pub fn predict(&self, pre: &str, action: &str) -> Option<&str> {
        self.rules
            .get(&(pre.to_string(), action.to_string()))
            .map(|r| r.post.as_str())
    }

    /// Append held-out transitions (never used to learn, only to validate).
    pub fn record_holdout(&mut self, transition: Transition) {
        self.holdout.push(transition);
    }

    pub fn holdout(&self) -> &[Transition] {
        &self.holdout
    }

    /// Per-transition prediction score over the held-out log with an
    /// optional extra candidate rule overlaid: 1.0 for a correct prediction,
    /// 0.0 for a wrong or absent one. Returned per-item so the admission
    /// test can run a paired statistical comparison.
    pub fn holdout_scores(&self, candidate: Option<&TransitionRule>) -> Vec<f64> {
        self.holdout
            .iter()
            .map(|t| {
                let predicted = match candidate {
                    Some(c) if c.pre == t.pre && c.action == t.action => Some(c.post.as_str()),
                    _ => self.predict(&t.pre, &t.action),
                };
                match predicted {
                    Some(p) if p == t.post => 1.0,
                    _ => 0.0,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::RuleScope;

    fn rule(pre: &str, action: &str, post: &str) -> TransitionRule {
        TransitionRule {
            pre: pre.into(),
            action: action.into(),
            post: post.into(),
            support: 10,
            confidence: 0.9,
            scope: RuleScope::Global,
        }
    }

    #[test]
    fn predicts_installed_rules_and_scores_holdout() {
        let mut w = WorldModel::new();
        w.install(rule("a", "go", "b"));
        w.record_holdout(Transition {
            pre: "a".into(),
            action: "go".into(),
            post: "b".into(),
        });
        w.record_holdout(Transition {
            pre: "b".into(),
            action: "go".into(),
            post: "c".into(),
        });
        assert_eq!(w.predict("a", "go"), Some("b"));
        assert_eq!(w.holdout_scores(None), vec![1.0, 0.0]);
        // A candidate covering the second transition lifts its score.
        let candidate = rule("b", "go", "c");
        assert_eq!(w.holdout_scores(Some(&candidate)), vec![1.0, 1.0]);
    }
}
