//! The federated unit of knowledge: a bounded, scoped transition rule with a
//! deterministic binary codec small enough for one Air frame. All multibyte
//! integers are big endian, matching the Air wire conventions.

use serde::{Deserialize, Serialize};

/// Hard cap on one encoded rule — comfortably inside a single 240-byte Air
/// frame payload after envelope overhead.
pub const MAX_RULE_BYTES: usize = 200;

const RULE_MAGIC: &[u8; 4] = b"LMFR";
const RULE_VERSION: u8 = 1;
/// magic + version + scope + cluster + support + confidence(milli) + 3 label lengths
const FIXED_HEADER_BYTES: usize = 4 + 1 + 1 + 4 + 4 + 2 + 3;
const MAX_LABEL_BYTES: usize = 48;

/// FedWorld's knowledge classes (ADR-007): who a rule may be shared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleScope {
    /// Valid everywhere; freely federated.
    Global,
    /// Valid within one named cluster; offered only to same-cluster peers.
    Cluster(u32),
    /// Never leaves the node. The transmit encoder refuses it.
    Private,
    /// Not yet classified; transmissible but flagged for the receiver's
    /// validation to decide.
    Unresolved,
}

impl RuleScope {
    fn code(self) -> u8 {
        match self {
            RuleScope::Global => 0,
            RuleScope::Cluster(_) => 1,
            RuleScope::Private => 2,
            RuleScope::Unresolved => 3,
        }
    }

    fn cluster_id(self) -> u32 {
        match self {
            RuleScope::Cluster(id) => id,
            _ => 0,
        }
    }
}

/// Typed federation failures.
#[derive(Clone, Debug, PartialEq)]
pub enum FederationError {
    /// Encoding a `Private` rule for transmission is refused by construction.
    PrivateRule,
    /// A label exceeded [`MAX_LABEL_BYTES`] or the rule exceeded
    /// [`MAX_RULE_BYTES`].
    TooLarge(usize),
    /// The byte input did not parse as a rule.
    Malformed(&'static str),
    /// Air-layer failure (envelope, CRC, replay), carried through.
    Air(latentmesh_air_core::AirError),
}

impl core::fmt::Display for FederationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FederationError::PrivateRule => {
                write!(f, "private-scoped rules never leave the node")
            }
            FederationError::TooLarge(n) => write!(f, "rule encoding of {n} bytes exceeds bound"),
            FederationError::Malformed(why) => write!(f, "malformed rule bytes: {why}"),
            FederationError::Air(e) => write!(f, "air transport error: {e}"),
        }
    }
}

impl std::error::Error for FederationError {}

impl From<latentmesh_air_core::AirError> for FederationError {
    fn from(e: latentmesh_air_core::AirError) -> Self {
        FederationError::Air(e)
    }
}

/// One learned dynamics fact: `(pre, action) → post`, with the evidence
/// weight (`support` observations) and the node's own confidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionRule {
    pub pre: String,
    pub action: String,
    pub post: String,
    /// Number of local observations supporting the rule.
    pub support: u32,
    /// Sender-side confidence in `[0, 1]`.
    pub confidence: f32,
    pub scope: RuleScope,
}

impl TransitionRule {
    /// Deterministic bounded encoding. Refuses `Private` scope — that is the
    /// structural privacy guarantee, not a courtesy check.
    pub fn encode_for_transmission(&self) -> Result<Vec<u8>, FederationError> {
        if self.scope == RuleScope::Private {
            return Err(FederationError::PrivateRule);
        }
        let labels = [
            self.pre.as_bytes(),
            self.action.as_bytes(),
            self.post.as_bytes(),
        ];
        for label in labels {
            if label.len() > MAX_LABEL_BYTES {
                return Err(FederationError::TooLarge(label.len()));
            }
        }
        let total = FIXED_HEADER_BYTES + labels.iter().map(|l| l.len()).sum::<usize>();
        if total > MAX_RULE_BYTES {
            return Err(FederationError::TooLarge(total));
        }
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(RULE_MAGIC);
        out.push(RULE_VERSION);
        out.push(self.scope.code());
        out.extend_from_slice(&self.scope.cluster_id().to_be_bytes());
        out.extend_from_slice(&self.support.to_be_bytes());
        let confidence_milli = (self.confidence.clamp(0.0, 1.0) * 1000.0).round() as u16;
        out.extend_from_slice(&confidence_milli.to_be_bytes());
        for label in labels {
            out.push(label.len() as u8);
        }
        for label in labels {
            out.extend_from_slice(label);
        }
        Ok(out)
    }

    /// Decode with full validation; every malformed shape is a typed error.
    pub fn decode(input: &[u8]) -> Result<Self, FederationError> {
        if input.len() > MAX_RULE_BYTES {
            return Err(FederationError::TooLarge(input.len()));
        }
        if input.len() < FIXED_HEADER_BYTES {
            return Err(FederationError::Malformed("truncated header"));
        }
        if &input[0..4] != RULE_MAGIC {
            return Err(FederationError::Malformed("bad magic"));
        }
        if input[4] != RULE_VERSION {
            return Err(FederationError::Malformed("unsupported version"));
        }
        let cluster = u32::from_be_bytes([input[6], input[7], input[8], input[9]]);
        let scope = match input[5] {
            0 => RuleScope::Global,
            1 => RuleScope::Cluster(cluster),
            2 => return Err(FederationError::Malformed("private rule on the wire")),
            3 => RuleScope::Unresolved,
            _ => return Err(FederationError::Malformed("unknown scope")),
        };
        let support = u32::from_be_bytes([input[10], input[11], input[12], input[13]]);
        let confidence_milli = u16::from_be_bytes([input[14], input[15]]);
        if confidence_milli > 1000 {
            return Err(FederationError::Malformed("confidence above 1.0"));
        }
        let lens = [input[16] as usize, input[17] as usize, input[18] as usize];
        for len in lens {
            if len > MAX_LABEL_BYTES {
                return Err(FederationError::Malformed("label too long"));
            }
        }
        let expected = FIXED_HEADER_BYTES + lens.iter().sum::<usize>();
        if input.len() != expected {
            return Err(FederationError::Malformed("length mismatch"));
        }
        let mut cursor = FIXED_HEADER_BYTES;
        let mut labels = Vec::with_capacity(3);
        for len in lens {
            let bytes = &input[cursor..cursor + len];
            let s = core::str::from_utf8(bytes)
                .map_err(|_| FederationError::Malformed("label not utf-8"))?;
            labels.push(s.to_string());
            cursor += len;
        }
        let post = labels.pop().unwrap_or_default();
        let action = labels.pop().unwrap_or_default();
        let pre = labels.pop().unwrap_or_default();
        Ok(TransitionRule {
            pre,
            action,
            post,
            support,
            confidence: f32::from(confidence_milli) / 1000.0,
            scope,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(scope: RuleScope) -> TransitionRule {
        TransitionRule {
            pre: "door_closed".into(),
            action: "push".into(),
            post: "door_open".into(),
            support: 17,
            confidence: 0.9,
            scope,
        }
    }

    #[test]
    fn round_trips_and_stays_bounded() {
        let r = rule(RuleScope::Cluster(7));
        let bytes = r.encode_for_transmission().unwrap();
        assert!(bytes.len() <= MAX_RULE_BYTES);
        assert_eq!(TransitionRule::decode(&bytes).unwrap(), r);
    }

    #[test]
    fn private_rules_are_refused_at_the_encoder() {
        assert_eq!(
            rule(RuleScope::Private).encode_for_transmission(),
            Err(FederationError::PrivateRule)
        );
    }

    #[test]
    fn a_forged_private_rule_on_the_wire_is_rejected_on_decode() {
        let mut bytes = rule(RuleScope::Global).encode_for_transmission().unwrap();
        bytes[5] = 2; // forge scope = Private
        assert!(matches!(
            TransitionRule::decode(&bytes),
            Err(FederationError::Malformed(_))
        ));
    }

    #[test]
    fn malformed_inputs_are_typed_errors_never_panics() {
        let good = rule(RuleScope::Global).encode_for_transmission().unwrap();
        for cut in 0..good.len() {
            let _ = TransitionRule::decode(&good[..cut]);
        }
        let mut wrong_len = good.clone();
        wrong_len.push(0);
        assert!(TransitionRule::decode(&wrong_len).is_err());
        let mut bad_conf = good;
        bad_conf[14] = 0xff;
        assert!(TransitionRule::decode(&bad_conf).is_err());
    }

    #[test]
    fn oversized_labels_are_refused() {
        let mut r = rule(RuleScope::Global);
        r.pre = "x".repeat(MAX_LABEL_BYTES + 1);
        assert!(matches!(
            r.encode_for_transmission(),
            Err(FederationError::TooLarge(_))
        ));
    }
}
