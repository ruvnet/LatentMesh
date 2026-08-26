//! The `LatentMemory` contract and the deterministic in-memory backend.
//! Recall is "query → cognitive state → continuation" (ADR-005): nearest
//! stored latents by cosine similarity, plus prototype/rule promotion and
//! procedural topology records. The RuVector backend (`ruvector` feature)
//! implements the same trait over `ruvector-core`'s HNSW index.

use crate::compress::compress_latent;
use crate::record::{Fidelity, SymbolicRule, TopologyRecord, TrajectoryRecord};
use latentmesh_core::Encoding;
use std::collections::BTreeMap;

/// Configuration shared by backends.
#[derive(Clone, Copy, Debug)]
pub struct MemoryConfig {
    /// Latent dimensionality every stored record must match.
    pub dimensions: usize,
    /// Minimum measured causal value (ADR-003 ΔV) required for the Raw tier.
    pub raw_tier_causal_floor: f64,
    /// Successful reuses at which a prototype may be promoted to a rule.
    pub rule_promotion_uses: u32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            dimensions: 64,
            raw_tier_causal_floor: 0.5,
            rule_promotion_uses: 3,
        }
    }
}

/// Typed failures — a malformed record is an error, never a panic.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    DimensionMismatch {
        got: usize,
        expected: usize,
    },
    DuplicateId(String),
    UnknownId(String),
    /// Raw tier refused: measured causal value below the floor.
    CausalValueBelowFloor {
        value: f64,
        floor: f64,
    },
    /// A fidelity change tried to move up the continuum.
    FidelityRegression {
        from: Fidelity,
        to: Fidelity,
    },
    /// Prototype promotion needs at least two members.
    NotEnoughMembers(usize),
    /// Rule promotion requires the configured reuse count.
    NotReusedEnough {
        uses: u32,
        required: u32,
    },
    Backend(String),
}

impl core::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MemoryError::DimensionMismatch { got, expected } => {
                write!(f, "latent has {got} dims, store expects {expected}")
            }
            MemoryError::DuplicateId(id) => write!(f, "record id already stored: {id}"),
            MemoryError::UnknownId(id) => write!(f, "no record with id {id}"),
            MemoryError::CausalValueBelowFloor { value, floor } => {
                write!(f, "causal value {value} below Raw-tier floor {floor}")
            }
            MemoryError::FidelityRegression { from, to } => {
                write!(
                    f,
                    "fidelity may only move down the continuum ({from:?} → {to:?})"
                )
            }
            MemoryError::NotEnoughMembers(n) => {
                write!(f, "prototype needs at least 2 members, got {n}")
            }
            MemoryError::NotReusedEnough { uses, required } => {
                write!(
                    f,
                    "rule promotion needs {required} uses, prototype has {uses}"
                )
            }
            MemoryError::Backend(why) => write!(f, "backend failure: {why}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// One recall hit.
#[derive(Clone, Debug)]
pub struct Recalled {
    pub record: TrajectoryRecord,
    /// Cosine similarity to the query in `[-1, 1]`.
    pub similarity: f32,
}

/// The ADR-016 contract every backend implements.
pub trait LatentMemory {
    /// Store a trajectory. Raw-tier admission enforces the causal floor;
    /// a record that misses it must be compressed first (or stored at
    /// `Compressed` by the caller via [`LatentMemory::store_compressed`]).
    fn store(&mut self, record: TrajectoryRecord) -> Result<(), MemoryError>;

    /// Convenience: quantize to `Int8`, record the measured error, and store
    /// at `Compressed` — the admission path for trajectories that don't earn
    /// the Raw tier.
    fn store_compressed(&mut self, record: TrajectoryRecord) -> Result<(), MemoryError>;

    /// Nearest stored trajectories to `query` by cosine similarity.
    fn recall(&self, query: &[f32], k: usize) -> Result<Vec<Recalled>, MemoryError>;

    /// Re-encode an existing record at a lower fidelity, accumulating the
    /// measured reconstruction error. Upward moves are refused.
    fn compress_to(&mut self, id: &str, target: Fidelity) -> Result<f32, MemoryError>;

    /// Fold `member_ids` into a centroid prototype stored under
    /// `prototype_id`, preserving lineage in `parents`.
    fn promote_to_prototype(
        &mut self,
        prototype_id: &str,
        family: &str,
        member_ids: &[&str],
    ) -> Result<(), MemoryError>;

    /// Number of stored trajectory records.
    fn len(&self) -> usize;

    /// True when no trajectory records are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Cosine similarity; 0 for zero-norm inputs.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

pub(crate) fn admit_fidelity(
    record: &TrajectoryRecord,
    config: &MemoryConfig,
) -> Result<(), MemoryError> {
    if record.latent.len() != config.dimensions {
        return Err(MemoryError::DimensionMismatch {
            got: record.latent.len(),
            expected: config.dimensions,
        });
    }
    if record.fidelity == Fidelity::Raw && record.causal_value < config.raw_tier_causal_floor {
        return Err(MemoryError::CausalValueBelowFloor {
            value: record.causal_value,
            floor: config.raw_tier_causal_floor,
        });
    }
    Ok(())
}

/// Deterministic brute-force backend: exact cosine ranking, insertion-order
/// tiebreak, zero extra dependencies.
pub struct InMemoryStore {
    config: MemoryConfig,
    records: BTreeMap<String, TrajectoryRecord>,
    insertion_order: Vec<String>,
    topologies: Vec<TopologyRecord>,
    rules: Vec<SymbolicRule>,
}

impl InMemoryStore {
    pub fn new(config: MemoryConfig) -> Self {
        InMemoryStore {
            config,
            records: BTreeMap::new(),
            insertion_order: Vec::new(),
            topologies: Vec::new(),
            rules: Vec::new(),
        }
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    pub fn get(&self, id: &str) -> Option<&TrajectoryRecord> {
        self.records.get(id)
    }

    /// Store a successful topology (procedural memory, ADR-005 §skills).
    pub fn store_topology(&mut self, record: TopologyRecord) {
        self.topologies.push(record);
    }

    /// Warm-start lookup for ADR-006: the most-reused topology whose family
    /// label matches.
    pub fn recall_topology(&self, family: &str) -> Option<&TopologyRecord> {
        self.topologies
            .iter()
            .filter(|t| t.family == family)
            .max_by_key(|t| t.successful_uses)
    }

    /// Record a successful reuse of a stored topology.
    pub fn record_topology_use(&mut self, family: &str) {
        if let Some(t) = self
            .topologies
            .iter_mut()
            .filter(|t| t.family == family)
            .max_by_key(|t| t.successful_uses)
        {
            t.successful_uses = t.successful_uses.saturating_add(1);
        }
    }

    /// Promote a sufficiently-reused prototype to a symbolic rule (the
    /// cheapest tier — no vector lookup needed afterwards).
    pub fn promote_to_rule(
        &mut self,
        prototype_id: &str,
        uses: u32,
        rule_text: &str,
    ) -> Result<(), MemoryError> {
        let proto = self
            .records
            .get_mut(prototype_id)
            .ok_or_else(|| MemoryError::UnknownId(prototype_id.into()))?;
        if uses < self.config.rule_promotion_uses {
            return Err(MemoryError::NotReusedEnough {
                uses,
                required: self.config.rule_promotion_uses,
            });
        }
        proto.fidelity = Fidelity::Rule;
        self.rules.push(SymbolicRule {
            family: proto.action.clone(),
            rule: rule_text.into(),
            promoted_from: prototype_id.into(),
            uses,
        });
        Ok(())
    }

    /// Rules are retrievable without the vector index at all.
    pub fn rules(&self) -> &[SymbolicRule] {
        &self.rules
    }
}

impl LatentMemory for InMemoryStore {
    fn store(&mut self, record: TrajectoryRecord) -> Result<(), MemoryError> {
        admit_fidelity(&record, &self.config)?;
        if self.records.contains_key(&record.id) {
            return Err(MemoryError::DuplicateId(record.id));
        }
        self.insertion_order.push(record.id.clone());
        self.records.insert(record.id.clone(), record);
        Ok(())
    }

    fn store_compressed(&mut self, mut record: TrajectoryRecord) -> Result<(), MemoryError> {
        let outcome = compress_latent(&record.latent, Encoding::Int8);
        record.latent = outcome.latent;
        record.reconstruction_error += outcome.mean_abs_error;
        record.fidelity = Fidelity::Compressed;
        self.store(record)
    }

    fn recall(&self, query: &[f32], k: usize) -> Result<Vec<Recalled>, MemoryError> {
        if query.len() != self.config.dimensions {
            return Err(MemoryError::DimensionMismatch {
                got: query.len(),
                expected: self.config.dimensions,
            });
        }
        let mut hits: Vec<Recalled> = self
            .insertion_order
            .iter()
            .filter_map(|id| self.records.get(id))
            .map(|r| Recalled {
                similarity: cosine(query, &r.latent),
                record: r.clone(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    fn compress_to(&mut self, id: &str, target: Fidelity) -> Result<f32, MemoryError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| MemoryError::UnknownId(id.into()))?;
        if target <= record.fidelity {
            return Err(MemoryError::FidelityRegression {
                from: record.fidelity,
                to: target,
            });
        }
        let encoding = match target {
            Fidelity::Compressed => Encoding::F16,
            _ => Encoding::Int8,
        };
        let outcome = compress_latent(&record.latent, encoding);
        record.latent = outcome.latent;
        record.reconstruction_error += outcome.mean_abs_error;
        record.fidelity = target;
        Ok(record.reconstruction_error)
    }

    fn promote_to_prototype(
        &mut self,
        prototype_id: &str,
        family: &str,
        member_ids: &[&str],
    ) -> Result<(), MemoryError> {
        if member_ids.len() < 2 {
            return Err(MemoryError::NotEnoughMembers(member_ids.len()));
        }
        let mut centroid = vec![0.0f32; self.config.dimensions];
        let mut reward = 0.0f32;
        let mut causal = 0.0f64;
        let mut parents = Vec::with_capacity(member_ids.len());
        for id in member_ids {
            let member = self
                .records
                .get(*id)
                .ok_or_else(|| MemoryError::UnknownId((*id).into()))?;
            for (c, z) in centroid.iter_mut().zip(member.latent.iter()) {
                *c += z;
            }
            reward += member.reward;
            causal += member.causal_value;
            parents.push((*id).to_string());
        }
        let n = member_ids.len() as f32;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        let prototype = TrajectoryRecord {
            id: prototype_id.into(),
            latent: centroid,
            reward: reward / n,
            context_hash: format!("prototype:{family}"),
            action: family.into(),
            outcome: "prototype".into(),
            causal_value: causal / member_ids.len() as f64,
            fidelity: Fidelity::Prototype,
            reconstruction_error: 0.0,
            parents,
        };
        self.store(prototype)
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}
