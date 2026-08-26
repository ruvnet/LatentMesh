//! The RuVector backend (ADR-016): the same [`LatentMemory`] contract over
//! `ruvector-core`'s `VectorDB` with an HNSW index — the live wiring ADR-005
//! named as its integration target. Trajectory metadata rides in the entry's
//! metadata map, so recall returns full records. HNSW is approximate: tests
//! assert set-level recall, not exact ordering (the deterministic
//! `InMemoryStore` covers ordering-sensitive properties).

use crate::compress::compress_latent;
use crate::record::{Fidelity, TrajectoryRecord};
use crate::store::{admit_fidelity, LatentMemory, MemoryConfig, MemoryError, Recalled};
use latentmesh_core::Encoding;
use ruvector_core::types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::vector_db::VectorDB;
use std::collections::HashMap;

const METADATA_KEY: &str = "latentmesh_record";

fn backend_err<E: core::fmt::Display>(e: E) -> MemoryError {
    MemoryError::Backend(e.to_string())
}

/// `ruvector-core`-backed store. In-memory HNSW (no storage path) — a
/// persistent path is a `DbOptions` change, not an API change.
pub struct RuVectorStore {
    config: MemoryConfig,
    db: VectorDB,
}

impl RuVectorStore {
    pub fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let db = VectorDB::new(DbOptions {
            dimensions: config.dimensions,
            distance_metric: DistanceMetric::Cosine,
            storage_path: String::new(),
            hnsw_config: Some(HnswConfig::default()),
            quantization: None,
        })
        .map_err(backend_err)?;
        Ok(RuVectorStore { config, db })
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    fn entry_for(record: &TrajectoryRecord) -> Result<VectorEntry, MemoryError> {
        let json = serde_json::to_value(record).map_err(backend_err)?;
        let mut metadata = HashMap::new();
        metadata.insert(METADATA_KEY.to_string(), json);
        Ok(VectorEntry {
            id: Some(record.id.clone()),
            vector: record.latent.clone(),
            metadata: Some(metadata),
        })
    }

    fn record_from_metadata(
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<TrajectoryRecord, MemoryError> {
        let value = metadata
            .and_then(|m| m.get(METADATA_KEY))
            .ok_or_else(|| MemoryError::Backend("hit missing trajectory metadata".into()))?;
        serde_json::from_value(value.clone()).map_err(backend_err)
    }

    fn get_record(&self, id: &str) -> Result<TrajectoryRecord, MemoryError> {
        let entry = self
            .db
            .get(id)
            .map_err(backend_err)?
            .ok_or_else(|| MemoryError::UnknownId(id.into()))?;
        Self::record_from_metadata(entry.metadata.as_ref())
    }

    fn replace(&mut self, record: &TrajectoryRecord) -> Result<(), MemoryError> {
        self.db.delete(&record.id).map_err(backend_err)?;
        self.db
            .insert(Self::entry_for(record)?)
            .map_err(backend_err)?;
        Ok(())
    }
}

impl LatentMemory for RuVectorStore {
    fn store(&mut self, record: TrajectoryRecord) -> Result<(), MemoryError> {
        admit_fidelity(&record, &self.config)?;
        if self.db.get(&record.id).map_err(backend_err)?.is_some() {
            return Err(MemoryError::DuplicateId(record.id));
        }
        self.db
            .insert(Self::entry_for(&record)?)
            .map_err(backend_err)?;
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
        let hits = self
            .db
            .search(SearchQuery {
                vector: query.to_vec(),
                k,
                filter: None,
                ef_search: None,
            })
            .map_err(backend_err)?;
        hits.into_iter()
            .map(|hit| {
                let record = Self::record_from_metadata(hit.metadata.as_ref())?;
                // Cosine *distance* from ruvector → similarity.
                Ok(Recalled {
                    similarity: 1.0 - hit.score,
                    record,
                })
            })
            .collect()
    }

    fn compress_to(&mut self, id: &str, target: Fidelity) -> Result<f32, MemoryError> {
        let mut record = self.get_record(id)?;
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
        let err = record.reconstruction_error;
        self.replace(&record)?;
        Ok(err)
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
            let member = self.get_record(id)?;
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
        self.store(TrajectoryRecord {
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
        })
    }

    fn len(&self) -> usize {
        self.db.len().unwrap_or(0)
    }
}
