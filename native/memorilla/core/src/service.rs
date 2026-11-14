use crate::{MemoryCache, MemoryModule, Result, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub struct ReadResult {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub checkpoint_hash: String,
    pub collection_hash: String,
    pub query_hash: String,
    pub scope_hash: String,
}
struct Collection {
    cache: MemoryCache,
    hash: String,
}
pub struct MemoryService {
    pub model: MemoryModule,
    collections: HashMap<String, Collection>,
    scope_hash: String,
    budget: usize,
    used: usize,
}
pub fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        candle_core::bail!("nonfinite embedding value");
    }
    Ok(())
}
fn digest(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
impl MemoryService {
    pub fn new(model: MemoryModule, scope: &str, budget: usize) -> Result<Self> {
        if scope.is_empty() || scope.len() > 1024 || budget == 0 {
            candle_core::bail!("nonempty scope and positive cache budget required");
        }
        Ok(Self {
            model,
            collections: HashMap::new(),
            scope_hash: format!("{:x}", Sha256::digest(scope.as_bytes())),
            budget,
            used: 0,
        })
    }
    pub fn write(&mut self, id: &str, documents: &[f32]) -> Result<usize> {
        let d = self.model.config().embedding_dim;
        if id.is_empty()
            || id.len() > 1024
            || documents.is_empty()
            || !documents.len().is_multiple_of(d)
        {
            candle_core::bail!("invalid collection id or embedding dimensions");
        }
        finite(documents)?;
        let bytes = documents
            .len()
            .checked_mul(
                2 * self.model.config().num_cross_attn_layers * self.model.dtype().size_in_bytes(),
            )
            .ok_or_else(|| candle_core::Error::Msg("cache size overflow".into()))?;
        let old = self.collections.get(id).map_or(0, |c| c.cache.nbytes);
        let next = self
            .used
            .checked_sub(old)
            .and_then(|n| n.checked_add(bytes))
            .ok_or_else(|| candle_core::Error::Msg("cache budget overflow".into()))?;
        if next > self.budget {
            candle_core::bail!("cache budget exceeded: {next} > {}", self.budget);
        }
        let tensor =
            Tensor::from_slice(documents, (1, documents.len() / d, d), self.model.device())?;
        let cache = self.model.write_cache(&tensor, None)?;
        self.collections.insert(
            id.into(),
            Collection {
                cache,
                hash: digest(documents),
            },
        );
        self.used = next;
        Ok(bytes)
    }
    pub fn read(&self, id: &str, questions: &[f32]) -> Result<ReadResult> {
        finite(questions)?;
        let d = self.model.config().embedding_dim;
        if questions.is_empty() || !questions.len().is_multiple_of(d) {
            candle_core::bail!("question dimension mismatch");
        }
        let collection = self
            .collections
            .get(id)
            .ok_or_else(|| candle_core::Error::Msg("unknown collection in this scope".into()))?;
        // Bound the three largest f32 attention temporaries before allocating.
        // Model weights and caller owned inputs are outside the cache budget.
        let workspace = (questions.len() / d)
            .checked_mul(self.model.config().num_heads)
            .and_then(|n| n.checked_mul(self.model.config().num_memories))
            .and_then(|n| n.checked_mul(collection.cache.documents))
            .and_then(|n| n.checked_mul(12))
            .ok_or_else(|| candle_core::Error::Msg("attention workspace overflow".into()))?;
        if workspace > self.budget {
            candle_core::bail!("attention workspace exceeds budget; reduce the query batch");
        }
        let questions_tensor =
            Tensor::from_slice(questions, (questions.len() / d, d), self.model.device())?;
        let tokens = self
            .model
            .read_cache(&collection.cache, &questions_tensor)?;
        let values = tokens
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        finite(&values)?;
        Ok(ReadResult {
            values,
            shape: tokens.dims().to_vec(),
            checkpoint_hash: self.model.checkpoint_hash().to_owned(),
            collection_hash: collection.hash.clone(),
            query_hash: digest(questions),
            scope_hash: self.scope_hash.clone(),
        })
    }
    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(collection) = self.collections.remove(id) {
            self.used -= collection.cache.nbytes;
            true
        } else {
            false
        }
    }
    pub fn cache_bytes(&self) -> usize {
        self.used
    }
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }
}
